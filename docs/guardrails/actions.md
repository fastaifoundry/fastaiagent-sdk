# What a failure costs — actions, severity and floor

A guardrail answers three independent questions. Until 1.57.0 the SDK could only
answer two of them, so every failure meant the same thing: stop.

| Axis | Field | Question it answers |
|---|---|---|
| Scheduling | `blocking` (`validation_mode` on the plane) | Does this run inline, and can it halt? |
| Degradation | `on_error` | What does an *un-runnable* check mean? |
| **Consequence** | **`action`** | **What does a genuine failure cost?** |

Keep them apart when you reason about a rule. `blocking=False` means "run
concurrently, don't wait for the verdict". `action="warn"` is a different thing:
the check still runs inline and the caller still waits — it just doesn't stop.

## The five actions

```python
import fastaiagent as fa
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType

Guardrail(
    name="mask-ssn",
    guardrail_type=GuardrailType.regex,
    position=GuardrailPosition.output,
    config={"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "mask_token": "[REDACTED]"},
    action="mask",
)
```

| `action` | What happens on a failure |
|---|---|
| `block` | Raise `GuardrailBlockedError`. The default, and what every rule authored before wire v1.9 does. |
| `warn` | Record the failure and continue. The verdict is still a failure — the evidence stays honest — it just doesn't halt. |
| `mask` | Replace the offending spans and continue with the redacted value. Defined for `regex` and `classifier` only, the two types that locate the offending text. |
| `override` | Replace the payload with `config["override_message"]` (falling back to the rule's tripwire message) and continue. |
| `reask` | Re-prompt the model with the failure as feedback, bounded by `AgentConfig.guardrail_retries`. |

An action you didn't configure, or one this build has never heard of, becomes
`block`. For a safety control the strict reading is the safe one: an SDK that
meets an action it cannot perform must not quietly let the payload through.

## Two rules that are never negotiable

**An errored check always blocks.** When the check itself could not run, nothing
is known about the payload — there is nothing to mask and nothing to warn about
with confidence. `on_error` still decides whether the error counts as a failure
at all; the action spectrum only speaks about failures.

**A mask that finds nothing to mask blocks.** If the rule fails because a
pattern is *absent* (`should_match=True`), there is no span to redact. Rather
than passing the payload through untouched, the run stops.

Both fall out of one design choice: every caller branches on what the action
**actually did** (`GuardrailResult.action_taken`), never on what it was
configured to do.

```python
result = guardrail.execute("call me on 123-45-6789")
result.action        # "mask"     — configured
result.action_taken  # "masked"   — what happened
result.modified_data # "call me on [REDACTED]"
```

`action_taken` is one of `none` (a clean pass), `blocked`, `warned`, `masked`,
`overridden` or `reask` — the same six strings the plane records in
`guardrail_executions.result_detail`.

## Where a rewrite can and cannot be applied

`mask` and `override` change the payload, so `execute_guardrails` returns a
`GuardrailOutcome` carrying the value to carry forward:

```python
outcome = await execute_guardrails(rails, text, GuardrailPosition.output)
outcome.data      # the rewritten text, or the original when nothing rewrote it
outcome.modified  # did anything rewrite it?
outcome.results   # the verdicts, as before — the outcome iterates and indexes like the old list
```

Inside an `Agent` this is wired up for you. Where the rewrite cannot be applied
faithfully, the run **blocks** instead — passing the payload through untouched
would defeat the control:

| Position | Rewrite applies | Blocks instead when |
|---|---|---|
| `input` | The text reaching the model | The input is multimodal. The judged text is a *summary* of the parts, so substituting it would drop your images. |
| `output` (`run` / `arun`) | `AgentResult.output`, with `parsed` re-derived | — |
| `output` (`astream`) | — | **Always.** Every `TextDelta` has already been yielded, so a mask cannot un-emit the span it was meant to redact. If a rewriting rule matters to you, don't stream. |
| `tool_call` | The arguments handed to the tool | The rewrite no longer parses as an arguments object (an `override` replaces it with prose). |
| `tool_result` | The text the model is told | The tool returned image/PDF content parts the rewrite cannot cover. |

An **observe-only** rule (`blocking=False`) never halts and never rewrites,
whatever its action. It records what it would have done and the run carries on.

## `reask`

The one action the plane cannot perform — it runs no agent loop, so centrally it
records the intent and fails closed. At the edge it works:

```python
from fastaiagent.agent.agent import Agent, AgentConfig

agent = Agent(
    name="support",
    llm=llm,
    guardrails=[no_competitor_mentions],   # action="reask"
    config=AgentConfig(guardrail_retries=2),
)
```

The failure is fed back to the model as a correction turn. Exhausting the cap
**blocks** — a re-ask that never converges must not become a silent pass.
`guardrail_retries=0` makes a reask rule block outright. It defaults to `1`,
deliberately lower than `output_retries`: a guardrail re-ask is a full extra turn
on a reply that already cost one.

`reask` only means something where there is a model turn to redo. At every other
position, and when streaming, it blocks.

## Three model-backed check types

All three are judges with structure, distributed from the plane like any other rule.

### `content_safety`

Scores the payload against the [MLCommons hazard taxonomy](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard4/12B/MODEL_CARD.md)
(S1–S14) with **a bar per category** — the thing `llm_judge`'s PASS/FAIL cannot
express:

```json
{"categories": ["S1", "S3", "S4", "S10", "S11", "S12"], "threshold": 0.5, "thresholds": {"S10": 0.3}}
```

"Block hate at 0.3 but allow borderline specialised advice up to 0.8" is the
policy real operators write. With no `categories` it scores six common harms
rather than all fourteen: a judge asked about everything at once is longer,
costlier and measurably less reliable. The result's metadata carries the
per-category `scores`, the `thresholds` they were judged against, which ones
`tripped`, and any the judge declined to score (`unscored` — recorded, never
assumed absent).

An unparseable judge response raises, so `on_error` decides. Treating it as
all-zeros would turn a model outage into a silent pass.

### `groundedness`

Scores an answer against the context it was supposed to use:

```json
{"threshold": 0.7, "context_key": "context", "answer_key": "answer"}
```

This is the only rule that reads a **pair**, and an output guardrail only ever
receives the answer. Supply the context with a run-scoped slot:

```python
import fastaiagent as fa

docs = retriever.search(question)
with fa.guardrail_context(context=docs):
    reply = agent.run(question)
```

The slot is backed by a `ContextVar`, so it is per-task: concurrent runs never
see each other's context. `config.context_key` names the key the rule wants, so
the retrieval step and the rule never have to know about each other.

**A missing context fails closed.** An answer scored against nothing blocks
everything, and scored against itself blocks nothing; both are worse than
reporting that the rule could not run.

This is a different engine from the [`grounded()`](responsible-ai.md#grounded-vs-the-groundedness-type)
builtin, on purpose. `grounded()` and the `Faithfulness` eval scorer decompose an
answer into claims and verify each one — richer, N+1 model calls, offline-friendly.
The `groundedness` *type* uses the plane's single-call judge, because a
distributed rule has to reach the same verdict at the edge as it does at
`POST /guardrails/{id}/test`, and it has to be cheap enough to run every turn.

### `topic`

Classifies the payload against named topics, then applies a polarity:

```json
{
  "topics": [
    {"name": "Competitor products",
     "description": "Any mention, comparison or evaluation of a competing vendor's product."},
    {"name": "Medical advice",
     "description": "Diagnosis, treatment or medication guidance for a specific person."}
  ],
  "mode": "deny"
}
```

`mode: "deny"` fails when a listed topic is present (a blocklist); `mode: "allow"`
fails when **none** is (an on-topic gate). One type with a polarity rather than
two types, because they share a prompt, a parser and a config — an operator
choosing between two rule types when they mean one rule with a direction is a
worse console.

**The description is the point.** A bare label is a poor prompt: "crypto" cannot
tell a judge whether a mention of blockchain patents counts, and that is the
difference between topic control and a keyword list with extra steps. A bare
string is still accepted (the name doubles as the definition) because a console
may legitimately have no description yet, and judging a topic on its name alone
beats refusing the whole rule.

**No per-topic threshold.** `content_safety` has a bar per category because a
hazard score is a calibrated quantity. Topic presence is closer to a boolean, and
asking a judge "how much is this about medicine, 0 to 1" invites false precision
from a number nobody could tune. At most `MAX_TOPICS` (20) topics are judged: a
rule naming forty is describing a taxonomy, not a policy.

The metadata carries the `mode`, the `matched` topic names in your own wording,
and the full list of `topics` the rule asked about.

**`mode` does not decide what a failure costs, and it does not decide what an
error costs either.** A typo in `mode` *raises* rather than falling back to a
default — every other resolver tolerates a bad input, but this one inverts the
rule's meaning, and a whitelist silently read as a blocklist passes exactly the
traffic it was written to stop. What that failure costs is then `on_error`'s
call, not the polarity's: the two builtins deliberately disagree on the default
(see [`banned_topics()` vs the `topic` type](responsible-ai.md#banned_topics-allowed_topics-vs-the-topic-type)),
and a plane-distributed rule carries its own.

The prompt asks only *which topics are present* and never states the polarity.
Telling a judge that a topic is forbidden invites it to be helpful about the
verdict rather than accurate about the content, and it would make the two modes
classify identical text differently.

`mask` is refused on this type — a judge returns a verdict, not spans, so a mask
finds nothing to redact and degrades to a block. `block`, `warn`, `override` and
`reask` all apply.

## `severity` and `floor`

Neither changes enforcement.

`severity` is `low`, `medium`, `high`, `critical`, or unset. It is carried,
traced and shown; nothing branches on it.

`floor` marks the domain-wide baseline that only an admin may change. The plane
enforces that — it refuses to let a project narrow or switch off a floor rule. At
the edge it is worth showing, because "this is not your team's rule to argue
with" is useful context in a local run.

Both round-trip through `to_dict()` / `from_dict()` and land on the guardrail
span.

## What lands on the trace

Each guardrail run emits one `GUARDRAIL` child span carrying, alongside the
existing `name` / `position` / `passed` / `errored` / `checks`:

```
fastaiagent.guardrail.action        # block | warn | mask | override | reask
fastaiagent.guardrail.action_taken  # none | blocked | warned | masked | overridden | reask
fastaiagent.guardrail.severity      # low | medium | high | critical
fastaiagent.guardrail.floor         # bool
fastaiagent.guardrail.detail        # JSON: the check's own structured findings
```

The `checks` JSON keeps its three-value vocabulary (`pass` / `block` / `error`)
unchanged — the plane parses that string into `output.checks`.

### `detail`, and why it is an allowlist

`detail` is what lets an audit row say *which* topic tripped rather than only
that one did. Without it a rule enforced at the edge records less than the same
rule run centrally — the asymmetry a mirrored judge exists to prevent.

It is deliberately **not** the whole of a result's `metadata`. Most guardrail
metadata is derived from the payload: `toxic_words` holds the offending words,
`matches` a regex fragment — for a PII rule, the matched value itself —
`unsupported_claims` model output over customer content. None of that may leave
the machine as a side effect of reporting a verdict. So
`guardrail.executor.EXPORTABLE_DETAIL_KEYS` names, per type, the keys the SDK
runtime will send, and **a type absent from it exports nothing**:

| Type | Exported |
|---|---|
| `topic` | `mode`, `matched`, `topics` |
| `content_safety` | `taxonomy`, `scores`, `thresholds`, `tripped`, `unscored` |
| `groundedness` | `score`, `threshold`, `unsupported_claims` |
| everything else | nothing |

Most of those are payload-free by construction. `topic`'s `mode` is an enum, its
`topics` is the rule's own config — which a centrally-authored rule came *from* —
and its `matched` is intersected back against `topics` by `parse_topics`, so a
model cannot smuggle content into it. `content_safety` and `groundedness` export
scores, the bars they were judged against, and category codes: derived numbers
and rule config, no text.

!!! warning "`unsupported_claims` is the exception, and it is deliberate"
    It quotes the answer, so it **is** payload-derived. It is exported anyway,
    because the payload gate is the right control for it rather than exclusion:
    `FASTAIAGENT_TRACE_PAYLOADS=0` already means "no customer content leaves",
    and a deployment with payloads *on* has consented to span inputs and outputs
    — strictly more content than five clipped claims (`parse_verdict` caps the
    list at five). Excluding it would let an operator see the claims at
    `POST /guardrails/{id}/test` but not for the run that actually failed, which
    is the one worth debugging.

    Do **not** move it out from behind the payload gate on the grounds that its
    neighbours are safe. They are; it is not.

The attribute is in `SENSITIVE_ATTR_KEYS`, so `FASTAIAGENT_TRACE_PAYLOADS=0`
drops it and an installed redaction policy reaches it. For the safe keys that is
a backstop; for `unsupported_claims` it is the control itself. Local capture is
unaffected — the Local UI reads the full metadata either way.

Borrowing a runtime's own tracer? `emit_guardrail(..., detail=...)` takes
whatever you give it; the allowlist is the SDK runtime's own discipline, so
apply the same judgement to what you pass.

In the Local UI a rewriting rule shows as **`✎ filtered`** with a before/after
diff, distinct from a block.

## Foreign frameworks

The LangChain, CrewAI and PydanticAI wrappers own the verdict but not the payload
and not the loop. There, `warn` records and continues and everything else blocks,
saying why. Run the rule on a `fastaiagent` `Agent` to get the rewrite.

## Compatibility

Wire minor **v1.9** adds `action`, `severity` and `floor` to every rule in
`GET /public/v1/policy`, plus the two new `implementation_type` values. It is
purely additive:

- A rule with **no `action` key** — from a plane that predates v1.9 — reconstructs
  as `block`, exactly as it behaved before.
- A rule with an action this build does not know reconstructs as `block`.
- An SDK older than 1.57.0 ignores the new keys and skips the two new types
  rather than mis-enforcing them.

## Next steps

- [Concepts & Mental Model](concepts.md) — the execution model these three axes sit in
- [Managed governance](managed-governance.md) — how a rule reaches your process
