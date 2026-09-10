# Concepts & Mental Model

This page is the mental model for guardrails — *why* they exist, *where* they
fire in the agent run loop, *how* the executor runs them (blocking vs
non-blocking), and how the implementation types relate to the safety concerns
they cover. Read it first, then use the [Guardrails reference](index.md),
[Responsible AI](responsible-ai.md), and [Managed governance](managed-governance.md)
for depth.

## Why guardrails exist

An agent takes untrusted input, calls a model, and acts on the world through
tools. Each of those boundaries is a place something can go wrong: a prompt
injection in the input, PII or secrets in the output, an unsafe argument to a
tool. A **guardrail** is an *assertion* placed at one of those boundaries — it
inspects the data and either lets it pass or blocks the run.

!!! info "Guardrails assert; middleware transforms"
    A guardrail is pass/fail — it validates and can raise. [Middleware](../agents/middleware.md)
    *changes* the data flowing through (trim history, redact, rewrite). Use a
    guardrail for a policy check that should block on failure; use middleware
    when you want to modify what flows through the loop.

## The four positions

Guardrails attach at four positions — the boundaries the [agent run
loop](../agents/concepts.md#the-run-loop) crosses:

| Position | Fires | Guards against |
|----------|-------|----------------|
| `input` | Before the model sees the user input | Prompt injection, off-topic/abuse, disallowed requests |
| `tool_call` | On a tool's arguments, before it runs | Unsafe/destructive tool arguments |
| `tool_result` | On a tool's output, after it runs | Leaking sensitive data a tool returned |
| `output` | On the final answer, after the loop ends | PII, secrets, toxicity, hallucination, off-policy replies |

Think of a run as data flowing through up to four gates: `input` → (loop:
`tool_call` → `tool_result`, per tool call) → `output`. Positions are
independent — use any combination. You attach them all the same way:
`Agent(guardrails=[...])`, and each guardrail declares its own `position`.

## The execution model

At each position the executor (`fastaiagent/guardrail/executor.py`) runs the
applicable guardrails with a deliberate two-phase strategy:

1. **Blocking guardrails run first, sequentially.** The first one that fails
   raises `GuardrailBlockedError` immediately — the run stops and nothing after
   it executes (fail-fast).
2. **Non-blocking guardrails run after, in parallel** (`asyncio.gather`). A
   non-blocking failure is *recorded* but does **not** stop the run, and an
   exception inside one is caught and turned into a failed result rather than
   crashing the agent (fail-open).

!!! info "Verified against a live run"
    With one blocking + two non-blocking guardrails at `input`: on clean data
    the order was **blocking → then both non-blocking**, and a non-blocking
    "fail" was recorded without stopping the run. When the blocking guardrail
    failed, it raised `GuardrailBlockedError` and **the non-blocking guardrails
    never ran** — fail-fast, as designed.

So the mental model is: **blocking = a gate that can stop the run; non-blocking
= an observer that records but never blocks.** Set `blocking=True` for policy
you must enforce, `blocking=False` for signals you want to watch.

Concretely, `execute_guardrails(guardrails, data, position)` filters the list to
that position, then:

```python
for g in blocking:                        # sequential, fail-fast
    result = await g.aexecute(outcome.data)
    if halts(g, result):                  # ← what the action actually did
        raise GuardrailBlockedError(...)  # stops the run
results += await asyncio.gather(          # non-blocking, parallel
    *[g.aexecute(outcome.data) for g in non_blocking],
    return_exceptions=True,               # an exception → GuardrailResult(passed=False)
)
```

A blocking failure at **any** position raises `GuardrailBlockedError`, which
propagates out of the agent run — that's how `input`/`tool_call`/`tool_result`/`output`
all "stop" the run when they must.

### The third axis: what a failure costs

`blocking` decides *whether a rule runs inline and can halt*. It does not decide
what a failure is worth. That is `action`, a third and independent axis:

| Axis | Field | Question |
|---|---|---|
| Scheduling | `blocking` | Does this run inline, and can it halt? |
| Degradation | `on_error` | What does an *un-runnable* check mean? |
| Consequence | `action` | What does a genuine failure cost? |

`action` is one of `block` (the default), `warn`, `mask`, `override` or `reask`.
A `warn` rule still runs inline and the caller still waits for its verdict — it
just doesn't stop; a `mask` rule redacts the offending span and the run carries
on with the redacted value. Because `mask` and `override` change the payload,
`execute_guardrails` returns a `GuardrailOutcome` carrying both the verdicts and
the value to carry forward.

The rule that keeps this honest: every caller branches on what the action
**actually did** (`action_taken`), never on what it was configured to do. That
is why "an errored check always blocks" and "a mask with nothing to mask blocks"
need no special case anywhere.

See [Actions, severity & floor](actions.md) for the full contract, including
where a rewrite cannot be applied and the run blocks instead.

### The verdict object

Every guardrail — whatever its type — resolves to one
`GuardrailResult(passed, score, message, execution_time_ms, metadata, errored,
action, action_taken, modified_data)`. The executor branches on `passed` and
`action_taken`; `score`, `message` and `metadata` are for observability (the
Local UI reads them), and `modified_data` carries the rewritten payload when the
action produced one. For a `code` guardrail, your `fn` can
return a bare `bool` (coerced to `GuardrailResult(passed=...)`) or a full
`GuardrailResult`. A guardrail crash never crashes the agent — a raised
exception is caught and resolved according to the guardrail's `on_error` policy
(see below), with `errored=True` so a degraded result is never mistaken for a
real verdict.

### When the check itself fails: `on_error`

A model-judged guardrail (`toxicity_check(mode="llm")`, `grounded`, an
`llm_judge`, OpenAI moderation, …) depends on an LLM call that can *fail* —
a timeout, a 5xx, an unparseable response. That is different from the check
running and returning a verdict, and you get to decide what it means:

```python
toxicity_check(mode="llm", on_error="block")   # fail closed: an error blocks
toxicity_check(mode="llm", on_error="allow")   # fail open: an error passes through
```

- **`on_error="block"`** (fail closed) — an errored check is treated as a
  failure. Use it for policy you must enforce even when the checker is down (a
  bank would rather block than guess). This is the default for the guardrails
  that already behaved this way (`grounded`, `openai_moderation`, `allowed_topics`,
  and any `Guardrail(...)` / `llm_judge` you build yourself).
- **`on_error="allow"`** (fail open) — an errored check lets the content
  through. Use it when availability beats strictness (a high-traffic chatbot
  would rather serve than let a flaky moderation API take it offline). This is
  the default for the convenience classifiers that already behaved this way
  (`toxicity_check`, `no_prompt_injection`, `banned_topics`).

Either way the outcome is **visible**: the result carries `errored=True`, the
trace span records a `guardrail.errored` attribute and an `"error"` check
result, and the Local UI logs the event with an `errored` outcome — so you can
see exactly how often a guardrail is degrading instead of guarding. Guardrail
LLM calls also get a small automatic retry, so a single transient blip doesn't
trip the policy at all.

The `no_pii()`, `no_secrets()`, `json_valid()` and `allowed_domains()` builtins
don't make a fallible call, so `on_error` doesn't come up for them — they are
reliable hard blocks. The `pii` **type** is different: it raises on an unknown
entity, an unknown backend, or `backend="presidio"` without the `[safety]`
extra, and `on_error` decides what each costs. Detection that cannot run is not
detection that found nothing.

### How each type decides

`run_guardrail` dispatches on `GuardrailType` to ten deciders, all producing the
same `GuardrailResult`:

| Type | How it decides |
|------|----------------|
| `code` | Runs your Python `fn(data)` — arbitrary logic |
| `regex` | Matches a pattern; `match_type` flips whether a match means pass or fail |
| `schema` | Validates the data against a JSON Schema |
| `llm_judge` | Calls a model with a rubric and parses the verdict **fail-closed** (ambiguous → fail) |
| `classifier` | Calls a classification endpoint (e.g. a moderation model) and thresholds the score |
| `content_safety` | Scores the payload against the MLCommons hazard taxonomy, with a bar per category |
| `groundedness` | Scores an answer against the context it was given |
| `topic` | Classifies against named topics, then `deny` (blocklist) or `allow` (on-topic gate) |
| `pii` | Detects personal data with the shared detectors; can redact, not just refuse |
| `secrets` | Detects leaked credentials and tokens; can redact, not just refuse |

The last three are model-backed judges with structure — see
[Actions, severity & floor](actions.md#three-model-backed-check-types).

This is the mechanical basis for the two-axis view below: the *type* is which
decider runs; the *concern* is what you point it at.

## Two axes: implementation type × what it checks

A guardrail is described by two independent things — don't conflate them:

- **Implementation type** (`GuardrailType`) — *how* it decides:
  `code`, `regex`, `schema`, `llm_judge`, `classifier`, `content_safety`,
  `groundedness`, `topic`, `pii`, `secrets`.
- **What it checks** — the concern: prompt injection, PII, secrets, toxicity,
  groundedness, topic, moderation. The [Responsible AI](responsible-ai.md)
  bundle is a curated set of these, each implemented as an ordinary `Guardrail`.

The same concern can be implemented different ways, with a real cost trade-off:
`code`/`regex`/`schema` are free and instant; `llm_judge`/`classifier` and
LLM-backed safety checks cost an inference call but catch things patterns
can't. Reach for cheap deterministic checks first, LLM-backed ones where
nuance matters.

## Composition

- **Stack them** — put several guardrails on one agent; the executor groups
  them by position and applies the blocking/non-blocking rules per position.
- **Bundle them** — `responsible_ai(...)` returns a list of `Guardrail`s you
  spread into `guardrails=[...]`; see [Responsible AI](responsible-ai.md).
- **Govern them centrally** — a connected agent can defer high-stakes tool
  calls to a platform policy that can require human approval; see [Managed
  governance](managed-governance.md). That path *pauses* the run rather than
  simply passing/failing.

## What a guardrail run puts on the trace

Every guardrail that runs emits **one child span** — on a pass as well as a
block, so the console shows green checks and not only failures. The span is
classified with the OpenInference standard kind and carries the outcome in the
`fastaiagent.guardrail.*` namespace:

| Attribute | Meaning |
|-----------|---------|
| `openinference.span.kind` | Always `"GUARDRAIL"` — the standard classifier |
| `fastaiagent.guardrail.name` | The guardrail's name. How the platform resolves the span to a guardrail |
| `fastaiagent.guardrail.position` | `input` / `tool_call` / `tool_result` / `output` |
| `fastaiagent.guardrail.passed` | The verdict |
| `fastaiagent.guardrail.errored` | `true` when the check *couldn't run* and `passed` reflects `on_error`, not a real verdict |
| `fastaiagent.guardrail.checks` | JSON: `[{"name": ..., "result": "pass"｜"block"｜"error"}]` |
| `fastaiagent.guardrail.action` | What the rule was configured to cost: `block` / `warn` / `mask` / `override` / `reask` |
| `fastaiagent.guardrail.action_taken` | What it actually did: `none` / `blocked` / `warned` / `masked` / `overridden` / `reask` |
| `fastaiagent.guardrail.severity` | `low` / `medium` / `high` / `critical`, when the rule carries one |
| `fastaiagent.guardrail.floor` | `true` when the rule is the domain-wide baseline |

The split matters: **OpenInference standardizes the span *kind*, not the outcome
fields.** There is no ecosystem convention for "what did this guardrail decide",
so `fastaiagent.guardrail.*` is ours, and it rides *under* the standard kind.
Anything that understands OpenInference recognizes the span as a guardrail;
anything that understands FastAIAgent additionally reads the verdict.

The span status follows the verdict — `OK` on pass, `ERROR` on block — with one
deliberate exception: a *degraded pass* (`errored=true` with `on_error="allow"`)
keeps an `OK` status, because the run did continue. The `errored` attribute is
what tells a fail-open apart from a genuine pass.

!!! note "Legacy `span_type` marker"
    Guardrail spans also carry `span_type="guardrail"` alongside the standard
    kind. That dual-write exists for platform deployments predating the
    OpenInference reader and is transitional — don't build on it.

If you're enforcing guardrails from a runtime that isn't `fa.Agent`, you emit
this same span yourself with `fa.emit_guardrail(...)`. See [Guardrails and evals
without the runtime](../integrations/primitives-without-the-runtime.md).

## Next steps

- [Guardrails](index.md) — the full reference: all seven types, built-in factories, custom guardrails, serialization
- [Actions, severity & floor](actions.md) — what a failure costs, and the two model-backed check types
- [Guardrails & evals without the runtime](../integrations/primitives-without-the-runtime.md) — borrowing `run_guardrail` from a foreign framework
- [Responsible AI (Trust Layer)](responsible-ai.md) — the safety bundle by concern
- [Managed governance](managed-governance.md) — platform-enforced, approval-gated tool policy
- [Agents — the run loop](../agents/concepts.md#the-run-loop) — exactly where each position fires
