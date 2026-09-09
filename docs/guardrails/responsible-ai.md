# Responsible AI — the Trust Layer

The **Trust Layer** is a coherent set of runtime guardrails (plus one middleware)
that make an agent's I/O *truthful, safe, and on-policy* — the things an
enterprise review asks about before an agent ships:

| Concern | Surface | Cost |
| --- | --- | --- |
| Prompt injection / jailbreak | `no_prompt_injection()` | zero-dependency |
| PII leakage | `no_pii()` | zero-dependency |
| **Leaked secrets / credentials** | `no_secrets()` | zero-dependency |
| **Toxicity** | `toxicity_check()` / `toxicity_check(mode="llm")` | keyword free; LLM opt-in |
| **Groundedness / hallucination** | `grounded()` / `no_hallucination()` | one LLM check |
| **Topic controls** | `banned_topics()` / `allowed_topics()` | keyword free; LLM opt-in |
| Content moderation | `openai_moderation()` | OpenAI moderation API |
| **Self-critique / reflection** | `Reflect` middleware | one LLM call per answer |

Everything shares one detector core with the [eval scorers](../evaluation/safety-metrics.md)
— *one core detector, two surfaces* — so what you test for offline is exactly
what you enforce at runtime.

## One call: `responsible_ai()`

`responsible_ai()` composes a bundle you spread into `guardrails=[...]`. The
**zero-dependency** checks (prompt-injection on input, PII + secrets on output)
are on by default; the LLM-backed checks are opt-in, so the default bundle adds
**no** extra LLM calls.

```python
import fastaiagent as fa
from fastaiagent import Agent, LLMClient, responsible_ai

llm = LLMClient(provider="openai", model="gpt-4o-mini")

latest_context = ""  # set this from your retrieval each turn

agent = Agent(
    name="support",
    llm=llm,
    guardrails=responsible_ai(
        # defaults: prompt_injection=True, pii=True, secrets=True
        grounded_to=lambda: latest_context,   # block hallucinations vs your sources
        banned=["politics", "legal advice"],  # semantic topic blocklist
        toxicity=True,                         # LLM toxicity scoring
        llm=llm,
    ),
)
```

You can also assemble the pieces by hand — every item below is an ordinary
`Guardrail` (or middleware) you can use on its own.

## Groundedness — block hallucinations

`grounded()` verifies the **factual claims** in the output against a reference
(your source text), and blocks when too few are supported. It reuses the exact
engine behind the [`Faithfulness`](../evaluation/safety-metrics.md) eval scorer.

```python
from fastaiagent import grounded

# reference can be a string, or a zero-arg callable returning the latest
# retrieved context (evaluated at check time).
g = grounded(lambda: latest_context, llm=llm, threshold=0.7)
agent = Agent(name="rag", llm=llm, guardrails=[g])
```

`no_hallucination` is an alias for `grounded`.

!!! note "Why an explicit reference?"
    Output guardrails receive only the output text — not the agent's retrieved
    context. So you pass the reference here (often a `lambda` closing over your
    latest retrieval) rather than it being auto-wired.

### `grounded()` vs the `groundedness` type

Two engines, on purpose. Pick by where the rule is authored:

| | `grounded()` | `groundedness` type |
|---|---|---|
| Authored | In your code | In the console, distributed over `/policy` |
| Context from | A string or zero-arg callable | The run-scoped [`fa.guardrail_context(...)`](actions.md#groundedness) slot, or a `{context, answer}` payload |
| Engine | Claim decomposition — N+1 model calls, names each unsupported claim | One structured judge call returning `{score, unsupported}` |
| Missing context | Silently passes (the rule is opted out) | **Fails closed** |
| Shares its engine with | The [`Faithfulness`](../evaluation/safety-metrics.md) eval scorer | The plane, so a rule reaches the same verdict at the edge and at `POST /guardrails/{id}/test` |

A callable cannot be serialised into a policy rule, which is why the distributed
form needs the context slot and a cheaper single-call judge. Neither changes the
other; use `grounded()` for a rule you write in Python, the type for one an
operator writes centrally.

## Secrets detection

`no_secrets()` blocks leaked credentials — private keys, AWS / GitHub / Slack /
Google / OpenAI / Stripe tokens, JWTs, and generic `api_key = "..."`
assignments. Detected values are **masked** in the guardrail metadata, so the
secret is never re-leaked into logs or the local UI.

```python
from fastaiagent import no_secrets

agent = Agent(name="safe", llm=llm, guardrails=[no_secrets()])
```

## Toxicity — keyword or LLM

The default `toxicity_check()` is the original zero-dependency keyword check.
Opt into a much stronger LLM classifier (scored 0–1) with `mode="llm"`:

```python
from fastaiagent import toxicity_check

agent = Agent(
    name="safe",
    llm=llm,
    guardrails=[toxicity_check(mode="llm", llm=llm, threshold=0.5)],  # lower = stricter
)
```

## Topic controls

`banned_topics()` (blocklist) and `allowed_topics()` (whitelist) keep an agent
on-mission. They classify **semantically** by default (`mode="llm"`); use
`mode="keyword"` for a zero-dependency literal match.

```python
from fastaiagent import allowed_topics, banned_topics

guardrails = [
    banned_topics(["politics", "competitor pricing"]),
    allowed_topics(["billing", "shipping", "returns"]),
]
```

Give a topic a **definition** and the judge stops guessing what you meant. Either
shape works:

```python
banned_topics({
    "Competitor products": "Any mention or comparison of a competing vendor's product.",
    "Medical advice": "Diagnosis, treatment or medication guidance for a specific person.",
})

banned_topics([
    {"name": "Competitor products", "description": "Any mention of a rival vendor's product."},
])
```

A bare name still works — it just asks the judge to infer the scope from the
label, and "crypto" cannot tell it whether blockchain patents count.

### `banned_topics()` / `allowed_topics()` vs the `topic` type

Unlike [`grounded()` and `groundedness`](#grounded-vs-the-groundedness-type),
these are **not** two engines. Since 1.58.0 the factories are thin wrappers that
emit a [`topic`](actions.md#topic) rule — same prompt, same parser, same verdict
as one authored in the console:

```python
banned_topics(["politics"]).to_dict()["guardrail_type"]   # -> 'topic'
```

That matters because of what it fixes. These factories used to emit `code`
guardrails, whose logic is a local Python callable — so pushing one to the
console produced an opaque row the plane could neither run, edit, nor
re-distribute. Now the same call **round-trips**: it arrives as a first-class
rule an operator can open, change, and hand to every other agent in the domain.

Two call shapes deliberately stay local `code` rules, because neither can be
reproduced centrally:

| Call | Emits | Why |
|---|---|---|
| `banned_topics([...])` | `topic` | The default. Round-trips to the console. |
| `banned_topics([...], llm={"model": "gpt-4o-mini"})` | `topic` | Kwargs serialise, so the plane can rebuild the same judge. |
| `banned_topics([...], mode="keyword")` | `code` | A substring match with no judge — there is nothing to distribute. |
| `banned_topics([...], llm=LLMClient(...))` | `code` | A live client cannot be serialised into a stored config. |

`responsible_ai(banned=[...], llm=client)` passes a client through, so it lands in
that last row; drop the `llm=` argument (or pass a kwargs dict) to get a
distributable rule.

!!! note "The two defaults disagree, on purpose"
    `banned_topics()` defaults to `on_error="allow"` and `allowed_topics()` to
    `on_error="block"`. A whitelist that cannot classify must not pass: "no topic
    matched" and "the judge could not answer" would otherwise both fail the rule,
    and only one of them is a verdict. The polarity never overrides `on_error` —
    a plane-distributed rule carries its own and the runner simply honours it.

## Reflection — self-critique and revise

`Reflect` is **middleware** (not a guardrail): on each *final* answer it asks the
model to critique itself against optional `facts` (non-negotiable
truths/policies) and `criteria`, then revise. Responses that carry tool calls are
passed through untouched — only the terminal answer is reflected. It **fails
open**: any reviewer error keeps the original answer.

```python
from fastaiagent import Agent, Reflect

agent = Agent(
    name="grounded-writer",
    llm=llm,
    middleware=[Reflect(
        facts=["Refunds are only valid within 30 days of purchase."],
        criteria="Be concise and cite the policy.",
    )],
)
```

## Guardrails vs middleware — which to use

`grounded`, `no_secrets`, `toxicity_check`, and the topic controls are
**guardrails**: they *assert* and **block** on failure (`GuardrailBlockedError`).
`Reflect` is **middleware**: it *transforms* the answer (revises it) rather than
blocking. Combine them — block on hard violations, reflect to improve borderline
answers. See [Middleware](../agents/middleware.md) for the hook model.

## Open-source vs hosted

Every check here runs **in-process** in the open-source SDK — no hosted runtime,
nothing phones home. The zero-dependency checks need no API key at all; the
LLM-backed checks reuse whatever `LLMClient` you already configured.
