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
    banned_topics(["politics", "competitor pricing"], llm=llm),
    allowed_topics(["billing", "shipping", "returns"], llm=llm),
]
```

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
