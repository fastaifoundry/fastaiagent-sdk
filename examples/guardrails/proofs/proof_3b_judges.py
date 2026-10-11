"""Live companion to "3 · Between 'could not run' and 'found nothing'" — the model-backed judges.

Live: needs OPENAI_API_KEY (gpt-4.1-mini). Costs a few cents.

A topic rule answers with the topics it found; a groundedness rule scores an answer
against the context the retrieval step put in the run-scoped slot. Take the slot
away and the rule does not pass — it reports that it could not run.
"""

import asyncio
import os
import sys

import _common  # noqa: F401
from _common import heading

import fastaiagent as fa
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType, run_guardrail

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set — this proof calls a real model.")

LLM = {"provider": "openai", "model": os.environ.get("PROOF_MODEL", "gpt-4.1-mini")}

heading("topic, mode=deny: the judge names what it found")
rule = Guardrail(name="no-competitors", guardrail_type=GuardrailType.topic, position=GuardrailPosition.output,
                 config={"topics": ["competitor pricing", "legal advice"], "mode": "deny", "llm": LLM})
for text in ("Our Pro plan is $40 a month.", "Acme charges $35 for the same tier, so we are pricier."):
    r = asyncio.run(run_guardrail(rule, text))
    print(f"passed={str(r.passed):<5} matched={r.metadata.get('matched')}  ← {text!r}")

heading("groundedness: the context comes from the run-scoped slot")
rule = Guardrail(name="grounded", guardrail_type=GuardrailType.groundedness, position=GuardrailPosition.output,
                 config={"context_key": "context", "threshold": 0.7, "llm": LLM})
docs = "Refunds are processed within 5 business days. Support is open Monday to Friday, 9:00-17:00 CET."
answer = "Refunds take 5 business days and support is available on weekends."
with fa.guardrail_context(context=docs):
    r = asyncio.run(run_guardrail(rule, answer))
print(f"with context    → passed={r.passed} score={r.metadata.get('score')} unsupported={r.metadata.get('unsupported_claims')}")
r = asyncio.run(run_guardrail(rule, answer))
print(f"without context → passed={r.passed} errored={r.errored}: {(r.message or '')[:90]}…")
