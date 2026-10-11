"""Proof for "4 · Between the verdict and the trace" on docs/guardrails/guardrail-boundaries.md.

Offline: a real Agent over the SDK's TestModel, a real pii rule, real OTel spans.

Every rule that runs leaves one span, on a pass as well as a block. The span carries
the verdict and, for the types that have one, a detail that is an allowlist: counts
and entity names, never the matched values. Local capture keeps the full result; the
export policy strips the detail when payloads are off.
"""

import json
import os

import _common  # noqa: F401
from _common import SpanCollector, guardrail_spans, heading

from fastaiagent import Agent
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType, run_guardrail
from fastaiagent.testing import TestModel
from fastaiagent.trace.redaction import apply_export_policy

col = SpanCollector()
pii = Guardrail(name="no-email", guardrail_type=GuardrailType.pii, position=GuardrailPosition.output,
                config={"entities": ["email", "phone"]})
ssn = Guardrail(name="no-ssn", guardrail_type=GuardrailType.regex, position=GuardrailPosition.output,
                config={"pattern": r"\b\d{3}-\d{2}-\d{4}\b"}, blocking=False)
clean = Guardrail(name="length", position=GuardrailPosition.output, fn=lambda t: len(t) < 500)

heading("one span per rule, pass and block alike")
agent = Agent(name="t", llm=TestModel(response="Reach me at bob@acme.com, SSN 123-45-6789."),
              guardrails=[clean, pii, ssn])
try:
    agent.run("go")
except GuardrailBlockedError as e:
    print("run      :", f"blocked by {e.guardrail_name}")
for attrs in guardrail_spans(col):
    keep = {k.removeprefix("fastaiagent.guardrail."): v for k, v in attrs.items() if "guardrail." in k}
    print("span     :", json.dumps(keep))

heading("what stays local vs what the span carries")
import asyncio  # noqa: E402

from fastaiagent.guardrail import execute_guardrails  # noqa: E402

TEXT = "Reach me at bob@acme.com, SSN 123-45-6789."
for rule in (pii, ssn):
    r = asyncio.run(run_guardrail(rule, TEXT))                      # compute: the full result
    try:
        asyncio.run(execute_guardrails([rule], TEXT, GuardrailPosition.output))   # emit: the span
    except GuardrailBlockedError:
        pass
    span = [a for a in guardrail_spans(col) if a.get("fastaiagent.guardrail.name") == rule.name][-1]
    print(f"{rule.name} ({rule.guardrail_type.value})")
    print("  result.metadata, local :", json.dumps(r.metadata)[:120])
    print("  span detail, exported  :", span.get("fastaiagent.guardrail.detail"))
detail = [a for a in guardrail_spans(col) if a.get("fastaiagent.guardrail.name") == "no-email"][-1]

heading("the export policy with FASTAIAGENT_TRACE_PAYLOADS=0")
os.environ["FASTAIAGENT_TRACE_PAYLOADS"] = "0"
exported = apply_export_policy(dict(detail))
keys = sorted(k for k in detail if k.startswith("fastaiagent.guardrail."))
print("kept   :", [k.removeprefix("fastaiagent.guardrail.") for k in keys if k in exported])
print("dropped:", [k.removeprefix("fastaiagent.guardrail.") for k in keys if k not in exported])
