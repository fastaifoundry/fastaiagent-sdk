"""Proof for "3 · Between capture and egress" on docs/tracing/trace-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db, and
OpenTelemetry's own InMemorySpanExporter standing in for Datadog or Jaeger.

local.db always holds the full span. What an exporter receives depends on two
switches: FASTAIAGENT_TRACE_PAYLOADS=0 strips the payload keys on the way out,
and a RedactionPolicy masks what it matches. The master switch,
FASTAIAGENT_TRACE_ENABLED=0, captures nothing at all.
"""

import os
import subprocess
import sys
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,  # noqa: E402
)

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import RedactionPolicy, TraceStore, set_redaction_policy  # noqa: E402
from fastaiagent.trace.otel import add_exporter, get_tracer_provider  # noqa: E402

exporter = InMemorySpanExporter()
add_exporter(exporter)
store = TraceStore()
agent = fa.Agent(name="support", llm=FunctionModel(lambda m: "Your card ending 4242 is active."))

QUESTION = "Is my card 4111 1111 1111 4242 active?"


def run(label: str) -> None:
    exporter.clear()
    r = agent.run(QUESTION)
    get_tracer_provider().force_flush()
    local = {s.name: s.attributes for s in store.get_trace(r.trace_id).spans}
    exported = {s.name: dict(s.attributes) for s in exporter.get_finished_spans()}
    root = f"agent.{agent.name}"
    print(f"{label}")
    print(f"  local.db  agent.input = {local[root].get('agent.input')!r}")
    print(f"  exporter  agent.input = {exported[root].get('agent.input', '(stripped)')!r}")
    llm = next(n for n in local if n.startswith("llm."))
    print(f"  local.db  gen_ai.request.messages present: {'gen_ai.request.messages' in local[llm]}")
    print(
        f"  exporter  gen_ai.request.messages present: {'gen_ai.request.messages' in exported[llm]}"
    )
    msgs = exported[llm].get("gen_ai.request.messages", "")
    print(f"  exporter  gen_ai.request.messages carries the card number: {'4242' in msgs}")
    print(f"  exporter  gen_ai.request.model = {exported[llm].get('gen_ai.request.model')!r}")


run("default: payloads exported")
os.environ["FASTAIAGENT_TRACE_PAYLOADS"] = "0"
run("FASTAIAGENT_TRACE_PAYLOADS=0")
del os.environ["FASTAIAGENT_TRACE_PAYLOADS"]
set_redaction_policy(RedactionPolicy(patterns=[r"\b(?:\d[ -]*?){13,16}\b"], replacement="[CARD]"))
run("RedactionPolicy(card pattern)")
set_redaction_policy(None)

code = """
import fastaiagent as fa
from fastaiagent.testing import FunctionModel
from fastaiagent.trace import TraceStore
agent = fa.Agent(name="support", llm=FunctionModel(lambda m: "ok"))
r = agent.run("hello")
print("FASTAIAGENT_TRACE_ENABLED=0")
print("  trace_id on the result:", r.trace_id)
rows = TraceStore()._db.fetchall("SELECT count(*) AS n FROM spans")[0]["n"]
print("  rows in a fresh local.db:", rows)
"""
out = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    env={
        **os.environ,
        "FASTAIAGENT_TRACE_ENABLED": "0",
        "FASTAIAGENT_LOCAL_DB": os.path.join(tempfile.mkdtemp(), "local.db"),
    },
)
print(out.stdout.strip() or out.stderr[-500:])
