"""Proof for "2 · Between the span and the disk" on docs/tracing/trace-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

Three things about the write:
  * a span is on disk the moment it ends — a tool reads the store mid-run and
    finds the LLM span that preceded it, while the root is still open;
  * a run that raises leaves an ERROR root carrying the exception, and every
    span that ended before the crash;
  * a store that cannot be written drops the span and never fails the run.
"""

import os
import tempfile

DB = os.path.join(tempfile.mkdtemp(), "local.db")
os.environ["FASTAIAGENT_LOCAL_DB"] = DB

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import TraceStore  # noqa: E402

store = TraceStore()
seen_mid_run: list[str] = []


@fa.tool(name="peek")
def peek(order_id: str) -> str:
    """Read the trace store from inside the run."""
    seen_mid_run.extend(f"{rec.span.name} [{rec.span.status}]" for rec in store.list_spans(since=0))
    return "shipped"


@fa.tool(name="explode")
def explode(order_id: str) -> str:
    """A tool that fails."""
    raise RuntimeError("carrier API returned 503")


def responder(messages):
    question = next(m.content for m in messages if m.role.value == "user")
    if messages[-1].role.value != "tool":
        tool = "explode" if "tool fails" in str(question) else "peek"
        return "", [{"name": tool, "arguments": {"order_id": "1042"}}]
    if "model fails" in str(question):
        raise RuntimeError("provider returned 503")  # the second model call dies
    return "Order 1042 has shipped."


agent = fa.Agent(
    name="shipping",
    llm=FunctionModel(responder),
    tools=[peek, explode],
    config=fa.AgentConfig(max_iterations=2),
)


def show(trace_id: str) -> None:
    for span in store.get_trace(trace_id).spans:
        exc = next(
            (
                e["attributes"].get("exception.message")
                for e in span.events
                if e["name"] == "exception"
            ),
            None,
        )
        extra = f"  exception: {exc}" if exc else ""
        if span.attributes.get("tool.status") == "error":
            extra = f"  tool.status=error  tool.error={span.attributes.get('tool.error')!r}"
        print(f"    {span.name:<26}[{span.status}]{extra}")


# 1. on disk the moment it ends
r = agent.run("Where is order 1042?")
print("rows on disk while the tool was running:")
for row in seen_mid_run:
    print("   ", row)
print("rows on disk after the run:")
show(r.trace_id)

# 2a. a tool that fails is a result the model sees, not a crash
r = agent.run("Where is order 1042? (tool fails)")
print(f"tool failed, run finished: {r.output!r}")
show(r.trace_id)

# 2b. a model call that fails ends the run; what ended before it is on disk
try:
    agent.run("Where is order 1042? (model fails)")
except RuntimeError as e:
    print(f"model failed, run raised: {e}")
show(store.list_traces()[0].trace_id)

# 3. an unwritable store drops the span, never the run
os.environ["FASTAIAGENT_LOCAL_DB"] = "/nonexistent-dir/local.db"
import subprocess  # noqa: E402
import sys  # noqa: E402

code = """
import fastaiagent as fa
from fastaiagent.testing import FunctionModel
agent = fa.Agent(name="shipping", llm=FunctionModel(lambda m: "Order 1042 has shipped."))
print("answer with an unwritable store:", repr(agent.run("Where is order 1042?").output))
"""
out = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    env={**os.environ, "FASTAIAGENT_LOCAL_DB": "/nonexistent-dir/local.db"},
)
print(out.stdout.strip())
warning = [line for line in out.stderr.splitlines() if "could not write span" in line]
print("warning logged once:", len(warning) == 1, "—", warning[0][:90] + "…" if warning else "none")
