"""Proof for "3 · Between the record and the world" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

Recorded mode fixes the model's words, not the tools. Since 1.89.0 a rerun runs
only the tools you supply: by default every tool is blocked, the agent gets a
tool error, and because the next model turn is served from the recording, the
output still matches. with_tool_policy("live", allow=[...]) runs the real
function again, but only in the process that defined it: a fresh process has
no function to run. with_tool_override swaps the tool for a stub. A tool's
replay class is recorded on its span, defaults to side_effecting, and is never
inferred, not even for a GET.
"""

import os
import subprocess
import sys
import tempfile

DB = os.path.join(tempfile.mkdtemp(), "local.db")
os.environ["FASTAIAGENT_LOCAL_DB"] = DB

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay, TraceStore  # noqa: E402

charges: list[float] = []


@fa.tool(name="charge_card")  # unmarked
def charge_card(order_id: str, amount: float) -> str:
    """Charge the customer's card."""
    charges.append(amount)
    return f"charged {amount:.2f}"


status = fa.RESTTool(name="carrier_status", url="http://127.0.0.1:9/status", method="GET")
print(
    f"replay_class: charge_card={charge_card.replay_class!r}  "
    f"GET carrier_status={status.replay_class!r}"
)


def responder(messages):
    if messages[-1].role.value != "tool":
        return "", [{"name": "charge_card", "arguments": {"order_id": "1042", "amount": 42.0}}]
    return f"Done: {messages[-1].content}."


def tool_span(trace_id):
    span = next(s for s in TraceStore().get_trace(trace_id).spans if s.name == "tool.charge_card")
    return span.attributes


agent = fa.Agent(name="billing", llm=FunctionModel(responder), tools=[charge_card, status])
r = agent.run("Charge order 1042")
replay_class = tool_span(r.trace_id)["fastaiagent.tool.replay_class"]
print(f"tool span: fastaiagent.tool.replay_class={replay_class!r}")
print(f"original run: {r.output!r}   charges={charges}")

rerun = Replay.load(r.trace_id).fork_at(0).with_determinism("recorded", on_miss="error").rerun()
print(f"default rerun: {rerun.new_output!r}   charges={charges}")
print(f"  tool_dispositions={rerun.tool_dispositions}")
print(f"  blocked_tool_calls={rerun.blocked_tool_calls}")
attrs = tool_span(rerun.trace_id)
print(f"  tool.status = {attrs.get('tool.status')}")

rerun = (
    Replay.load(r.trace_id)
    .fork_at(0)
    .with_determinism("recorded", on_miss="error")
    .with_tool_policy("live", allow=["charge_card"])
    .rerun()
)
print(f"live, same process: {rerun.new_output!r}   charges={charges}")
print(f"  tool_dispositions={rerun.tool_dispositions}")

code = f"""
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = {DB!r}
from fastaiagent.trace import Replay, TraceStore
rerun = (
    Replay.load({r.trace_id!r})
    .fork_at(0)
    .with_determinism("recorded", on_miss="error")
    .with_tool_policy("live", allow=["charge_card"])
    .rerun()
)
span = next(s for s in TraceStore().get_trace(rerun.trace_id).spans if s.name == "tool.charge_card")
print("  output:", repr(rerun.new_output))
print("  disposition:", rerun.tool_dispositions["charge_card"])
status, error = span.attributes.get("tool.status"), span.attributes.get("tool.error")
print("  tool.status =", status, "| tool.error =", repr(error))
"""
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print("live, fresh process:")
print(out.stdout.strip() or out.stderr.strip())


def stub(order_id: str, amount: float) -> str:
    return "charged 0.00 (stub)"


rerun = (
    Replay.load(r.trace_id)
    .fork_at(0)
    .with_tool_override("charge_card", stub)
    .with_determinism("recorded", on_miss="error")
    .rerun()
)
print(f"with_tool_override: {rerun.new_output!r}   charges={charges}")
print(f"  tool_dispositions={rerun.tool_dispositions}")
