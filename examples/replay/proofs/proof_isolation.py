"""Proof for "5 · Between a rerun and production" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

A rerun is rebuilt from the trace alone: it has no memory, so it reads none and
writes none; its guardrails are rebuilt from the trace and run again; and it has
no checkpointer, so a tool that asks for a human's approval ends the rerun with
a ReplayError that names the tool.
"""

import json
import os
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent._internal.errors import ReplayError  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay, TraceStore  # noqa: E402


@fa.tool(name="refund")
def refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


def responder(messages):
    if messages[-1].role.value != "tool":
        return "", [{"name": "refund", "arguments": {"order_id": "1042"}}]
    return "Order 1042 has been refunded."


memory = fa.Memory(window=6)
agent = fa.Agent(
    name="refunds",
    system_prompt="You handle refunds.",
    llm=FunctionModel(responder),
    tools=[refund],
    memory=memory,
    guardrails=[fa.no_secrets()],
)
agent.run("Hi, I'm Dana.")
r = agent.run("Refund order 1042")
store = TraceStore()
first_llm = next(s for s in store.get_trace(r.trace_id).spans if s.name.startswith("llm."))
roles = [m["role"] for m in json.loads(first_llm.attributes["gen_ai.request.messages"])]
print(f"original run: prompt roles {roles}   memory holds {len(memory.messages)} messages")

rerun = Replay.load(r.trace_id).fork_at(0).with_determinism("recorded", on_miss="error").rerun()
spans = store.get_trace(rerun.trace_id).spans
first_llm = next(s for s in spans if s.name.startswith("llm."))
roles = [m["role"] for m in json.loads(first_llm.attributes["gen_ai.request.messages"])]
print(f"rerun:        prompt roles {roles}   memory holds {len(memory.messages)} messages")
print(f"rerun spans:  {[s.name for s in spans]}")


@fa.tool(name="refund")
def refund_with_approval(order_id: str) -> str:
    """Refund an order, after a human approves."""
    fa.interrupt("refund needs a human", {"order_id": order_id})
    return "refunded"


try:
    (
        Replay.load(r.trace_id)
        .fork_at(0)
        .with_tool_override("refund", refund_with_approval)
        .with_determinism("recorded", on_miss="error")
        .rerun()
    )
except ReplayError as e:
    print(f"a tool that pauses → ReplayError: {e}")
