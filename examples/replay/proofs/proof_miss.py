"""Proof for "4 · Between the recording and the rerun" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

The original run died on its second model call, so the recording holds one
response. A recorded rerun needs two. With on_miss="error" the rerun stops
before any provider call and says so. With the default, on_miss="live", it
warns and makes a live call.
"""

import logging
import os
import sys
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent._internal.errors import ReplayError  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay, TraceStore  # noqa: E402


@fa.tool(name="lookup_order", replay_class="read_only")
def lookup_order(order_id: str) -> str:
    """Look an order up in the shipping system."""
    return "shipped via DHL on 12 September"


def responder(messages):
    if messages[-1].role.value != "tool":
        return "", [{"name": "lookup_order", "arguments": {"order_id": "1042"}}]
    raise RuntimeError("provider returned 503")


agent = fa.Agent(name="shipping", llm=FunctionModel(responder), tools=[lookup_order])
try:
    agent.run("Where is order 1042?")
except RuntimeError as e:
    print(f"original run raised: {e}")
trace_id = TraceStore().list_traces()[0].trace_id
replay = Replay.load(trace_id)
print(replay.summary())
captured = sum(
    1
    for s in replay.steps()
    if "gen_ai.response.content" in s.attributes or "gen_ai.response.tool_calls" in s.attributes
)
print(f"captured model responses: {captured}")

try:
    replay.fork_at(0).with_determinism("recorded", on_miss="error").rerun()
except ReplayError as e:
    print(f"on_miss='error' → ReplayError: {e}")

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="  warning: %(message)s")
try:
    replay.fork_at(0).with_determinism("recorded").rerun()
except Exception as e:
    print(f"on_miss='live' (default) → the live call to provider 'test' failed: {type(e).__name__}")
