"""Proof for "2 · Between the model and the record" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

A two-turn tool loop is run once. A recorded rerun serves both captured model
responses in order: the tool-call turn, so the tool really runs, then the
answer. No model is called, the output is byte-identical, and every llm span
of the rerun is marked replay.mode=recorded. Changing the prompt under recorded
mode changes the prompt the agent is given, not the recorded answer.
"""

import os
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay, TraceStore  # noqa: E402

tool_calls: list[str] = []


@fa.tool(name="lookup_order", replay_class="read_only")
def lookup_order(order_id: str) -> str:
    """Look an order up in the shipping system."""
    tool_calls.append(order_id)
    return "shipped via DHL on 12 September"


def responder(messages):
    if messages[-1].role.value != "tool":
        return "", [{"name": "lookup_order", "arguments": {"order_id": "1042"}}]
    return f"Order 1042 was {messages[-1].content}."


model = FunctionModel(responder)
agent = fa.Agent(
    name="shipping", system_prompt="You answer shipping questions.", llm=model, tools=[lookup_order]
)
r = agent.run("Where is order 1042?")
print(f"original run: {r.output!r}   model calls={len(model.calls)} tool calls={len(tool_calls)}")

forked = Replay.load(r.trace_id).fork_at(0).with_determinism("recorded", on_miss="error")
rerun = forked.rerun()
print(f"recorded rerun: {rerun.new_output!r}")
print(f"  byte-identical: {rerun.original_output == rerun.new_output}")
print(f"  model calls during the rerun: {len(model.calls) - 2}   tool calls: {len(tool_calls) - 1}")
print("  the rerun's own trace:")
for s in TraceStore().get_trace(rerun.trace_id).spans:
    print(f"    {s.name:<26} replay.mode={s.attributes.get('replay.mode')}")
cmp = forked.compare(rerun)
print(f"  compare(): status={cmp.compare_status}  diverged_at={cmp.diverged_at}")

forked = (
    Replay.load(r.trace_id)
    .fork_at(0)
    .modify_prompt("Répondez en français.")
    .with_determinism("recorded", on_miss="error")
)
rerun = forked.rerun()
root = TraceStore().get_trace(rerun.trace_id).spans[0]
print("modify_prompt under recorded mode:")
print(f"  prompt the rerun was given: {root.attributes['agent.system_prompt']!r}")
print(f"  answer: {rerun.new_output!r}")
