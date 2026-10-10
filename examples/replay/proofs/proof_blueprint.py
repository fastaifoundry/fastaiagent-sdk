"""Proof for "1 · Between the trace and the agent" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

The root span of a run carries what Agent.from_dict needs: the resolved system
prompt, the model configuration with no api_key, the tool schemas with their
replay class, the guardrails and the config. Replay.load reads it back. The
rebuilt agent gets the live tool when the tool is registered in this process,
and has no memory whatever the original had.
"""

import json
import os
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay, TraceStore  # noqa: E402


@fa.tool(name="lookup_order", replay_class="read_only")
def lookup_order(order_id: str) -> str:
    """Look an order up in the shipping system."""
    return "shipped via DHL on 12 September"


def responder(messages):
    if messages[-1].role.value != "tool":
        return "", [{"name": "lookup_order", "arguments": {"order_id": "1042"}}]
    return f"Order 1042 was {messages[-1].content}."


memory = fa.Memory(window=4)
agent = fa.Agent(
    name="shipping",
    system_prompt="You answer shipping questions.",
    llm=FunctionModel(responder),
    tools=[lookup_order],
    guardrails=[fa.no_secrets()],
    memory=memory,
    config=fa.AgentConfig(max_iterations=3),
)
r = agent.run("Where is order 1042?")

a = TraceStore().get_trace(r.trace_id).spans[0].attributes
print("the root span carries:")
for k in (
    "agent.name",
    "agent.system_prompt",
    "agent.llm.config",
    "agent.tools",
    "agent.guardrails",
    "agent.config",
    "agent.input",
    "agent.output",
):
    v = a.get(k)
    s = v if isinstance(v, str) else json.dumps(v)
    print(f"  {k:<20} {s if len(s) <= 88 else s[:85] + '…'}")
print(f"  api_key in agent.llm.config: {'api_key' in a['agent.llm.config']}")
print()
print(Replay.load(r.trace_id).summary())
print()

# What ForkedReplay.arerun() builds from those attributes, then hands to Agent.from_dict.
blueprint = {
    "name": a["agent.name"],
    "system_prompt": a["agent.system_prompt"],
    "llm_endpoint": json.loads(a["agent.llm.config"]),
    "tools": json.loads(a["agent.tools"]),
    "guardrails": json.loads(a["agent.guardrails"]),
    "config": json.loads(a["agent.config"]),
}
rebuilt = fa.Agent.from_dict(blueprint)
print("the rebuilt agent:")
print(
    f"  llm         {type(rebuilt.llm).__name__}(provider={rebuilt.llm.provider!r}, "
    f"model={rebuilt.llm.model!r})"
)
print(
    f"  tools       {[t.name for t in rebuilt.tools]}   the live function: "
    f"{rebuilt.tools[0] is lookup_order}   replay_class={rebuilt.tools[0].replay_class}"
)
print(f"  guardrails  {[g.name for g in rebuilt.guardrails]}")
print(f"  config      max_iterations={rebuilt.config.max_iterations}")
print(
    f"  memory      {rebuilt.memory}   (the original's window holds "
    f"{len(memory.messages)} messages)"
)
