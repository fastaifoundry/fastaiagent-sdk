# Swarm — Peer-to-Peer Multi-Agent

A `Swarm` is a mesh of agents that can hand off control to each other by calling `handoff_to_<peer>` tools. Unlike [Supervisor](teams.md), there is no central coordinator — the currently active agent itself decides when to transfer control and to whom.

Use a Swarm when:

- The routing decision is best made by the specialist, not a coordinator (the specialist knows when they're out of their depth).
- Agents should loop naturally (writer → critic → writer → critic → done) without a hub mediating every round.
- You want lower latency and fewer tokens than a supervisor's fan-in/fan-out.

Use a [Supervisor](teams.md) instead when:

- A single LLM should synthesize multiple workers' outputs into one answer.
- The coordinator needs to run the same worker multiple times with different inputs.
- You want the supervisor to be accountable for the final answer.

## Quickstart

```python
from fastaiagent import Agent, LLMClient, Swarm

llm = LLMClient(provider="openai", model="gpt-4o-mini")

researcher = Agent(
    name="researcher",
    system_prompt=(
        "Research the user's topic. When you have enough material, hand off "
        "to the writer to produce the draft."
    ),
    llm=llm,
)
writer = Agent(
    name="writer",
    system_prompt=(
        "Turn research notes into a polished draft. After drafting, hand off "
        "to the critic for review."
    ),
    llm=llm,
)
critic = Agent(
    name="critic",
    system_prompt=(
        "Review the draft. If it needs revision, hand off back to the writer "
        "with specific feedback. If it's good, produce the final answer."
    ),
    llm=llm,
)

swarm = Swarm(
    name="content_team",
    agents=[researcher, writer, critic],
    entrypoint="researcher",
    handoffs={
        "researcher": ["writer"],
        "writer":     ["critic"],
        "critic":     ["writer"],   # critic may send back to writer for revision
    },
    max_handoffs=6,
)

result = swarm.run("Write a 500-word brief on large language models.")
print(result.output)
```

## How it works

Every turn, the currently active agent is cloned and given one `handoff_to_<peer>` `FunctionTool` per peer listed in `handoffs[current_agent]`. When the agent's LLM calls one of those tools:

1. The swarm's outer loop notices the handoff in the agent's `tool_calls`.
2. It enforces the allowlist: the target must be in `handoffs[current]`.
3. It bumps `state.handoff_count` and checks `max_handoffs`.
4. It constructs a briefing message for the next agent: "`researcher` handed off to you with reason: '...'. Earlier request: '...'. Current shared state: {...}. Please continue."
5. The next agent runs. When *it* either produces a final response with no tool call, or calls another handoff, the loop continues.

No central LLM. No state graph. A plain `while` loop plus tool-call inspection.

## API

```python
class Swarm:
    def __init__(
        self,
        name: str,
        agents: Sequence[Agent],
        entrypoint: str,
        handoffs: dict[str, list[str]] | None = None,
        max_handoffs: int = 8,
    ): ...

    def run(self, input: str, *, context: RunContext | None = None) -> AgentResult: ...
    async def arun(self, input: str, *, context=None, **kwargs) -> AgentResult: ...
    async def astream(self, input: str, *, context=None, **kwargs) -> AsyncGenerator[StreamEvent, None]: ...
    def stream(self, input: str, *, context=None) -> AgentResult: ...

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict, agents: Sequence[Agent]) -> Swarm: ...
```

A `Swarm` implements the same `run / arun / astream / stream` surface as `Agent`, so it drops into a [Chain](../chains/index.md) node, wraps inside another `Swarm`, or plugs into anything else that takes an agent-shaped object.

### `handoffs` allowlist

The default is full mesh — every agent may hand off to every other agent:

```python
Swarm(name="s", agents=[a, b, c], entrypoint="a")
# equivalent to handoffs={"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
```

Explicit allowlists constrain routing and clarify intent:

```python
Swarm(
    name="triage",
    agents=[triage, coder, writer, support],
    entrypoint="triage",
    handoffs={
        "triage":  ["coder", "writer", "support"],  # triage fans out
        "coder":   [],                              # specialists terminate
        "writer":  [],
        "support": [],
    },
)
```

Attempting to hand off to a peer not in the allowlist raises `SwarmError`.

### `max_handoffs`

Hard cap on handoffs per run. Exceeding it raises `SwarmError` with the full path so you can see where the cycle formed.

```python
Swarm(..., max_handoffs=3)
```

Default is `8`. Tune to your topology: a linear pipeline of 4 agents needs at most 3 handoffs.

### Shared blackboard (`SwarmState.shared`)

Handoff tools accept an optional `context=` dict whose entries merge into `SwarmState.shared` and are visible to every subsequent agent through the briefing message:

```
researcher calls handoff_to_writer(
    reason="Research complete, draft incoming",
    context={"sources": ["arxiv:2024.12345"], "tone": "academic"},
)
```

The writer sees `Current shared state: {'sources': ['arxiv:2024.12345'], 'tone': 'academic'}` in its briefing.

## Streaming

`swarm.astream(input)` yields the full stream of the currently active agent (TextDelta, ToolCallStart, ToolCallEnd, Usage). When a handoff fires, a single `HandoffEvent(from_agent, to_agent, reason)` is emitted before the target agent starts streaming:

```python
from fastaiagent.llm.stream import TextDelta, HandoffEvent

async for event in swarm.astream("Write a poem about bridges."):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, HandoffEvent):
        print(f"\n[{event.from_agent} → {event.to_agent}: {event.reason}]\n")
```

## Composing with other primitives

### With agent tools

Each agent keeps its own tools. Handoff tools are **added** to the agent's tool list per turn — they don't replace the agent's normal tools:

```python
researcher = Agent(
    name="researcher",
    llm=llm,
    tools=[web_search, read_url],   # still available when researcher is active
    ...,
)
```

### With `ComposableMemory`

Each agent keeps its own memory. You can share memory across the swarm by passing the same `ComposableMemory` to multiple agents — they'll all write to, and read from, the same store:

```python
from fastaiagent.agent import ComposableMemory, AgentMemory
from fastaiagent.agent.memory_blocks import VectorBlock, FactExtractionBlock
from fastaiagent.kb.backends.faiss import FaissVectorStore

shared_memory = ComposableMemory(
    blocks=[
        VectorBlock(store=FaissVectorStore(dimension=384)),
        FactExtractionBlock(llm=llm, max_facts=100),
    ],
    primary=AgentMemory(max_messages=30),
)

swarm = Swarm(
    name="team",
    agents=[
        Agent(name="a", llm=llm, memory=shared_memory, ...),
        Agent(name="b", llm=llm, memory=shared_memory, ...),
    ],
    entrypoint="a",
)
```

See [Memory](memory.md) for the full block reference.

### With a KB tool

An agent that uses a `LocalKB` as a tool keeps that capability inside the swarm:

```python
from fastaiagent.kb import LocalKB

kb = LocalKB(name="product-docs")
kb.add("docs/")

support = Agent(
    name="support",
    llm=llm,
    tools=[kb.as_tool()],
    system_prompt="Search product-docs before answering.",
)

swarm = Swarm(
    name="triage_swarm",
    agents=[triage, support, billing],
    entrypoint="triage",
)
```

## Serialization

```python
data = swarm.to_dict()
# {'name': 'triage_swarm', 'agent_names': ['triage', 'support'], 'entrypoint': ..., 'handoffs': {...}, 'max_handoffs': 8}

restored = Swarm.from_dict(data, agents=[triage, support])
```

`to_dict` captures the structural data only. The caller must supply the live `Agent` instances when rehydrating (we don't auto-reconstruct agents — see `Agent.to_dict`/`from_dict`).

## Swarm vs Supervisor — when to use which

| | Swarm | Supervisor |
|---|---|---|
| Coordinator | None | Central LLM |
| Agent wrapping | Raw `Agent` | `Worker(agent, role, description)` |
| Tool auto-injected | `handoff_to_<peer>` — **transfers control** | `delegate_to_<role>` — **executes inline, returns result** |
| Control flow | Mesh, peer-to-peer | Star, hub-and-spoke |
| Final answer from | Whichever agent produces a non-handoff response | Supervisor synthesizes |
| Iteration cap | `max_handoffs` | `max_delegation_rounds` |
| Typical use | Specialist networks, loops | Fan-out / synthesis patterns |

## Errors

```python
from fastaiagent import SwarmError

try:
    swarm.run("...")
except SwarmError as e:
    # Covers: missing entrypoint, duplicate agent names, unknown peer in
    # handoffs, disallowed handoff attempt, max_handoffs exceeded.
    print(e)
```

`SwarmError` subclasses `AgentError`, so `except AgentError` catches both supervisor and swarm failures.

---

## Next Steps

- [Supervisor / Worker Teams](teams.md) — Centralized delegation topology
- [Memory](memory.md) — Add long-term memory to swarm agents
- [KB Backends](../knowledge-base/backends.md) — Give swarm agents searchable knowledge
- [Chains](../chains/index.md) — Wrap a swarm as a chain node
