# Agent Memory

Memory lets agents remember previous conversations across multiple `run()` calls. Two flavors ship in the SDK:

| | `AgentMemory` | `ComposableMemory` |
|---|---|---|
| Since | 0.1.x | 0.4.0 |
| Stores | Sliding window of raw messages | Sliding window **plus** any number of long-term blocks |
| Best for | Chatbots, short sessions | Long-running assistants, personal memory, fact tracking |
| Drop-in replacement | — | Yes — `Agent(memory=...)` accepts either |

If you were using `AgentMemory` before, no code changes are needed. Swap to `ComposableMemory` when a sliding window is no longer enough.

## Sliding-window memory: `AgentMemory`

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.agent import AgentMemory

memory = AgentMemory(max_messages=20)

agent = Agent(
    name="assistant",
    system_prompt="Remember what users tell you. Be brief.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    memory=memory,
)

agent.run("My name is Alice.")
result = agent.run("What's my name?")
print(result.output)  # "Your name is Alice."
```

### How it works

1. On each `run()` call, the agent prepends stored messages to the conversation.
2. After the agent responds, the new user message and assistant response are added to memory.
3. If `max_messages` is reached, the oldest messages are dropped (FIFO).

### Persistence

```python
memory.save("memory.json")

new_memory = AgentMemory()
new_memory.load("memory.json")

agent = Agent(name="assistant", llm=..., memory=new_memory)
result = agent.run("What's my name?")  # "Your name is Alice."
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | `int` | `20` | Maximum number of messages to retain |

---

## Long-term memory: `ComposableMemory` + blocks

`ComposableMemory` wraps a primary `AgentMemory` sliding window with a list of **memory blocks** that contribute SystemMessage fragments to every turn:

```
┌───────────────────────────────────────────────┐
│  ComposableMemory.get_context(query)          │
│                                               │
│    ← SystemMessage(s) from block[0].render    │
│    ← SystemMessage(s) from block[1].render    │
│    ← SystemMessage(s) from block[n].render    │
│    ← primary window (last N messages)         │
└───────────────────────────────────────────────┘
```

### Quick start

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.agent import (
    ComposableMemory, AgentMemory,
    StaticBlock, SummaryBlock, VectorBlock, FactExtractionBlock,
)
from fastaiagent.kb.backends.faiss import FaissVectorStore

llm = LLMClient(provider="openai", model="gpt-4o-mini")

memory = ComposableMemory(
    blocks=[
        StaticBlock("The user's name is Upendra and they prefer terse answers."),
        SummaryBlock(llm=llm, keep_last=10, summarize_every=5),
        VectorBlock(store=FaissVectorStore(dimension=384, index_type="flat")),
        FactExtractionBlock(llm=llm, max_facts=100),
    ],
    primary=AgentMemory(max_messages=20),
)

agent = Agent(name="assistant", llm=llm, memory=memory)
```

Each block is optional and independently useful. Use only what you need.

## Built-in blocks

### `StaticBlock`

A fixed system-level fact, injected on every turn. Zero state, zero LLM calls.

```python
StaticBlock("The user's timezone is UTC+1. Today's date is 2026-04-18.")
```

### `SummaryBlock`

Maintains a rolling LLM-generated summary of older turns. Refreshes every `summarize_every` messages, summarizing everything older than `keep_last`.

```python
SummaryBlock(
    llm=llm,
    keep_last=10,       # never summarize the N most recent messages
    summarize_every=5,  # refresh cadence
    max_chars=800,      # soft cap on summary length
)
```

**When to use**: long conversations that otherwise blow the context window. Cheaper than re-embedding everything, but introduces one extra LLM call every `summarize_every` turns.

### `VectorBlock`

Semantic recall over past messages. Each incoming message (above `min_content_chars`) is embedded and stored in a [`VectorStore`](../knowledge-base/backends.md). On each turn, the query is embedded and the top-k most similar past messages are surfaced.

```python
from fastaiagent.kb.backends.faiss import FaissVectorStore

VectorBlock(
    store=FaissVectorStore(dimension=384, index_type="flat"),
    top_k=5,
    namespace="default",       # tag so multiple blocks can share a store
    min_content_chars=10,      # skip trivial messages ("ok", "yes")
)
```

Any backend implementing the `VectorStore` protocol works — `FaissVectorStore` (default), `QdrantVectorStore`, `ChromaVectorStore`, your own. See [KB Backends](../knowledge-base/backends.md).

**When to use**: conversations that span days or sessions, where long-ago facts should be retrievable by meaning, not just recency.

### `FactExtractionBlock`

Uses a cheap LLM to extract durable facts from each user/assistant message and stores them as a dedup'd list. Renders as a bullet-point `Known facts: …` SystemMessage.

```python
FactExtractionBlock(
    llm=llm,            # use a fast model (gpt-4o-mini, claude-haiku)
    max_facts=200,      # cap; oldest drop on overflow
    extract_every=1,    # run extraction every N messages
)
```

**When to use**: user-focused assistants where you want stable facts ("user is allergic to peanuts", "user's kids are named Maya and Omar") to persist independently from the conversation log.

## Composing blocks

Block order matters — they render in declaration order, and the resulting SystemMessages appear in the prompt in that order. Typical ordering:

1. `StaticBlock` — hard facts that never change
2. `SummaryBlock` — what has happened so far
3. `FactExtractionBlock` — what we know about the user
4. `VectorBlock` — relevant past exchanges

Followed by the primary sliding window's recent messages.

## Persistence

`ComposableMemory.save(path)` writes to a directory:

```
path/
  ├── primary.json              # sliding window
  └── blocks/
      ├── summary.json          # SummaryBlock state
      ├── facts.json            # FactExtractionBlock state
      └── static.json           # (no-op for StaticBlock; file omitted)
```

`load(path)` restores into the same blocks, matched by `block.name`. You must reconstruct the blocks (with the same `llm` / `store` / `embedder`) before calling `load` — blocks that hold live resources (LLM clients, vector stores) are not themselves serialized.

```python
memory.save("/var/state/agent-alice")
# ... later, new process ...
memory = ComposableMemory(blocks=[SummaryBlock(llm=...), FactExtractionBlock(llm=...)])
memory.load("/var/state/agent-alice")
```

## Writing your own block

Subclass `MemoryBlock` and implement `on_message` and `render`:

```python
from fastaiagent.agent import MemoryBlock
from fastaiagent.llm.message import Message, SystemMessage


class MoodBlock(MemoryBlock):
    """Tracks the user's emoji reactions and pins the latest mood."""

    name = "mood"

    def __init__(self):
        self.latest_mood = ""

    def on_message(self, message: Message) -> None:
        content = message.content or ""
        for emoji in ("🎉", "😡", "😊", "😢"):
            if emoji in content:
                self.latest_mood = emoji

    def render(self, query: str):
        if not self.latest_mood:
            return []
        return [SystemMessage(f"User's latest mood: {self.latest_mood}")]
```

Then just drop it into `ComposableMemory(blocks=[MoodBlock(), ...])`.

If your block holds persistent state worth saving, override `save(path)` and `load(path)`. See [`fastaiagent/agent/memory_blocks.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/fastaiagent/agent/memory_blocks.py) for the shipped implementations.

## Safety

Each block runs inside a try/except inside `ComposableMemory`. A failing block is logged and skipped — it cannot break the agent run. Individual block state survives across the failure.

## Future work

Async parallel methods (`aon_message`, `arender`) are planned as an additive 0.5.x feature. The sync API shipped in 0.4.0 will not break when the async methods are added — same pattern as `Agent.run` / `Agent.arun`.

---

## Next Steps

- [Agents](index.md) — Core agent documentation
- [KB Backends](../knowledge-base/backends.md) — `VectorStore` backends used by `VectorBlock`
- [Middleware](middleware.md) — Transform agent messages and responses (complements memory)
- [Tracing](../tracing/index.md) — Debug agent execution with traces
