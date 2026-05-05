# Personal Assistant (`ComposableMemory` showcase)

A long-lived REPL personal assistant built with [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk) v1.6.1+. The single most distinctive thing about this template: **it uses every memory-block type the SDK ships, simultaneously**, plumbed through `ComposableMemory` with on-disk persistence between sessions.

Inspired by [Payhawk's Memory Bank](https://payhawk.com/blog/memory-bank), [mem.ai](https://mem.ai), and [Rewind](https://rewind.ai) — but actually runnable on your laptop in 60 seconds.

```
                    ┌─────────────────────┐
        every       │ ComposableMemory    │       primary
        turn  ───►  │                     │  ◄──  AgentMemory(20)
                    │  StaticBlock        │       sliding
                    │  + SummaryBlock     │       window
                    │  + VectorBlock      │
                    │  + FactExtractionBlock
                    └─────────┬───────────┘
                              ▼
                          ┌────────┐  + tools:
                          │ Agent  │     add_note, search_notes,
                          │  REPL  │     list_facts, today
                          └────────┘
                              │
                              ▼
                       memory.save() → .fastaiagent/memory/
                       memory.load() ← (next invocation)
```

**What this example demonstrates** (vs. prior templates):

- **Every memory-block type, wired together** — the canonical demo of `ComposableMemory`. Each block plays a different role; their composition is the point.
- **Cross-session persistence** — `memory.save()` writes each block's state to its own JSON file under `.fastaiagent/memory/blocks/`; the next REPL startup reads them back and the assistant picks up where it left off.
- **Memory introspection from the agent** — `list_facts()` is a tool that lets the agent (and the user) see what `FactExtractionBlock` has captured. Demystifies the "what does the assistant know about me" question.
- **VectorBlock with `FaissVectorStore`** — semantic recall over every prior message, top-k=5, 384-dim embeddings.
- **`StaticBlock` for pinned identity** — the user's name, role, timezone, and today's date appear in every turn's system context. Cheapest possible memory layer.
- **`PromptRegistry`-backed system prompt** — the assistant's tone-and-tools prompt is registered locally on first run as `personal-assistant-prompt`; subsequent runs read the latest version. Edit it in the Local UI's Playground (`http://127.0.0.1:7842/playground/personal-assistant-prompt`) and the next REPL turn picks up the change with no restart.

---

## Quick Start

```bash
# from the SDK root
pip install -e .
cd examples/personal-assistant
cp .env.example .env       # OPENAI_API_KEY required; USER_NAME / ROLE / TZ optional
pip install -r requirements.txt   # installs faiss-cpu

python agent.py                       # interactive REPL with memory persistence
python agent.py --query "Hi! I'm Riley, staff engineer at Strato Labs."
python agent.py --query "What do you remember about me?"   # (different process)
python agent.py --reset               # wipe saved memory and start fresh
```

The first turn introduces yourself. Subsequent turns — even from a *different process* — recall what you said. The `FactExtractionBlock` lifts durable facts (`"User's name is Riley."`, `"User works at Strato Labs."`); `VectorBlock` semantically recalls prior turns when relevant; `SummaryBlock` compresses older turns once you cross 4–6 messages.

---

## Files

```
personal-assistant/
├── README.md
├── .env.example
├── requirements.txt
├── memory_setup.py      # builds ComposableMemory with all 4 blocks + persistence
├── tools.py             # add_note / search_notes / list_facts / today
├── agent.py             # REPL with memory load-on-startup / save-on-exit
├── eval_suite.py        # 6-turn canonical session + 3 memory scorers
└── tests/
    └── test_smoke.py    # 8 offline regression tests
```

---

## How it's wired

### Building the memory ([memory_setup.py](memory_setup.py))

```python
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock, StaticBlock, SummaryBlock, VectorBlock,
)
from fastaiagent.kb.backends.faiss import FaissVectorStore

memory = ComposableMemory(
    blocks=[
        StaticBlock(text=identity_text),                       # pinned
        SummaryBlock(llm=cheap_llm, keep_last=6,
                     summarize_every=4, max_chars=600),         # rolling
        VectorBlock(store=FaissVectorStore(dimension=384,
                                            index_type="flat"),
                    top_k=5, namespace="default",
                    min_content_chars=12),                      # semantic
        FactExtractionBlock(llm=cheap_llm, max_facts=120,
                            extract_every=1),                   # durable
    ],
    primary=AgentMemory(max_messages=20),
)
```

Every turn, `memory.get_context(query)` renders each block in declaration order. The agent's prompt looks like:

```
[system] Conversation summary so far: <SummaryBlock output>
[system] User identity (pinned every turn): Name: Riley. Role: ...
[system] Relevant prior exchanges:
         - [user] I'm rebuilding our deployment pipeline...
         - [assistant] Here are three Argo CD hook tips...
[system] Known facts:
         - User's name is Riley.
         - User works at Strato Labs.
[user] What did we discuss about deployment earlier?
[assistant] (synthesized response)
```

### Persistence ([memory_setup.py](memory_setup.py))

```python
def save_memory(memory: ComposableMemory, memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory.save(memory_dir)   # writes primary.json + blocks/<name>.json
```

`ComposableMemory.save(dir)` walks every block and calls `block.save(dir/blocks/<name>.json)`. `block.load(...)` reads it back on next startup.

| Block | Persisted state |
|---|---|
| `StaticBlock` | `text` (rebuilt fresh on startup, since "today's date" advances) |
| `SummaryBlock` | `{summary: str, messages_seen: int}` |
| `VectorBlock` | (state lives in the `FaissVectorStore` you pass in — bring your own persistence) |
| `FactExtractionBlock` | `[fact, fact, ...]` JSON array |
| Primary `AgentMemory(20)` | `[message, ...]` (the literal 20 most recent messages) |

The `VectorBlock` deliberately doesn't persist — different `VectorStore` backends have different durability stories (Faiss in-memory vs Chroma on-disk vs Qdrant remote). Wire your own if you need the embeddings to survive restart.

### Tools ([tools.py](tools.py))

| Tool | Purpose |
|---|---|
| `add_note(text)` | Append to `.fastaiagent/notes.jsonl`. Use when the user asks for a specific durable note. |
| `search_notes(query)` | Substring search over the notes log, top-3 most recent matches. Distinct from `VectorBlock`'s automatic recall. |
| `list_facts()` | Show what `FactExtractionBlock` has captured. Demystifies what the assistant knows. |
| `today()` | Current ISO date. The `StaticBlock` pins this at startup, but it goes stale at midnight without a tool call. |

---

## Running each entry point

```bash
# Interactive REPL — recommended for exercising memory across many turns
python agent.py

# Single-shot — useful for testing / cron
python agent.py --query "What's on my plate today?"

# Wipe memory and start fresh
python agent.py --reset

# 6-turn canonical eval session — verifies all 4 blocks fired correctly
python eval_suite.py

# Smoke tests — 8 offline tests, ~4s
python -m pytest tests/
```

---

## Local UI

```bash
fastaiagent ui start            # http://127.0.0.1:7842
```

What this example populates:

- **`/traces`** — every REPL turn lands as `agent.personal-assistant`. Inside: the LLM call plus any tool span (`tool.add_note`, `tool.list_facts`, etc.). The memory blocks themselves don't currently emit spans — they run inline as part of `_build_messages`.
- **`/agents`** — `personal-assistant` (and `eval-suite`'s `personal-assistant-eval`) with run counts.
- **`/evals`** — `personal-assistant eval` against `canonical-session` with the three memory-block scorers.

There's no `/memory` page yet in the Local UI — for direct inspection of `FactExtractionBlock` state today, use the `list_facts` tool inside the REPL or `cat .fastaiagent/memory/blocks/facts.json`.

---

## Customising

**Swap to Chroma or Qdrant for production** — same `VectorStore` protocol, swap one import in `memory_setup.py`:

```python
from fastaiagent.kb.backends.qdrant import QdrantVectorStore
return QdrantVectorStore(host="localhost", port=6333, collection="pa-memory")
```

**Add your own memory block** — implement the `MemoryBlock` protocol (`on_message`, `render`, optionally `save`/`load`). Example: a `MoodBlock` that infers user mood from each message and pins the latest:

```python
class MoodBlock(MemoryBlock):
    name = "mood"
    def __init__(self, llm: fa.LLMClient): ...
    def on_message(self, message): ...   # update self.latest_mood
    def render(self, query): return [SystemMessage(f"User mood: {self.latest_mood}")]
```

Drop it into `ComposableMemory(blocks=[MoodBlock(llm=...), ...])`.

**Tune cost** — `SummaryBlock` and `FactExtractionBlock` make an LLM call on a cadence (`summarize_every` / `extract_every`). The defaults in `memory_setup.py` use `gpt-4o-mini` for both; bump or skip via `MEMORY_LLM_MODEL` env. To run *fully* without per-turn LLM overhead, drop these two blocks and keep just `StaticBlock` + `VectorBlock` — you lose summary compression and fact extraction but the rest still works.

**Multi-user** — pass a per-user `namespace` into the `VectorBlock` (`namespace=user_id`) and a per-user `memory_dir`. The `FactExtractionBlock` is per-instance, so you'd build a fresh `ComposableMemory` per user.

---

## What this example does NOT demonstrate

- **HITL approval gates** — see `examples/customer-support-agent/` or `examples/sales-sdr-agent/`.
- **Multi-agent orchestration** — see `examples/research-agent/` (Supervisor) or `examples/sales-sdr-agent/` (Chain DAG).
- **Multimodal** — the assistant is text-only. Add `fa.Image` / `fa.PDF` to the user input shape if you want it to look at screenshots / PDFs.
- **Tool-call HITL** — `add_note` writes immediately; for high-trust environments wrap the body in `fa.interrupt()` like the SDR template does for `send_outreach_email`.

---

## License

Apache 2.0 — same as the SDK.
