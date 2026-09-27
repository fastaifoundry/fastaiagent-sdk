# How memory works

The mental model behind [`Memory`](memory.md), the [trace learning loop](../learning/memory-loop.md) and plane facts — what is kept, where, for whom, and when it is written. Read this once; the other memory pages assume it.

## Four kinds of memory

| Kind | What it holds | Lives | Scoped by | Survives a restart |
|---|---|---|---|---|
| **Conversation window** | the recent messages, verbatim (`AgentMemory`, `Memory(window=)`) | the process | the `Memory` object — one per user with a `user_id` resolver | no — save it with `mem.for_user(id).save(path)` |
| **In-conversation blocks** | a running summary, semantic recall of past messages, facts extracted this session | the process (recall can use your vector store) | the same subject as the window | the summary and extracted facts via `save()`; an `"auto"` recall index, no |
| **Durable facts** | short statements such as "Alice is on the Pro plan" | the fact store — `local.db`, Postgres or Redis (the `learned_memory` table) | `scope` + `scope_id` (+ `project_id`) | yes |
| **Plane facts** | facts curated and approved on the Enterprise plane | the plane | the plane's agent id | yes — the SDK only reads them |

The first two are *working* memory: fast, per conversation, gone when the process goes unless you save them. Durable facts are the long-term layer every run reads.

## Tiers and scopes — two names for the same partition

`Memory` speaks in **tiers**; the store, the `fastaiagent learn` CLI and the Local UI's Memory page speak in **scopes**:

| `Memory` tier | Store scope | `scope_id` | Meaning |
|---|---|---|---|
| `global` | `agent` | the `Memory`'s `agent_id` | true for everyone using the agent |
| `user` | `user` | the user id (`id=`, or the resolver) | about one user |
| — | `project` | a project key | about one project (CLI / low-level store) |
| `session` | — | — | the conversation window; not a durable store |

`project_id` is a separate **tenant partition** under all of them: a fact stored under one `project_id` is invisible to reads under another.

A global fact is injected only by a `Memory` with the **same** `agent_id`. `Memory(agent_id="")` raises, because an empty id would read every agent's facts; persisting a global fact on a `Memory` without an `agent_id` warns, because no agent would ever see it.

## One turn, step by step

**Read**, before the model is called — the prompt is built in this order:

1. the agent's system prompt;
2. one system message per memory block, in the order the blocks are declared — global facts, the user's facts, the summary, recalled messages;
3. the conversation window (the recent user and assistant messages);
4. any `messages=` history you passed;
5. this turn's input.

**Run** — the model and its tools. Tool calls and results are **not** added to memory.

**Write**, only if the run succeeded — the user's input and the final answer are added: to the window, and shown to every block (`learn=` extracts facts from the user's message; recall embeds both). Nothing is written when the run fails, is blocked by a guardrail, stops at the iteration limit, or pauses for approval.

- A **resumed** or **forked** run records the question it resumed.
- A **swarm** run is recorded once — the user's request and the final answer — however many hand-offs it took.
- **Streaming** (`astream`) reads and writes exactly like `run`.
- **Middleware** such as `RedactPII` changes what the *model* is sent, never the user's messages in memory. The stored reply is the agent's answer after middleware — redacted, with `RedactPII` — for `run` and `astream` alike.

## Who writes durable facts

| Producer | When | Reads from | Scope | Confidence | Source on the Memory page |
|---|---|---|---|---|---|
| `Memory.persist` / `update`, `MemoryStore.add` | when your code calls it | — | any | `1.0` unless you set it | `manual` |
| `Memory(learn=llm)` (`FactExtractionBlock(persist=True)`) | during each run | the user's messages | `user` | `0.6` | `trace` |
| `fastaiagent learn` (`run_extraction`) | when you run it | past traces in `local.db` | `agent` by default | `1.0` | `trace` |
| the Enterprise plane | curated centrally | the plane's own traces | the plane's agent | — | read via `PlaneFactBlock`, never written by the SDK |

### The two "learn"s

They share a name and do different jobs:

| | `Memory(learn=llm)` | `fastaiagent learn` |
|---|---|---|
| Runs | online, inside every run | offline, when you invoke it |
| Reads | this conversation's user messages | stored traces — the window and agent you choose |
| Files facts under | the resolved user | the one `--scope` / `--scope-id` you pass |
| Bounded by | `max_learned_facts` (newest 200 per user kept) | `--max-traces`, and each trace is mined once |
| PII gate | none — it stores what users say about themselves | `--allow-personal` (+ `--attribute-all`) for user / project scope |

## Isolation

- **Per-user windows.** `Memory(user_id=lambda ctx: ...)` resolves the user from each run's `context=` and keeps a separate window and blocks per user — for at most `max_users` users (default 10,000), dropping the least recently used. A dropped user keeps their durable facts and starts a fresh conversation.
- **Unresolved means no memory.** A run whose user can't be resolved — no context, a `None` id, a resolver that raises — sees only global facts and writes nothing, so anonymous callers never share a window.
- **Facts are scoped.** At user and project scope an empty id reads nothing; `"*"` reads all on purpose.
- **Recall is namespaced.** A vector store shared by every user keeps each user's recall under `user:<id>`.

## What survives a restart

| | After a restart |
|---|---|
| Durable facts (any backend) | all there |
| Semantic fact search (`semantic=`) | works — each user's facts are re-indexed from the store on their first query |
| Conversation windows | gone unless you `save()` / `load()` them |
| `recall="auto"` index | gone; pass your own persistent `VectorStore` to keep recall |
| Plane facts | read from the plane again |

## Privacy

- **Local capture is full fidelity.** Traces in `local.db` keep memory reads, recalled snippets and the user id (`memory.scope_id`).
- **The user id stays out of the prompt.** A user's facts reach the model under the heading `Learned facts (user):` — the id (often an email) is not sent to the model provider. Agent and project facts keep their id in the heading (`Learned facts (agent:support):`).
- **Export is gated.** With `FASTAIAGENT_TRACE_PAYLOADS=0`, those payload attributes are dropped from exported spans, and `PlaneFactBlock` stops sending the user's question to the plane.
- **Redaction is for the model.** `RedactPII` redacts what the model sees and says; memory keeps the user's words as said and the model's reply as redacted. To keep PII out of memory entirely, redact the input before it reaches the agent.

## Where to go next

- [Memory reference](memory.md) — every keyword and block.
- [Memory, end to end](../tutorials/memory-guide.md) — the tutorial.
- [Memory loop](../learning/memory-loop.md) and [`fastaiagent learn`](../cli/learn.md) — the offline path.
