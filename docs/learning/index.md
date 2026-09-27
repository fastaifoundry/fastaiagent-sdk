# Learning from traces

Most agent SDKs ship the runtime; few ship the **improvement loop**.

`fastaiagent.learn` reads completed traces out of `local.db`, extracts durable facts via an LLM — filed under one agent, user or project per run — and re-injects them into future runs through [`PersistentFactBlock`](../agents/memory.md). It needs no platform: traces and facts stay on your machine, though the trace text is sent to the extraction LLM (OpenAI by default). For how this fits with the rest of memory, see [How memory works](../agents/memory-concepts.md).

This is the SDK's take on the "continual learning" framing Harrison Chase has been writing about: traces are the substrate; agents improve along the *context* layer (memory, scoped facts, learned skills) without retraining.

## Layout

| Doc | What it covers |
|---|---|
| [Memory loop](memory-loop.md) | The end-to-end flow: traces → `fastaiagent learn` → `learned_memory` table → `PersistentFactBlock` |
| [`fastaiagent learn` CLI](../cli/learn.md) | Flags, scopes, dry-run, conflict resolution |
| [Self-improving agents](../concepts/self-improving-agents.md) | Conceptual framing — what we extract, what we don't, why memory-only at v1 |

## What v1 ships

- **Memory only.** Durable user/project/agent facts. No skill extraction, no prompt mutation. (Those need replay-eval to avoid drift — out of scope for v1.)
- **Offline batch.** The `fastaiagent learn` CLI mines the traces you point it at (`--window`, `--agent`), skipping ones it already mined. For learning *during* a run, use `Memory(learn=llm)`.
- **Local first.** Reads `local.db`, writes the new `learned_memory` table. Push to platform is unidirectional and unchanged.
- **PII guardrails on the CLI.** Only `--scope agent` runs without an opt-in; `user` / `project` scopes require `--allow-personal` (plus `--agent`, `--scope-id` and `--attribute-all`). The extraction prompt asks the model to skip PII — best-effort, not a guarantee. `Memory(learn=)` has no such gate: it stores what users say about themselves.

## What's coming next

Tracked as future work in the plan file:

- Skill extraction (reusable mini-procedures).
- Meta-Harness style prompt/harness mutation.
- Replay-eval infrastructure (prerequisite for both above).
- Human review / annotation of learned facts in the Local UI (the Memory page shows them read-only today; the Enterprise plane has a curation queue).
