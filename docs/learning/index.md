# Learning from traces

Most agent SDKs ship the runtime; few ship the **improvement loop**.

`fastaiagent.learn` reads completed traces out of `local.db`, extracts durable per-user / per-project / per-agent facts via an LLM, and re-injects them into future runs through [`PersistentFactBlock`](../agents/memory.md). The loop runs entirely on the developer's local machine — no platform dependency.

This is the SDK's take on the "continual learning" framing Harrison Chase has been writing about: traces are the substrate; agents improve along the *context* layer (memory, scoped facts, learned skills) without retraining.

## Layout

| Doc | What it covers |
|---|---|
| [Memory loop](memory-loop.md) | The end-to-end flow: traces → `fastaiagent learn` → `learned_memory` table → `PersistentFactBlock` |
| [`fastaiagent learn` CLI](../cli/learn.md) | Flags, scopes, dry-run, conflict resolution |
| [Self-improving agents](../concepts/self-improving-agents.md) | Conceptual framing — what we extract, what we don't, why memory-only at v1 |

## What v1 ships

- **Memory only.** Durable user/project/agent facts. No skill extraction, no prompt mutation. (Those need replay-eval to avoid drift — out of scope for v1.)
- **Offline batch.** A `fastaiagent learn` CLI runs over the trace window you specify. No online mid-run learning.
- **Local first.** Reads `local.db`, writes the new `learned_memory` table. Push to platform is unidirectional and unchanged.
- **PII-safe by default.** Only `--scope agent` runs without an opt-in; `user` / `project` scopes require `--allow-personal`.

## What's coming next

Tracked as future work in the plan file:

- Skill extraction (reusable mini-procedures).
- Meta-Harness style prompt/harness mutation.
- Replay-eval infrastructure (prerequisite for both above).
- Online mid-run learning.
- UI for human review / annotation of learned facts.
