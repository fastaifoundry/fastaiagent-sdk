# `fastaiagent learn`

Extract durable facts from past traces and re-inject them via `PersistentFactBlock`.

```
fastaiagent learn [--scope SCOPE] [--scope-id ID] [--agent NAME] [--window N]
                  [--max-traces N] [--max-facts N] [--model NAME] [--provider NAME]
                  [--dry-run] [--reprocess] [--allow-personal] [--attribute-all]
fastaiagent learn list [--scope SCOPE] [--scope-id ID] [--limit N]
                       [--show-superseded]
fastaiagent learn supersede OLD_ID NEW_ID
```

## Default action — extract

Without a subcommand, `fastaiagent learn` mines the traces you point it at — newest first:

```sh
# my-agent's traces from the last 24h, filed as agent:my-agent facts.
fastaiagent learn --scope-id my-agent --agent my-agent

# Preview — no rows written, nothing recorded as mined.
fastaiagent learn --scope-id my-agent --agent my-agent --dry-run

# Wider window.
fastaiagent learn --scope-id my-agent --agent my-agent --window 168    # last week
```

Which traces it reads:

- only those that **started within `--window` hours**;
- with `--agent`, only traces in which **that agent ran** — at the root, or as a child span inside a swarm, chain or supervisor. Without it, every agent's traces in the window are read and every fact is filed under the one `--scope-id`;
- never its **own** extraction calls (they run under a `learn.extract` root span);
- not traces **already mined** for this scope and id — a re-run only reads new traces, and a trace that yielded nothing isn't billed again. `--reprocess` mines them again. A trace whose extraction failed isn't recorded, so the next run retries it;
- at most `--max-traces` of them.

The summary reports **new** facts separately from facts that were already known.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--scope` | `agent` | One of `user` \| `project` \| `agent`. |
| `--scope-id` | `""` | Identifier within scope. Required for `user` / `project`. |
| `--agent` | — | Only mine traces in which this agent ran. Required for `user` / `project`. |
| `--project-id` | `""` | The project partition the facts are **stored** under. It does not filter which traces are read. |
| `--window` / `--last-hours` | `24` | Only traces that started within this many hours. |
| `--max-traces` | `100` | Mine at most this many traces per run, newest first. |
| `--reprocess` | off | Mine traces again even if an earlier run already did for this scope and id. |
| `--max-facts` | `10` | Cap per trace. |
| `--model` | `gpt-4o-mini` | Extractor LLM — cheap + fast recommended. |
| `--provider` | `openai` | Any provider supported by `LLMClient`. |
| `--dry-run` | off | Show candidates without writing. |
| `--allow-personal` | off | **Required** for `--scope user` and `--scope project`. Default-off so PII extraction is always an explicit opt-in. |
| `--attribute-all` | off | **Required** for `--scope user` and `--scope project`: confirms every selected trace belongs to `--scope-id` (see Privacy). |

## `list` — inspect what's stored

```sh
fastaiagent learn list --scope agent --scope-id my-agent
fastaiagent learn list --show-superseded   # include audit history
```

Output is a Rich table with `id`, `scope`, `scope_id`, `fact` (cut to 120 characters), `source` (the first 12 characters of the source trace id), and status (active or `superseded by N`).

## `supersede` — manual conflict resolution

```sh
fastaiagent learn supersede 12 34
# → ok 12 superseded by 34
```

Marks fact `12` as replaced by fact `34`. The old row is preserved for audit; the new row becomes the active one for any consumer that filters on `superseded_by IS NULL` (which `list_active` and `PersistentFactBlock` do).

## Pairing with `PersistentFactBlock`

```python
import fastaiagent as fa
from fastaiagent.agent.memory_blocks import PersistentFactBlock

memory = fa.ComposableMemory(
    primary=fa.AgentMemory(),
    blocks=[PersistentFactBlock(scope="agent", scope_id="my-agent", max_facts=30)],
)
agent = fa.Agent(name="my-agent", system_prompt="…", llm=llm, memory=memory)
```

The block is **read-only at runtime**. New facts come from this CLI, from `Memory(learn=llm)` / `FactExtractionBlock(persist=True)` during runs, and from `Memory.persist` / `MemoryStore.add` in your own code.

## Privacy

`fastaiagent learn` extracts only `agent`-scoped facts by default. `--scope user` and `--scope project` need four things:

- `--allow-personal`, so PII extraction is never a surprise side effect;
- a non-empty `--scope-id`, since facts under an empty id can't be read back;
- `--agent`, so the run reads one agent's traces rather than everyone's;
- `--attribute-all`, because **traces carry no user id**: every trace the run reads is filed under that one `--scope-id`. Use it only when the selected traces all belong to that user or project.

The extraction prompt also instructs the model to skip names, emails, phone numbers, and addresses. This is best-effort, not a guarantee. Always review extracted facts before deploying to production.
