# `fastaiagent learn`

Extract durable facts from past traces and re-inject them via `PersistentFactBlock`.

```
fastaiagent learn [--scope SCOPE] [--scope-id ID] [--window N]
                  [--max-facts N] [--model NAME] [--provider NAME]
                  [--dry-run] [--allow-personal]
fastaiagent learn list [--scope SCOPE] [--scope-id ID] [--limit N]
                       [--show-superseded]
fastaiagent learn supersede OLD_ID NEW_ID
```

## Default action — extract

Without a subcommand, `fastaiagent learn` runs the extraction loop over the configured trace window:

```sh
# Default: agent-scope only, last 24h, no PII risk.
fastaiagent learn --scope-id my-agent

# Preview — no rows written.
fastaiagent learn --scope-id my-agent --dry-run

# Wider window.
fastaiagent learn --scope-id my-agent --window 168    # last week
```

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--scope` | `agent` | One of `user` \| `project` \| `agent`. |
| `--scope-id` | `""` | Identifier within scope. Empty means "no specific id". |
| `--project-id` | `""` | DB-side project filter (matches v4 project scoping). |
| `--window` / `--last-hours` | `24` | Trace history window in hours. |
| `--max-facts` | `10` | Cap per trace. |
| `--model` | `gpt-4o-mini` | Extractor LLM — cheap + fast recommended. |
| `--provider` | `openai` | Any provider supported by `LLMClient`. |
| `--dry-run` | off | Show candidates without writing. |
| `--allow-personal` | off | **Required** for `--scope user` and `--scope project`. Default-off so PII extraction is always an explicit opt-in. |

## `list` — inspect what's stored

```sh
fastaiagent learn list --scope agent --scope-id my-agent
fastaiagent learn list --show-superseded   # include audit history
```

Output is a Rich table with `id`, `scope`, `scope_id`, `fact`, `source_trace_id`, and status (active or `superseded by N`).

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

The block is **read-only at runtime** — only the CLI writes new facts.

## Privacy

`fastaiagent learn` extracts only `agent`-scoped facts by default. Both `--scope user` and `--scope project` require an explicit `--allow-personal` flag. This is a deliberate guardrail — PII extraction should never be a surprise side effect of running the CLI.

The extraction prompt also instructs the model to skip names, emails, phone numbers, and addresses. This is best-effort, not a guarantee. Always review extracted facts before deploying to production.
