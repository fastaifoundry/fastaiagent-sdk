# CLI Reference

The `fastaiagent` CLI wraps the most common operational tasks: managing traces, running evals, serving agents, exposing them over MCP, and connecting to the Platform.

## Installation

Installed automatically with the SDK:

```bash
pip install fastaiagent
fastaiagent --help
```

## Top-level commands

| Command | Purpose |
|---|---|
| `fastaiagent version` | Show SDK version and which optional extras are installed |
| `fastaiagent connect` | Save Platform credentials and verify the API key |
| `fastaiagent disconnect` | Remove saved Platform credentials |
| `fastaiagent auth` | Inspect saved credentials (`status`, `env`) |
| `fastaiagent traces` | List, export, and search local traces |
| `fastaiagent replay` | Show, inspect, and fork traces for debugging |
| `fastaiagent eval` | Run evaluations from the command line |
| `fastaiagent prompts` | Browse the prompt registry |
| `fastaiagent kb` | Manage local knowledge bases |
| `fastaiagent agent` | Run an Agent or Chain as an HTTP service |
| `fastaiagent mcp` | Expose an Agent or Chain as an MCP server |
| `fastaiagent resume` | Resume a paused execution (durability) |
| `fastaiagent list-pending` | List pending interrupts awaiting human approval |
| `fastaiagent inspect` | Show checkpoint history for an execution |
| `fastaiagent setup-checkpointer` | Provision the durability backend (SQLite or Postgres) |
| `fastaiagent migrate` | Copy legacy `traces.db` / `checkpoints.db` / `.prompts/` into `local.db` |
| `fastaiagent export-trace` | Export one trace as a self-contained JSON file (same payload as the Local UI's Export button) |

---

## `fastaiagent version`

```bash
$ fastaiagent version
fastaiagent 1.0.0 [openai, anthropic, langchain, crewai, kb, qdrant, chroma, mcp-server, otel-export, postgres]
```

Brackets list the optional extras whose upstream package is importable. Useful when debugging "which extras did this env install?" in bug reports.

## `fastaiagent connect` / `disconnect` / `auth`

Persist Platform credentials to `~/.fastaiagent/credentials.toml` (mode `0600`) so scripts and CI don't need to pass the API key each time.

```bash
# Save + verify
fastaiagent connect --api-key fa_live_...

# Override target / project
fastaiagent connect --api-key fa_live_... --target https://platform.mycorp.com --project billing

# Inspect
fastaiagent auth status
#   Connected (source: file)
#     Target:  https://app.fastaiagent.net
#     Project: (default)
#     API key: fa_liv…ab34

# Print shell exports for sourcing
eval "$(fastaiagent auth env)"
# -> exports FASTAIAGENT_API_KEY, FASTAIAGENT_TARGET, FASTAIAGENT_PROJECT

# Remove
fastaiagent disconnect
```

**Python interaction.** `fa.connect(api_key=...)` in Python stays explicit — the CLI does not auto-connect your scripts. The intended pattern is to either (a) `eval "$(fastaiagent auth env)"` before starting your process and read `os.environ["FASTAIAGENT_API_KEY"]` in `fa.connect(...)`, or (b) parse `~/.fastaiagent/credentials.toml` yourself. Environment variables always win over the file.

## `fastaiagent traces`

```bash
# List recent traces
fastaiagent traces list
fastaiagent traces list --limit 50

# Export a trace as JSON
fastaiagent traces export <trace_id> --output trace.json
```

## `fastaiagent replay`

```bash
# Show replay steps
fastaiagent replay show <trace_id>

# Inspect a specific step
fastaiagent replay inspect <trace_id> <step>

# Fork a trace at a step, optionally modify the prompt or input, then rerun
fastaiagent replay fork <trace_id> --step 3 --prompt "New system prompt" \
    --output rerun.json

fastaiagent replay fork <trace_id> --input "Try a different question"
```

`replay fork` is the CLI surface for
`Replay.load(trace_id).fork_at(step).modify_prompt(...).modify_input(...).rerun()`.

## `fastaiagent eval`

```bash
# Run an evaluation
fastaiagent eval run --dataset test_cases.jsonl --scorer contains

# Compare two eval runs
fastaiagent eval compare <run_id_a> <run_id_b>
```

## `fastaiagent prompts`

```bash
# List registered prompts
fastaiagent prompts list

# Diff two versions
fastaiagent prompts diff <name> --from v1 --to v2
```

## `fastaiagent kb`

```bash
# List all KBs under the default root (.fastaiagent/kb/)
fastaiagent kb list

# List under a custom root
fastaiagent kb list --path /srv/fastaiagent/kb/

# Status of one KB
fastaiagent kb status --name product-docs

# Ingest a file or directory
fastaiagent kb add docs/           --name product-docs
fastaiagent kb add docs/refund.md  --name product-docs

# Delete by source file
fastaiagent kb delete docs/old.md --name product-docs

# Clear the whole KB
fastaiagent kb clear --name product-docs
```

## `fastaiagent agent serve`

Run any `Agent` or `Chain` as a FastAPI service that exposes the [uniform deployment contract](../deployment/index.md):

```bash
# path/to/file.py:attr
fastaiagent agent serve examples/01_simple_agent.py:agent --port 8000

# pkg.module:attr
fastaiagent agent serve mypkg.agents:research_bot --port 9000 --reload
```

Exposes:
- `GET  /health`
- `POST /run`         — `{"input": "..."}` → `{"output", "latency_ms", "tokens_used", "trace_id"}`
- `POST /run/stream`  — Server-Sent Events (Agent targets only)

If you need custom routes / auth / middleware, copy [`examples/33_deploy_fastapi.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/33_deploy_fastapi.py) and extend it directly instead.

Requires: `pip install fastapi 'uvicorn[standard]'` (or `fastaiagent[all]`).

## `fastaiagent mcp serve`

Expose an `Agent` or `Chain` as an MCP server over stdio — registers with Claude Desktop, Cursor, Continue, Zed, and any other MCP client:

```bash
fastaiagent mcp serve path/to/my_agent.py:agent
fastaiagent mcp serve path/to/my_agent.py:agent --expose-tools --name research_bot
```

See [docs/tools/mcp-server.md](../tools/mcp-server.md) for Claude Desktop / Cursor config snippets.

Requires: `pip install 'fastaiagent[mcp-server]'`.

---

## Durability commands

The four commands below cover the v1.0 [durability](../durability/index.md) workflow: pause an execution with `interrupt()`, list what's waiting, then resume from any process.

### `fastaiagent list-pending`

Rich-rendered table of every pending interrupt in the local store:

```bash
fastaiagent list-pending
fastaiagent list-pending --db-path /var/lib/fastaiagent/local.db
fastaiagent list-pending --limit 50
```

### `fastaiagent inspect <execution_id>`

Checkpoint history for one execution — node-by-node statuses, timestamps, and `agent_path` for multi-agent topologies:

```bash
fastaiagent inspect refund-abc
fastaiagent inspect refund-abc --db-path ./.fastaiagent/local.db
```

Exits **1** when the execution has no checkpoints.

### `fastaiagent resume <execution_id>`

Loads the runner via a Python entrypoint and calls `aresume(...)`:

```bash
# Approve
fastaiagent resume refund-abc --runner myapp.flows:build_chain

# Reject with a reason
fastaiagent resume refund-abc \
    --runner myapp.flows:build_chain \
    --value '{"approved": false, "metadata": {"reason": "amount above threshold"}}'
```

Exits **2** on `AlreadyResumed` (another resumer claimed the pending row first — a deterministic outcome, not a bug).

### `fastaiagent setup-checkpointer`

Provisions the durability backend's schema. Idempotent for both backends.

```bash
# Local SQLite (default)
fastaiagent setup-checkpointer

# Postgres
fastaiagent setup-checkpointer \
    --backend postgres \
    --connection-string "$DATABASE_URL"

# Custom Postgres schema (so two installs share one DB)
fastaiagent setup-checkpointer \
    --backend postgres \
    --connection-string "$DATABASE_URL" \
    --schema fa_prod
```

See [docs/durability/checkpointers.md](../durability/checkpointers.md) for the full backend reference.

---

## Environment variables

| Variable | Used by |
|---|---|
| `FASTAIAGENT_API_KEY` | Platform connection (Python + CLI) |
| `FASTAIAGENT_TARGET` | Platform URL override |
| `FASTAIAGENT_PROJECT` | Platform project override |
| `FASTAIAGENT_LOCAL_DB` | Local SQLite store (traces, checkpoints, idempotency, prompts) |
| `FASTAIAGENT_CHECKPOINT_DB_PATH` | Override checkpoint store path (legacy; prefer `FASTAIAGENT_LOCAL_DB`) |
| `FASTAIAGENT_LIVE_OPENAI_MODEL` | Override OpenAI model in live tests |
| `FASTAIAGENT_LIVE_ANTHROPIC_MODEL` | Override Anthropic model in live tests |
| `OPENAI_API_KEY` | LLM calls (OpenAI) |
| `ANTHROPIC_API_KEY` | LLM calls (Anthropic) |

## `fastaiagent export-trace`

Export a single trace as a self-contained JSON file. Reads the local
SQLite directly — no UI server required.

```bash
fastaiagent export-trace --trace-id <id> --output trace.json
fastaiagent export-trace --trace-id <id> --output trace-full.json \
    --include-attachments --include-checkpoint-state
```

Flags:

- `--trace-id <id>` (required) — the trace to export.
- `--output <path>` (default `trace.json`) — destination file.
- `--include-attachments` — embed image / PDF bytes (base64) in the
  JSON. Off by default; files can balloon to many MB.
- `--include-checkpoint-state` — embed full `state_snapshot` blocks
  for each checkpoint. Off by default.
- `--db <path>` — override the local DB; defaults to whatever
  `SDKConfig.local_db_path` resolves to (typically
  `.fastaiagent/local.db`).

Same JSON shape comes out of the Local UI's Export dialog. See
[Export trace as JSON](../ui/export-trace.md) for the schema.

## Next Steps

- [Durability — pause and resume](../durability/index.md)
- [Tracing Guide](../tracing/index.md)
- [Agent Replay](../replay/index.md)
- [Evaluation](../evaluation/index.md)
- [MCP Server](../tools/mcp-server.md)
- [Deployment Recipes](../deployment/index.md)
