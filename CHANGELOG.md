# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.4] - 2026-04-21

### Added — per-agent Tools directory

New **Tools** section on `/agents/<name>` shows what each agent is
registered with and what it actually calls. One row per tool with:

- **Origin chip** — color-coded by kind: `function` (green) for
  `@tool` / `FunctionTool`, `kb` (blue) for `LocalKB.as_tool()`,
  `mcp` (purple) for MCP-backed tools, `rest` (amber) for `RESTTool`,
  `custom` (grey) for user-defined `Tool` subclasses, `unknown` (red)
  for LLM-hallucinated names that weren't registered.
- Call count, success rate, avg latency, last-used timestamp.
- Status badges: `unused` when a registered tool has never been
  called (dead-code signal), `unregistered` when the LLM invoked a
  name that wasn't in the agent's `tools=[...]` list (hallucination
  signal).

### Added — `Tool.origin` class attribute

`Tool` gets a public `origin: str = "custom"` class attr. Subclasses
override: `FunctionTool → "function"`, `MCPTool → "mcp"`,
`RESTTool → "rest"`. `LocalKB.as_tool()` instance-overrides to
`"kb"`. Serialized by `Tool.to_dict()`, so the SDK's existing
`agent.tools` span attribute is now automatically origin-typed on
every new run — no agent-side changes needed.

Also emits `tool.origin` on every `tool.*` execution span (falls
back to `"unknown"` for LLM-hallucinated tool names).

Traces emitted before 0.9.4 render with the `unknown` chip.

### Added — search bars on `/agents`, `/workflows`, `/kb`

Each directory page gets a lightweight client-side search input.
Substring match on the already-loaded list, `useMemo`-cached, zero
backend calls, zero debounce, zero new deps. The header description
updates to `"N of M match 'query'"` while filtering. Empty state
distinguishes "no data yet" vs "no match for this search."

`/traces` already has backend-backed full-text search via the
`q` query param — no change there.

### Added — `examples/41_agent_tools.py`

Runnable demo that registers one tool of each origin (`@tool` +
`LocalKB.as_tool()` + custom `Tool` subclass), runs the agent so two
of three get called, and points you at `/agents/tool-curator` to see
the Tools section populated with all three chip colors and an
`unused` badge.

## [0.9.3] - 2026-04-21

### Added — Eval Compare page + richer Run Detail

New `/evals/compare` page pairs two eval runs side-by-side. Cases are
matched by `ordinal` (falling back to `input` equality for reordered
datasets) and bucketed into **regressed** (passed in A, failed in B),
**improved** (failed in A, passed in B), and **unchanged**. Each
regressed / improved card renders two diffs — expected vs actual-B,
then actual-A vs actual-B — with per-scorer delta chips
ring-highlighted for scorers whose outcome flipped. Header stats show
pass-rate delta and cost delta.

`/evals/:id` gets:

- Expandable case rows with an inline `expected vs actual` diff
  powered by `react-diff-viewer-continued`.
- A filter bar (outcome / scorer / substring search).
- A scorer-chip header row with per-scorer `pass/total` counts,
  colored by pass-rate (green ≥90%, amber 70–89%, red <70%).
- Explicit **Trace** and **Replay** link buttons on every case.

`/evals` (list) gets **Cost** and **Avg latency** columns derived
from joining `eval_cases.trace_id` back to the `spans` table.

### Added — `results.run_id` on `evaluate()`

`evaluate()` and `EvalResults.persist_local()` now populate
`results.run_id` so callers can deep-link into the Local UI
(e.g. `/evals/<run_id>` or `/evals/compare?a=…&b=…`) without
re-querying the DB. Purely additive; existing code is unaffected.

### Added — `examples/40_evals_compare.py`

Runnable before/after demo: one vague system prompt (produces 0%),
one tight system prompt (produces 100%), prints the exact compare
URL. Drops two real eval runs into local.db ready to browse.

## [0.9.2] - 2026-04-21

### Added — Workflows directory in the Local UI

New `/workflows` sidebar entry enumerates every **chain, swarm, and
supervisor** that has produced a root span. Cards per workflow show
node count, runs, success rate, avg latency, avg cost, and last run;
top-of-page tabs filter by runner type. Click a card →
`/workflows/:type/:name` detail page with a filtered trace list for
that specific workflow (via a new `runner_name` query filter on
`/api/traces`).

The sidebar's previous `// AGENTS` one-item section is regrouped as
`// WORKFLOWS & AGENTS` holding both links.

### Changed — Events tab now renders exception tracebacks properly

The Trace Detail page's **Events** tab was a raw JSON dump. It now:

- Renders OpenTelemetry `record_exception` events as a dedicated
  error card: `exception.type` prominent, `exception.message` on one
  line, full `exception.stacktrace` hidden behind an expandable
  **Traceback** disclosure.
- Renders generic / custom events (`span.add_event(...)`) with a
  name row plus a collapsible JSON attributes viewer.
- Replaces the empty-state line with a short explainer of what span
  events are, so users landing cold don't see a cryptic message.

Required a small backend fix: `fastaiagent/trace/storage.py` and
`fastaiagent/trace/platform_export.py` were dropping `event.attributes`
during serialization, so the well-known `exception.type` /
`exception.message` / `exception.stacktrace` keys never reached the
UI. Both paths now preserve the full attribute bag.

### Fixed — OTel `UNSET` status no longer miscounted as error

`/api/agents` and `/api/workflows` were treating `span.status = UNSET`
as a failure. OpenTelemetry convention is `UNSET` = normally-completing
(unless code explicitly marks it), `OK` = explicit success, `ERROR` =
failure. The SDK doesn't mark successful spans as `OK`, so every real
agent and workflow was showing 0% success rate in the UI. Fixed both
aggregators — only `ERROR` counts as a failure now.

### Added — `examples/39_workflows_demo.py`

Runnable demo that executes one of each workflow runner against
OpenAI — 3-node chain, 3-agent swarm with handoffs, supervisor + 2
workers. Produces 9 real workflow traces in `local.db` so a new user
can run the demo and immediately see the Workflows page populated.

## [0.9.1] - 2026-04-21

### Changed — Agent Replay side-by-side comparison

The Local UI's `AgentReplayPage` now renders **real per-step input and
output diffs** when you fork and rerun a trace. Previously the
step-by-step grid only compared span names; divergence in actual
content was invisible. The `ReplayDiffView` component was rewritten:

- Each step row is now collapsible. Rows whose `input` or `output`
  differ show a chevron.
- On expand, the row renders a split-view diff powered by
  `react-diff-viewer-continued` for `input` and `output` separately,
  with word-level diff method and a theme-aware palette.
- Diverged rows (at or after `comparison.diverged_at`) are highlighted
  with a left-border bar + subtle background tint, and a "diverged at
  step N" badge appears at the top-right of the grid.

Closes the feature gap between the SDK's `ForkedReplay.compare()` and
what the UI actually rendered. No SDK-side changes.

New Playwright screenshot
[`20-replay-comparison.png`](docs/ui/screenshots/20-replay-comparison.png)
captures the post-rerun view with an expanded diverged row so the docs
stay in sync with what the code does.

Wheel bundle ships the rebuilt frontend — `pip install 'fastaiagent[ui]'`
picks up the new view automatically.

## [0.9.0] - 2026-04-21

### Added — Local UI knowledge-base browser

New read-only surface in the Local UI for every `LocalKB` collection on
disk. Routes for listing collections, document / chunk introspection,
a **search playground** (one request, one response — no streaming), and
a **lineage** tab that scans `spans` for `retrieval.<kb>` spans to show
which agents and traces have been hitting the KB.

- **Sidebar → Knowledge Bases** (`/kb`) enumerates every subdirectory of
  `./.fastaiagent/kb/` (or `$FASTAIAGENT_KB_DIR`) containing a
  `kb.sqlite` file.
- **Collection detail** (`/kb/:name`) with three tabs:
  - *Documents* — grouped by `metadata.source`, with per-chunk inspector.
  - *Search playground* — calls the same `kb.search()` used at runtime.
  - *Lineage* — agents + recent traces derived from retrieval spans.
- Strictly read-only. Adds/deletes/re-indexing stay in code, consistent
  with the rest of the Local tier.

New REST endpoints (`/api/kb/...`): `GET /api/kb`,
`GET /api/kb/{name}`, `GET /api/kb/{name}/documents`,
`GET /api/kb/{name}/chunks`, `POST /api/kb/{name}/search`,
`GET /api/kb/{name}/lineage`.

Tests: 9 pytest cases against a real `LocalKB` (no mocks); 4 e2e-gate
cases; 2 Vitest cases; 4 new Playwright screenshots (`16-kb-list`,
`17-kb-documents`, `18-kb-search`, `19-kb-lineage`) embedded in docs.

CI: matrix expanded with `ui` and `ui,kb` extras so the KB browser
routes are exercised under the minimum required dependency combo.

Docs: [docs/ui/kb.md](docs/ui/kb.md), README updated, new example
[examples/37_kb_ui.py](examples/37_kb_ui.py).

## [0.8.1] - 2026-04-21

### Fixed

- README: Local UI screenshot in the README now uses an absolute
  ``raw.githubusercontent.com`` URL so it renders on the PyPI project
  page (relative paths work on GitHub but not on PyPI).

## [0.8.0] - 2026-04-20

### Added — KB retrieval tracing

``LocalKB.search()``, ``PlatformKB.search()``, and ``PlatformKB.asearch()``
now emit a ``retrieval.<kb_name>`` span with ``retrieval.backend``,
``retrieval.search_type``, ``retrieval.query`` (payload-gated),
``retrieval.top_k``, ``retrieval.result_count``, ``retrieval.latency_ms``,
and ``retrieval.doc_ids`` (payload-gated). The span nests as a child of
the ``tool.*`` span when the KB is wired in via ``kb.as_tool()``, so the
trace tree becomes ``agent → tool → retrieval`` without extra work.

### Changed — unified workflow tracing for Chain / Swarm / Supervisor

Multi-agent runs used to fragment into N orphan agent traces. They now
emit a single root span with ``fastaiagent.runner.type`` set to
``chain``, ``swarm``, or ``supervisor``, so a chain of 3 agents shows up
as **one** trace with a Gantt-style tree of agents and LLM calls
underneath.

- ``Chain.aexecute()`` wraps in ``chain.<name>`` + sets ``chain.name``,
  ``chain.node_count``, ``chain.node_ids``, ``chain.input``,
  ``chain.output``, ``chain.execution_id``.
- ``Swarm.arun()`` wraps in ``swarm.<name>`` + sets ``swarm.name``,
  ``swarm.entrypoint``, ``swarm.agent_count``, ``swarm.input``,
  ``swarm.output``, ``swarm.handoff_count``.
- ``Supervisor.arun()`` wraps in ``supervisor.<name>`` + sets
  ``supervisor.name``, ``supervisor.worker_count``,
  ``supervisor.input``, ``supervisor.output``.
- UI: new Workflow badge in the traces table + trace detail summary bar;
  new Runner filter pill (Agent / Chain / Swarm / Supervisor); new
  ``runner_type`` query param on ``/api/traces``.

### Changed — graduated from Alpha to Beta

PyPI classifier moved from `Development Status :: 3 - Alpha` to
`Development Status :: 4 - Beta`. Public API is stable enough for
production use behind the usual "still pre-1.0" caveat.

### Added — Local UI

A single-user, Platform-lookalike web UI that ships inside the wheel. Run
`pip install 'fastaiagent[ui]'` then `fastaiagent ui` — bcrypt-hashed local
auth, browser opens automatically, nothing leaves your machine.

- **`fastaiagent ui`** CLI (`start`, `reset-password`) — FastAPI + uvicorn,
  `127.0.0.1:7842` by default, `--no-auth` / `--no-open` / `--port` / `--db` /
  `--auth-file` / `--host` flags, interactive first-run credential prompt.
- **Pages**: Overview, Traces list + Trace detail (Gantt-style span tree +
  Input/Output/Attributes/Events inspector), Agent Replay (fork dialog,
  rerun, side-by-side comparison, "save as regression test"), Eval Runs
  (trend chart + per-case scorer chips), Prompts browser + editor (gated on
  local registry), Guardrail events, Agent directory + agent detail.
- **Tech**: React 19 + Vite + Tailwind v4 + shadcn/ui, design tokens
  vendored from the SaaS Platform for visual parity; TanStack Query with
  manual refetch (no live stream — simple REST refresh UX).

### Changed — unified local storage (breaking)

All local persistence now lives in a single SQLite file at
`./.fastaiagent/local.db` (was: `traces.db` + `checkpoints.db` + `.prompts/`
YAML files).

- `PromptRegistry` is now SQLite-backed (`YAMLStorage` → `SQLiteStorage`);
  public API unchanged.
- `TraceStore`, `CheckpointStore` write to `local.db`.
- `EvalResults.persist_local()` writes one `eval_runs` row + N `eval_cases`
  rows; `evaluate()` calls it automatically (opt out with `persist=False`).
- `Guardrail.aexecute()` writes to `guardrail_events` when
  `FASTAIAGENT_UI_ENABLED=true` (no-op otherwise).
- Legacy env vars (`FASTAIAGENT_TRACE_DB_PATH`, `FASTAIAGENT_CHECKPOINT_DB_PATH`,
  `FASTAIAGENT_PROMPT_DIR`) still work but emit `DeprecationWarning`.
- **`fastaiagent migrate`** copies legacy `traces.db` + `checkpoints.db` +
  `.prompts/` into `local.db`. Auto-invoked by `fastaiagent ui start` on
  first run if legacy files are detected.

### Added — config fields

- `SDKConfig.local_db_path` (default `.fastaiagent/local.db`) + env var
  `FASTAIAGENT_LOCAL_DB`.
- `SDKConfig.ui_enabled`, `ui_host`, `ui_port` + env vars
  `FASTAIAGENT_UI_ENABLED`, `FASTAIAGENT_UI_HOST`, `FASTAIAGENT_UI_PORT`.
- `PromptRegistry.is_local()` returns True iff the DB file lives inside
  the current working directory — used by the UI to gate prompt editing.

## [0.7.0] - 2026-04-18

### Added — Platform-hosted Knowledge Bases
- **`fa.PlatformKB(kb_id=...)`** — new thin client for KBs hosted on the FastAIAgent platform. Calls `POST /public/v1/knowledge-bases/{id}/search`; the platform runs the full retrieval pipeline (hybrid search, reranking, relevance gate — whatever the KB is configured for). Exposes `.search()`, `.asearch()`, and `.as_tool()` — drop-in compatible with `LocalKB` for `Agent(tools=[kb.as_tool()])`. Requires `fa.connect(api_key=...)` and an API key with the `kb:read` scope.
- New docs page [docs/knowledge-base/platform-kb.md](docs/knowledge-base/platform-kb.md).
- New example [examples/34_platform_kb.py](examples/34_platform_kb.py) — `PlatformKB` wired into an `Agent`, end-to-end live-verified against a local platform + real OpenAI.
- New integration suite `tests/integration/test_platform_kb.py` — 8 cases against a live platform + real retrieval (no mocks): direct sync search, async search, `top_k` bounds, `PlatformNotFoundError` on bad id, empty `kb_id` rejected, `.as_tool()` + `SearchResult` parity, metadata passthrough, Agent end-to-end.

## [0.6.1] - 2026-04-18

### Added — CLI polish
- **`fastaiagent version`** now lists installed optional extras: `fastaiagent 0.6.1 [openai, anthropic, kb, qdrant, chroma, mcp-server]`. Handy in bug reports.
- **`fastaiagent connect --api-key ...`** / **`fastaiagent disconnect`** — save / remove Platform credentials at `~/.fastaiagent/credentials.toml` (chmod 0600). Auth check runs before the key is persisted.
- **`fastaiagent auth status`** — show whether credentials are saved, which source is active (env vs file), and a masked key preview.
- **`fastaiagent auth env`** — print `export` lines for sourcing: `eval "$(fastaiagent auth env)"`.
- **`fastaiagent kb list [--path ROOT]`** — enumerate all persistent KBs under a root directory. Shows name, chunk count, and path in a Rich table.
- **`fastaiagent agent serve path/to/file.py:my_agent [--port 8000]`** — run any `Agent` or `Chain` as a FastAPI service that implements the uniform deployment contract (`GET /health`, `POST /run`, `POST /run/stream`). Saves copy-pasting the 80-line starter server. Live-verified against real OpenAI `gpt-4o-mini` — `Paris.` response in 1353ms with trace_id; SSE stream delivered TextDelta events for "Hello there, friend".
- **`fastaiagent replay fork <trace_id> [--step N] [--prompt ...] [--input ...] [--output rerun.json]`** — CLI surface for `Replay.load(id).fork_at(step).modify_prompt(...).modify_input(...).rerun()`. Writes the rerun result to JSON or prints it.

### Deployment recipes (previously 0.6.1 was staged as unreleased — now shipped with this bump)
- New [docs/deployment/](docs/deployment/index.md) section with recipes for **FastAPI + Uvicorn**, **Docker → Cloud Run / Fly / Render / Railway / ECS**, **Modal**, and **Replicate (Cog)**. Every recipe exposes the same uniform contract as `agent serve`.
- **Example**: [examples/33_deploy_fastapi.py](examples/33_deploy_fastapi.py) — runnable FastAPI server. Live-verified end-to-end.

### Tests
- `tests/test_cli.py` — +9 tests covering the new commands (16/16 pass). `agent serve` live-smoke-tested against real OpenAI.

### Notes
- No breaking changes. No library-behavior changes. `Agent`, `Chain`, `Swarm`, `ComposableMemory`, MCP server, and KB protocols are untouched. Existing CLI commands work exactly as before.

## [0.6.0] - 2026-04-18

### Added
- **`FastAIAgentMCPServer`** — expose any `Agent` or `Chain` as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server. Claude Desktop, Cursor, Continue, Zed, or any MCP client (including fastaiagent's own `MCPTool`) can now invoke your agents as tools.
- **`Agent.as_mcp_server(...)`** and **`Chain.as_mcp_server(...)`** factory methods — one-liner creation of the server wrapper. Lazy import of the upstream `mcp` SDK: `import fastaiagent` does not pull it in unless a user touches the MCP server path.
- **`Tool.to_mcp_schema()`** — helper that renders a fastaiagent `Tool` in the MCP tool-schema shape (`name` / `description` / `inputSchema`), alongside the existing `to_openai_format()`.
- **CLI subcommand**: `fastaiagent mcp serve path/to/agent_file.py:my_agent` — starts an MCP stdio server from the command line. Accepts file paths or dotted module paths; `--expose-tools` surfaces inner tools; `--name` overrides the primary tool name.
- **Optional extra**: `pip install 'fastaiagent[mcp-server]'` → `mcp>=1.2`.
- **Docs**: new [docs/tools/mcp-server.md](docs/tools/mcp-server.md) with Claude Desktop / Cursor / Continue / Zed registration snippets and a full composed example (agent + KB + memory). Cross-linked from the existing [docs/tools/mcp-tools.md](docs/tools/mcp-tools.md).
- **Example**: [examples/32_mcp_expose_agent.py](examples/32_mcp_expose_agent.py) — a research assistant with a real `research_lookup` tool, ready to run and register with Claude Desktop. Verified live: real OpenAI drove the agent through the MCP protocol, tool was invoked, response flowed back.
- **Tests**: `tests/test_mcp_server.py` — 10 tests, including a **full-protocol end-to-end test** (initialize → tools/list → tools/call → prompts/list → prompts/get) using the upstream `mcp` SDK's in-memory transport. No protocol mocking.

### Deferred
- `transport="sse"` and `transport="streamable-http"` — accepted as values but raise `NotImplementedError` on `run()`. Only `stdio` ships in 0.6.0; the remote transports are tracked as 0.6.x follow-ups.
- MCP resources (mapping `LocalKB` namespaces to MCP resources) — not yet implemented.
- Auth middleware for remote transports — will compose with `AgentMiddleware` when SSE/HTTP land.

## [0.5.0] - 2026-04-18

### Added
- **`Swarm`** — peer-to-peer multi-agent topology. Each agent can hand off control to allowed peers via auto-injected `handoff_to_<peer>` tools; no central coordinator LLM. Implements the same `run`/`arun`/`astream`/`stream` surface as `Agent`, so it drops into any `Chain` node.
- **`SwarmState`** — a plain dataclass: `shared` (free-form blackboard), `handoff_count`, `path`, `last_reason`.
- **`SwarmError`** — raised on structural violations (missing entrypoint, duplicate agent names, disallowed handoff, cycle-guard exhausted).
- **`HandoffEvent(from_agent, to_agent, reason)`** — new stream event emitted by `Swarm.astream` on every transition, tagged onto the stream before the target agent starts streaming.
- **Handoff allowlist** — `handoffs: dict[str, list[str]]`. Default is full mesh. Attempts to hand off outside the allowlist raise `SwarmError`.
- **Cycle guard** — `max_handoffs` kwarg (default 8). The guard message includes the full visited path for debugging.
- **Shared blackboard** — the auto-generated handoff tool accepts an optional `context=` dict; entries merge into `SwarmState.shared` and are exposed to the next agent via the briefing prompt.
- **Serialization** — `Swarm.to_dict()` / `Swarm.from_dict(data, agents=...)` — structural round-trip; the caller supplies live `Agent` instances.
- **Docs**: new [docs/agents/swarm.md](docs/agents/swarm.md) with full reference, Swarm-vs-Supervisor decision matrix, streaming, shared-memory, and KB-integration patterns. Cross-linked from [docs/agents/teams.md](docs/agents/teams.md) and [docs/agents/index.md](docs/agents/index.md).
- **Example**: [examples/31_swarm_research_team.py](examples/31_swarm_research_team.py) — triage → coder/writer swarm with real tools (pypi-lookup), constrained allowlist, and streaming with `HandoffEvent`.
- **Tests**:
  - `tests/test_swarm.py` — 17 deterministic + 1 live-LLM test covering construction validation, one-hop and multi-hop routing, allowlist, cycle guard, blackboard, serialization, streaming.
  - `tests/test_multi_agent_integration.py` — 5 live integration tests covering Supervisor+KB, Supervisor+ComposableMemory, Swarm+KB, Swarm+shared-memory-with-VectorBlock, Swarm writer-critic loop — all exercised against real OpenAI `gpt-4o-mini` and real FAISS.
- `MockLLMClient.astream` — deterministic stream-path fixture in `tests/conftest.py` so other tests can exercise streaming code paths without live APIs. (Unit test infra only; does not alter the `LLMClient` public behavior.)

### Changed
- `execute_tool_loop` now appends the in-flight tool-call record to `all_tool_calls` **before** returning from a `StopAgent` catch, so callers inspecting the completed-but-stopping run still see the final tool call. This was required for Swarm handoff detection; also benefits any middleware that raises `StopAgent` from `wrap_tool`.

### Deferred
- Streaming-path `AgentMiddleware` integration (Gap 3 follow-up). The `_ExitAfterHandoff` middleware that Swarm uses internally runs on the non-streaming path only; `Swarm.astream` detects handoffs directly from stream events and breaks out of the inner agent's stream as soon as one fires. This is functionally equivalent for swarm semantics; general streaming middleware ships in a later minor version.

## [0.4.0] - 2026-04-18

### Added
- **`ComposableMemory`** — a sliding-window `AgentMemory` augmented with a list of long-term memory blocks. Accepted by `Agent(memory=...)` as a drop-in replacement for `AgentMemory`.
- **`MemoryBlock` ABC** — minimal two-method interface (`on_message`, `render`) for writing your own memory block. Shipped blocks:
  - **`StaticBlock(text)`** — a fixed system-level fact injected every turn.
  - **`SummaryBlock(llm=..., keep_last=..., summarize_every=...)`** — rolling LLM-generated summary of older turns.
  - **`VectorBlock(store=..., top_k=...)`** — semantic recall over past messages via any `VectorStore` backend (FAISS, Qdrant, Chroma, ...). Built on the 0.3.0 KB protocols.
  - **`FactExtractionBlock(llm=..., max_facts=...)`** — durable-fact extraction via a cheap LLM, deduped and capped.
- **`Agent._build_messages`** now passes the current user input as `query` to `memory.get_context(query=...)`. `AgentMemory` ignores the extra argument; `ComposableMemory` uses it for query-conditioned blocks like `VectorBlock`.
- **Persistence** — `ComposableMemory.save(path)` / `load(path)` round-trips the primary window and each block to a directory (`primary.json` + `blocks/{name}.json`).
- **Docs**: rewrote [docs/agents/memory.md](docs/agents/memory.md) with a composable-memory section, the four shipped blocks, a `MoodBlock` custom-block worked example, and ordering guidance. Cross-linked from [docs/agents/index.md](docs/agents/index.md) and [docs/knowledge-base/backends.md](docs/knowledge-base/backends.md).
- **Example**: [examples/30_memory_blocks.py](examples/30_memory_blocks.py) — a 6-turn conversation exercising all four blocks end-to-end; verified live against OpenAI `gpt-4o-mini`.
- **Tests**: `tests/test_memory_blocks.py` — 17 deterministic + 2 live-LLM tests covering block semantics, ordering, composition, backward compat, block-failure isolation, namespace isolation, save/load round-trip, and real LLM behavior for `SummaryBlock` and `FactExtractionBlock`.

### Changed
- `AgentMemory.get_context` grew an optional `query: str = ""` argument (ignored by `AgentMemory`; used by `ComposableMemory`). Backward compatible — existing calls with no args still work.
- `Agent.memory` accepts `AgentMemory | ComposableMemory | None`.

### Deferred
- Async parallel methods (`aon_message`, `arender`) on `MemoryBlock` — additive, planned as a follow-up. Same rationale as the KB protocols' async deferral (see `fastaiagent/kb/protocols.py` module docstring).

## [0.3.0] - 2026-04-18

### Added
- **Pluggable KB storage backends** — `LocalKB` now accepts `vector_store`, `keyword_store`, and `metadata_store` kwargs. Default behavior (FAISS + BM25 + SQLite) is byte-for-byte identical to 0.2.x.
- **Three storage protocols** exposed at the top level: `VectorStore`, `KeywordStore`, `MetadataStore` (structural `typing.Protocol` — no base class required).
- **`fastaiagent.kb.backends`** submodule with shipping adapters:
  - `FaissVectorStore` — wraps the existing FAISS index (default vector backend)
  - `BM25KeywordStore` — wraps the existing pure-Python BM25 index (default keyword backend)
  - `SqliteMetadataStore` — wraps the existing SQLite document+chunk store (default metadata backend)
  - `QdrantVectorStore` — Qdrant adapter. Install `fastaiagent[qdrant]`. Supports remote HTTP, Qdrant Cloud (`api_key=`), and in-process `location=":memory:"`.
  - `ChromaVectorStore` — Chroma adapter. Install `fastaiagent[chroma]`. Supports ephemeral, persistent (`persist_path=`), and remote (`host=`) modes.
- Optional dependencies: `fastaiagent[qdrant]`, `fastaiagent[chroma]`.
- Docs: new [docs/knowledge-base/backends.md](docs/knowledge-base/backends.md) (backend reference) and [docs/knowledge-base/custom-backend.md](docs/knowledge-base/custom-backend.md) (write-your-own guide). Cross-link added from `docs/knowledge-base/index.md`.
- Examples: [examples/28_kb_chroma.py](examples/28_kb_chroma.py), [examples/29_kb_qdrant.py](examples/29_kb_qdrant.py) — both runnable with only the respective extra installed.
- Tests: `tests/test_kb_protocols.py` (contract suite parametrized over backends), `tests/test_kb_backend_defaults.py` (backward-compat), `tests/test_kb_chroma.py` + `tests/test_kb_qdrant.py` (live, gated by pytest markers).

### Changed
- `LocalKB` internals renamed for clarity: `_faiss_index` → `_vector`, `_bm25_index` → `_keyword`, `_db` metadata path → `_metadata`. `_embeddings` list removed; embeddings now live inside their backend and `MetadataStore`. Public API unchanged.
- `SqliteMetadataStore` keeps the existing `chunks` table name so previously-persisted KBs load without migration.

### Deferred
- Async (`aadd`, `asearch`, `aembed`, ...) parallel methods on the protocols and backends — planned as an additive change, explicitly not in 0.3.0. See the `fastaiagent/kb/protocols.py` module docstring for the roadmap.

## [0.2.0] - 2026-04-18

### Added
- **`AgentMiddleware`** — composable pre/post model hooks and tool wrappers for transforming messages, responses, and tool calls without subclassing `Agent`. Three hooks:
  - `before_model(ctx, messages)` — transform messages before the LLM call. Runs in declaration order.
  - `after_model(ctx, response)` — inspect or rewrite the LLM response. Runs in reverse declaration order.
  - `wrap_tool(ctx, tool, args, call_next)` — onion-wrap each tool invocation. First middleware is outermost.
- **`MiddlewareContext`** — per-run context passed to every hook. Exposes `turn`, `tool_call_index`, mutable `scratch` dict, and `agent_name`.
- **Built-in middleware**:
  - `TrimLongMessages(keep_last=20)` — cap message-history size while preserving the leading system prompt.
  - `ToolBudget(max_calls=10, message=...)` — cooperatively stop the run after N tool invocations.
  - `RedactPII(patterns=..., placeholder="[REDACTED]")` — redact email, phone, SSN, and credit-card patterns from outbound messages and inbound responses.
- **`StopAgent`** exception — raise from any middleware hook to end a run cooperatively; the `AgentResult.output` carries the message.
- **`Agent.__init__`** now accepts `middleware: Sequence[AgentMiddleware] | None = None`. When `None` (the default), behavior is byte-for-byte identical to 0.1.8.
- Docs: new page [docs/agents/middleware.md](docs/agents/middleware.md) covering the hook reference, ordering semantics, built-ins, custom middleware patterns, and interaction with guardrails. Cross-links added from `docs/agents/index.md` and `docs/guardrails/index.md`.
- Example: [examples/27_middleware_tool_budget.py](examples/27_middleware_tool_budget.py) — demonstrates `ToolBudget` + `TrimLongMessages` + `RedactPII`. Includes an offline demo using `MockLLMClient` that runs without API keys.

### Changed
- `execute_tool_loop` (internal) gained optional `mw_pipeline` and `mw_ctx` parameters. When unset, the hot path is unchanged.

## [0.1.8] - 2026-04-12

### Fixed
- **`Replay.from_platform(trace_id)` was completely broken** — crashed with a Pydantic `ValidationError` on every call because the platform API returns a different span schema (`id` not `span_id`, attributes split into `input`/`output` dicts instead of a single `attributes` dict, no `trace_id` on individual spans). Fixed by mapping the platform schema to the SDK's internal `SpanData` model. Verified end-to-end: agent.run → push to platform → Replay.from_platform → fork → rerun → compare, all working.
- **Broken example links in tracing docs** — three links in `docs/tracing/index.md` pointed to `github.com/anthropics/fastaiagent-sdk` (does not exist). Corrected to `github.com/fastaifoundry/fastaiagent-sdk`.

### Added
- **`.github/CODEOWNERS`** — designates `@fastaifoundry` as the sole code owner. Paired with "Require review from Code Owners" ruleset on `main`, only the owner's approval can unblock PRs.
- **Expanded e2e quality gate coverage** — 20 gate files covering 81+ assertions across every user-facing surface (Anthropic, Azure, Ollama providers; streaming; structured output; chains + resume; supervisor/worker; RESTTool; LocalKB; prompt registry; error paths; LangChain; CrewAI; MCPTool; HITL; OTLP export to Jaeger).
- **`docs/internals/tracing-architecture.md`** — contributor-facing deep dive into the span lifecycle (creation → SQLite → platform → OTLP → replay), with attribute tables, span tree diagrams, common mistakes section.
- **`docs/internals/platform-api.md`** — contributor-facing deep dive into platform communication (connection lifecycle, PlatformAPI HTTP client, all 8 endpoints, prompt registry caching, graceful degradation patterns, authentication & scopes).
- **`docs/internals/evaluation-system.md`** — contributor-facing deep dive into the eval framework (evaluate loop, dataset/scorer resolution, all 18 built-in scorers, LLM judge pattern, EvalResults).
- **Doc fixes** — HITL rejection behavior documented (does NOT halt chain); chain tool-node `state.output` wrapping quirk documented; `RunContext` + `from __future__ import annotations` footgun documented; `examples/04_agent_replay.py` rewritten to use real agent run instead of a fake trace.

## [0.1.7] - 2026-04-11

### Added
- **End-to-end quality gate** — A pytest suite under `tests/e2e/test_quality_gate.py` that runs the full 16-step product lifecycle with real assertions: install → connect → create agent → add tool → add guardrail → run → inspect → trace_id → verify on platform → run eval → check scores → load replay → fork at step 2 → rerun → compare. Wired into `.github/workflows/ci.yml` as a required status check.
  - `E2E_REQUIRED=1` flips the gate from local-skip to hard-fail on missing env — CI sets this, local developers get clean skips.
  - `E2E_SKIP_PLATFORM=1` bypasses the `connect` and `verify-trace-on-platform` steps while still exercising agent, eval, and replay. Used on CI when hitting a remote platform every commit is not desired; locally, leave unset and point `FASTAIAGENT_TARGET` at your dev platform to exercise the full flow.
- **`ToolRegistry`** — A process-wide, name-keyed registry that holds live tool callables so Agent Replay can rebind them by name after reconstruction from a trace. `FunctionTool(name=..., fn=...)` and the `@tool` decorator auto-register on construction, so most code needs no changes. Exported as `fastaiagent.ToolRegistry`. See [docs/tools/index.md#toolregistry](docs/tools/index.md).
- **Real `ForkedReplay.arerun()`** — Replaces the previous stub. Reconstructs an `Agent` from the root agent span's attributes (`agent.config`, `agent.tools`, `agent.guardrails`, `agent.llm.config`, `agent.system_prompt`, `agent.input`), applies `modify_prompt`/`modify_config`/`modify_input`, rebinds tools via the `ToolRegistry`, and re-executes via `agent.arun`. `ComparisonResult.new_steps` now contains the spans from the rerun trace. v1 re-runs from the top with modifications applied; mid-trace resume (replaying messages up to `fork_point`) is planned as a follow-up.
- **Enriched span instrumentation** — Agent, tool, and LLM spans now carry the metadata needed for replay reconstruction and richer observability:
  - Agent root span: `agent.config`, `agent.tools`, `agent.guardrails`, `agent.llm.provider`, `agent.llm.model`, `agent.llm.config`, `agent.system_prompt` (payload-gated).
  - LLM span (`llm.{provider}.{model}`, wrapped around `LLMClient.acomplete`): `gen_ai.request.messages`, `gen_ai.request.tools`, `gen_ai.response.content`, `gen_ai.response.tool_calls`, `gen_ai.response.finish_reason`, plus existing model/token/temperature attributes. Emitted for every provider regardless of whether users call the bare provider SDK.
  - Tool span (`tool.{name}`, new): `tool.name`, `tool.status` (`ok`/`error`/`unknown`), `tool.args`, `tool.result`, `tool.error` (payload-gated).
- **`FASTAIAGENT_TRACE_PAYLOADS` env var** — Set to `0` to skip capturing payload-bearing span attributes (messages, responses, resolved prompts, tool args/results). Defaults to on; structural metadata (tool/guardrail/LLM schemas, token counts, finish reasons) is always captured.

### Fixed
- **LLM calls were producing no spans on the agent flow.** `LLMClient` hits provider HTTP APIs with `httpx.AsyncClient` and never imports the `openai`/`anthropic` Python SDKs, so the monkey-patches in `fastaiagent/integrations/openai.py` and `anthropic.py` were dead code for real agent execution — every `agent.run()` produced a trace with only the root agent span (and, after this release, the tool span), no LLM call span. Fixed by wrapping `LLMClient.acomplete` in an OTel span at the dispatch level so every provider produces consistent `llm.{provider}.{model}` spans. The integration-level patches still fire for users calling bare provider SDKs directly. Discovered by the new e2e quality gate on its first real run.
- **`fa.connect("localhost:8001")` threw an opaque `httpx` error on missing URL scheme.** Added `_normalize_target()` to `fastaiagent.client` which prepends `http://` for localhost and private hosts, `https://` otherwise. Users can now pass `localhost:8001`, `http://localhost:8001`, or `https://app.fastaiagent.net` interchangeably.

### Tests
- 566 unit tests + 16 e2e gate steps, all green locally against real OpenAI and a local docker-compose platform.
- `tests/test_replay.py` — added `_make_agent_trace()` helper and `stub_agent_arun` fixture; rewrote `test_rerun`/`test_compare` for the real `arerun`; added `test_rerun_applies_prompt_modification` and `test_rerun_raises_when_trace_has_no_spans`.

## [0.1.6] - 2026-04-06

### Fixed
- **Lazy imports for numpy/faiss** — `import fastaiagent` no longer fails when `faiss-cpu`/`numpy` are not installed. FAISS and numpy are now imported inside `FaissIndex` methods, not at module level.

### Tests
- **Prompt Registry edge case tests** — Added 15 tests covering auto-increment versioning, load-latest behavior, forced version gaps, non-existent version/alias errors, unresolved fragments, multiple fragment resolution, fragment overwrite, empty list, diff with no changes, missing variable formatting, and latest version discovery via `load()` and `list()`.

## [0.1.5] - 2026-04-06

### Added
- **LocalKB persistence** — Chunks and embeddings now auto-save to SQLite and auto-load on restart. No re-embedding on process restart. Use `persist=False` for in-memory-only throwaway KBs.
- **FAISS vector search** — Replaced pure-Python cosine similarity with FAISS (`IndexFlatIP`). Configurable index types: `"flat"` (default, exact), `"ivf"` (approximate, 100K-1M), `"hnsw"` (graph-based, fast recall). `faiss-cpu` added to `[kb]` extra.
- **BM25 keyword search** (`fastaiagent/kb/bm25.py`) — Lightweight in-memory BM25 index with no external dependencies. Catches exact terms, error codes, and IDs that vector search misses.
- **Hybrid search** (default) — Combined FAISS + BM25 with configurable `alpha` weighting. `search_type` parameter: `"hybrid"` (default), `"vector"`, `"keyword"`.
- **CRUD operations** on LocalKB — `delete(chunk_id)`, `delete_by_source(source)`, `update(chunk_id, content)`, `clear()`. All persist to SQLite and rebuild active indexes.
- **Directory ingestion** — `kb.add("docs/")` recursively ingests all `.txt`, `.md`, `.pdf` files.
- **Chunk UUID** — `Chunk` model now has an `id` field (auto-generated UUID) for update/delete operations.
- **Conditional index creation** — `search_type="keyword"` skips embedding entirely (no embedder needed, no FAISS, zero embedding cost).
- CLI commands: `fastaiagent kb clear`, `fastaiagent kb delete`.
- Exports: `Chunk` and `SearchResult` now exported from `fastaiagent.kb`.
- Integration tests with real FastEmbed embeddings (`tests/test_kb_integration.py`).
- Comprehensive docs: search types, index types, alpha tuning, CRUD, persistence, multi-KB pattern.

### Changed
- `LocalKB.__init__` — New parameters: `persist`, `search_type`, `index_type`, `alpha`. All backward compatible with defaults.
- `kb.status()` — Now includes `persist`, `search_type`, `index_type` fields.
- Customer-support-agent example updated to use persistence (no more `_ensure_kb()` global flag pattern).

## [0.1.4] - 2026-04-05

### Added
- **RAG evaluation metrics** (`fastaiagent/eval/rag.py`) — `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall` scorers for evaluating retrieval-augmented generation pipelines. All use LLM-as-judge with claim extraction and verification.
- **Safety evaluation metrics** (`fastaiagent/eval/safety.py`) — `Toxicity` and `Bias` (LLM-based), `PIILeakage` (regex-based email, phone, SSN, credit card detection) for content safety evaluation.
- **Similarity & NLP metrics** (`fastaiagent/eval/similarity.py`) — `SemanticSimilarity` (embedding cosine similarity), `BLEUScore` (n-gram precision), `ROUGEScore` (rouge-1 and rouge-l), `LevenshteinDistance` (normalized edit distance). BLEU/ROUGE/Levenshtein are pure Python with zero API cost.
- **`ToolCallCorrectness`** scorer in `fastaiagent/eval/trajectory.py` — validates tool names AND arguments with deep equality matching, stricter than `ToolUsageAccuracy`.
- All 11 new scorers registered in `BUILTIN_SCORERS` for string-based resolution in `evaluate()`. Total built-in scorers: 7 → 18.
- Documentation: `docs/evaluation/rag-metrics.md`, `docs/evaluation/safety-metrics.md`, `docs/evaluation/similarity-metrics.md`.
- Examples: `24_rag_eval.py`, `25_safety_eval.py`, `26_similarity_eval.py`.
- End-to-end evaluation tests with real LLM API calls (`tests/test_eval_e2e.py`).

### Fixed
- `ConversationCoherence` session scorer now detects self-contradictions and topic drift (previously a stub returning 1.0).
- `GoalCompletion` session scorer now uses keyword recall with stop-word filtering, bigram matching, and checklist detection (previously naive word overlap).

## [0.1.3] - 2026-04-05

### Added
- **Customer Support Agent template** (`examples/customer-support-agent/`) — Production-ready example demonstrating Agent with tools, RunContext dependency injection, knowledge base (LocalKB), guardrails (PII filter, toxicity check), evaluation suite (LLM-as-Judge), Agent Replay, and `fa.connect()` platform integration.

### Fixed
- **`examples/19_connect_e2e.py`** — Use timestamped prompt slugs to avoid collisions across runs; wrap dataset/eval publishing in try/except for scoped API keys.

## [0.1.2] - 2026-04-05

### Added
- **README Quickstart section** — Shows LLMClient and Agent creation front-and-center so new users see ease of use immediately.
- **`result.trace_id` shown everywhere** — README quickstart, `examples/01_simple_agent.py`, getting-started guide, tracing docs, and replay docs now all show how every run returns a `trace_id` for replay/debugging.

### Fixed
- **`docs/getting-started/first-agent.md`** — Fixed incorrect `result.trace.summary()` → proper `result.trace_id` + `Replay.load()` workflow.
- **`docs/replay/index.md`** — Added "Where Do Trace IDs Come From?" section explaining that every `agent.run()` returns a `trace_id`.
- **`docs/tracing/index.md`** — Quickstart now shows `result.trace_id` and how to use it with `Replay.load()`.

## [0.1.0] - 2026-04-04

### Added
- **`fa.connect()`** — Connect the SDK to FastAIAgent Platform for observability, prompt management, and evaluation services. All SDK features work without connect(). This adds platform backends alongside local storage.
- **`fa.disconnect()`** — Revert to local-only mode.
- **`fa.is_connected`** — Check connection status.
- **Platform trace export** — Traces automatically sent to platform via OTel `BatchSpanProcessor` when connected. Local SQLite always available as fallback.
- **`TraceData.publish()`** — Manual backfill of local traces to the platform.
- **`Replay.from_platform(trace_id)`** — Pull any trace from the platform and replay locally.
- **`PromptRegistry.get(slug, version, source)`** — Pull prompts from platform with TTL caching (`source="auto"`, `"platform"`, `"local"`).
- **`PromptRegistry.publish(slug, content, variables)`** — Publish prompts to the platform registry.
- **`PromptRegistry.refresh(slug)`** — Invalidate platform prompt cache.
- **`Dataset.from_platform(name)`** — Pull eval datasets from the platform.
- **`Dataset.publish(name)`** — Push datasets to the platform.
- **`EvalResults.publish(run_name)`** — Publish eval results to the platform.
- **`Scorer.from_platform(name)`** — Pull scorer configs (LLM judge) from the platform.
- **`PlatformNotConnectedError`** — Clear error when platform methods are called without `fa.connect()`.
- `PlatformAPI.get()` / `PlatformAPI.aget()` — GET request support for platform API client.
- `get_platform_api()` helper — Creates `PlatformAPI` from current connection state.

### Removed
- **`fa.init()`** — Replaced by `fa.connect()`.
- **`fa.push()` / `fa.push_all()`** — No agent definition sync. Agents are code, not config to push.
- **`FastAI` class** — Replaced by module-level `connect()`/`disconnect()`.
- **`PushResult`** — No longer needed.
- **`OfflineCache`** (`_platform/cache.py`) — Push buffer no longer needed.
- **`deploy/push.py`** — Push deployment logic removed.
- **CLI `push` command** — Removed.

### Changed
- `client.py` rewritten: `FastAI` class → `_Connection` singleton with `connect()`/`disconnect()`.
- `_platform/api.py` refactored: removed push-only docstrings, added GET methods.
- `__init__.py` updated: exports `connect`, `disconnect`, `is_connected` instead of `FastAI`, `init`.

## [0.1.0a7] - 2026-04-03

### Added
- **Supervisor/Worker context passthrough**: `RunContext` now flows from `Supervisor` through to all worker agents and their tools. Worker tools declaring `RunContext` parameters receive the same context the supervisor was called with.
- **Supervisor streaming**: `Supervisor.astream()` (async generator) and `Supervisor.stream()` (sync collector) for real-time token streaming during team delegation.
- **Supervisor dynamic instructions**: `Supervisor.system_prompt` now accepts `str | Callable[..., str]`, matching `Agent` behavior. Callable prompts receive the `RunContext` at execution time.
- **Top-level exports**: `Supervisor` and `Worker` now importable via `from fastaiagent import Supervisor, Worker`.
- `Worker.description` safely handles callable `system_prompt` (previously crashed with `TypeError`).
- Documentation: updated `docs/agents/teams.md` with context, streaming, dynamic instructions sections and API reference.
- Example: `18_supervisor_worker.py` demonstrating delegation, context passthrough, dynamic instructions, and streaming.
- 17 unit tests in `tests/test_team.py` covering all new supervisor/worker features.

### Fixed
- `Worker.__init__` no longer crashes when the wrapped agent has a callable `system_prompt`.
- `Supervisor._build_worker_tools()` now uses stateless per-call tool rebuilding (concurrent-safe).

## [0.1.0a6] - 2026-04-03

### Added
- Dynamic Instructions: `Agent.system_prompt` now accepts `str | Callable[..., str]`. Callable prompts are invoked with `RunContext` (or `None`) at the start of each `arun()`/`astream()` call, enabling per-request system prompt personalization.
- `Agent.to_dict()` raises `ValueError` when `system_prompt` is callable (callables cannot be serialized to the platform).
- Documentation: `docs/agents/dynamic-instructions.md`.
- Examples: `16_dynamic_instructions.py` (basic), `17_dynamic_instructions_advanced.py` (named functions, feature flags, streaming).
- 15 unit tests for dynamic instructions in `tests/test_dynamic_instructions.py`.

## [0.1.0a5] - 2026-04-03

### Added
- `RunContext[T]` — typed dependency injection for tools. Pass runtime dependencies (DB connections, API clients, user sessions) to tools cleanly without closures or globals.
- `_is_context_param()` helper and `FunctionTool._detect_context_param()` for automatic context parameter detection at tool init time.
- Context parameters are excluded from LLM-facing JSON schemas — the LLM never sees them.
- `context` keyword argument on `Agent.run()`, `Agent.arun()`, `Agent.astream()`, and `Agent.stream()`.
- `RunContext` exported from top-level: `from fastaiagent import RunContext`.
- Documentation: `docs/tools/context.md` with full usage guide.
- Examples: `14_context_di.py` (OpenAI), `15_context_di_anthropic.py` (Anthropic).
- 22 unit tests and 14 integration tests (OpenAI + Anthropic) for context injection.

### Changed
- `Tool.execute()` / `Tool.aexecute()` signatures now accept optional `context` parameter.
- `RESTTool.aexecute()` and `MCPTool.aexecute()` accept `context` for signature compatibility (ignored).
- `execute_tool_loop()` and `stream_tool_loop()` pass context through to tools.

## [0.1.0a4] - 2026-03-28

### Added
- Streaming support: `Agent.astream()` and `Agent.stream()` with `StreamEvent`, `TextDelta`.
- Structured output: `response_format` support across all LLM providers.

## [0.1.0a1] - Unreleased

### Added
- Initial SDK scaffold with package structure
- Agent class with tool-calling loop, sync/async/stream interfaces
- Chain class with cyclic graph execution, typed state, checkpointing, resume
- Tool system: FunctionTool, RESTTool, MCPTool with schema drift detection
- LLMClient abstraction with OpenAI, Anthropic, Ollama, Azure, Bedrock, Custom providers
- Guardrail system with 5 implementation types and built-in factories
- OTel-native tracing with local SQLite storage
- Agent Replay with fork-and-rerun debugging
- Prompt registry with fragment composition and versioning
- Local knowledge base with file ingestion and cosine similarity search
- Evaluation framework with multi-turn and trajectory scoring
- Auto-tracing integrations for OpenAI, Anthropic, LangChain, CrewAI
- CLI with commands for replay, eval, traces, prompts, kb
- Canonical format fixtures for SDK-platform contract testing
