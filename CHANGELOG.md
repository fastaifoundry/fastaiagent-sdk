# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
