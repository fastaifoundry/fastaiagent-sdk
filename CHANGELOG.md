# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
