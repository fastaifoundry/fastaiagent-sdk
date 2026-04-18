# API Reference

Auto-generated reference documentation for all public FastAIAgent SDK classes and functions.

## Core

- **[Agent](../agents/index.md)** — `Agent`, `AgentConfig`, `AgentResult`
- **[Chain](../chains/index.md)** — `Chain`, `ChainResult`, `ChainState`
- **[LLMClient](../getting-started/index.md)** — `LLMClient`, `LLMResponse`, `Message`

## Tools

- **[Tool](../tools/index.md)** — `Tool`, `ToolResult`, `FunctionTool`, `RESTTool`, `MCPTool`

## Safety

- **[Guardrail](../guardrails/index.md)** — `Guardrail`, `GuardrailResult`, `no_pii`, `json_valid`, `toxicity_check`

## Observability

- **[TraceStore](../tracing/index.md)** — `TraceStore`, `TraceData`, `SpanData`, `trace_context`
- **[Replay](../replay/index.md)** — `Replay`, `ReplayStep`, `ReplayResult`

## Intelligence

- **[PromptRegistry](../prompts/index.md)** — `PromptRegistry`, `Prompt`, `Fragment`
- **[LocalKB](../knowledge-base/index.md)** — `LocalKB`, `Document`, `SearchResult`
- **[PlatformKB](../knowledge-base/platform-kb.md)** — `PlatformKB` (hosted; thin client over `/public/v1/knowledge-bases/{id}/search`)

## Evaluation

- **[evaluate](../evaluation/index.md)** — `evaluate`, `Dataset`, `Scorer`, `EvalResults`

## Platform

- **[Connection](../platform/index.md)** — `fa.connect()`, `fa.disconnect()`, `fa.is_connected`

---

For detailed usage, see the individual module documentation pages linked above.
