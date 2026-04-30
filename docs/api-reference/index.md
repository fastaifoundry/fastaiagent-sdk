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

## Durability (v1.0)

- **[interrupt / Resume](../durability/api-reference.md)** — `interrupt(reason, context)`, `Resume(approved, metadata)`, `InterruptSignal`, `AlreadyResumed`
- **[@idempotent](../chains/idempotency.md)** — `idempotent`, `IdempotencyError`
- **[Checkpointer Protocol](../durability/checkpointers.md)** — `Checkpointer`, `SQLiteCheckpointer`, `PostgresCheckpointer`, `PendingInterrupt`, `Checkpoint`
- **Resume entrypoints** — `Chain.aresume(execution_id, *, resume_value=Resume(...))`, `Agent.aresume(...)`, `Swarm.aresume(...)`, `Supervisor.aresume(...)`. All four runner types share the same atomic-claim contract.

See the full [durability API reference](../durability/api-reference.md) for exact method signatures, the `Checkpointer` Protocol surface, and the `agent_path` hierarchy used by multi-agent topologies.

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
