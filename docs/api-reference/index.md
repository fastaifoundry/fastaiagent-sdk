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
- **[Guardrail primitives](../integrations/primitives-without-the-runtime.md)** — `run_guardrail(guardrail, data)` (compute only, applies `on_error`), `plane_guardrails_for_agent(agent_id)`, `guardrail_from_policy_rule(rule)`. Use these when the runtime isn't `fa.Agent`.

## Durability (v1.0)

- **[interrupt / Resume](../durability/api-reference.md)** — `interrupt(reason, context)`, `Resume(approved, metadata)`, `InterruptSignal`, `AlreadyResumed`
- **[@idempotent](../chains/idempotency.md)** — `idempotent`, `IdempotencyError`
- **[Checkpointer Protocol](../durability/checkpointers.md)** — `Checkpointer`, `SQLiteCheckpointer`, `PostgresCheckpointer`, `PendingInterrupt`, `Checkpoint`
- **Resume entrypoints** — `Chain.aresume(execution_id, *, resume_value=Resume(...))`, `Agent.aresume(...)`, `Swarm.aresume(...)`, `Supervisor.aresume(...)`. All four runner types share the same atomic-claim contract.

See the full [durability API reference](../durability/api-reference.md) for exact method signatures, the `Checkpointer` Protocol surface, and the `agent_path` hierarchy used by multi-agent topologies.

## Observability

- **[TraceStore](../tracing/index.md)** — `TraceStore`, `TraceData`, `SpanData`, `trace_context`
- **[Replay](../replay/index.md)** — `Replay`, `ReplayStep`, `ReplayResult`
- **[OpenInference emitters](../integrations/primitives-without-the-runtime.md#the-wire-contract)** — `emit_guardrail(tracer, ...)` / `emit_evaluation(tracer, ...)` open+stamp+close a child span on *your* tracer; `set_guardrail_attributes(span, ...)` / `set_evaluation_attributes(span, ...)` stamp a span you already own. `evaluation.score` is a 0..1 scale.

## Intelligence

- **[PromptRegistry](../prompts/index.md)** — `PromptRegistry`, `Prompt`, `Fragment`
- **[LocalKB](../knowledge-base/index.md)** — `LocalKB`, `Document`, `SearchResult`
- **[PlatformKB](../knowledge-base/platform-kb.md)** — `PlatformKB` (hosted; thin client over `/public/v1/knowledge-bases/{id}/search`)

## Evaluation

- **[evaluate](../evaluation/index.md)** — `evaluate`, `Dataset`, `Scorer`, `EvalResults`

## Platform

- **[Connection](../platform/index.md)** — `fa.connect()`, `fa.disconnect()`, `fa.is_connected`. `connect(export_traces=False)` connects for policy/scorers/prompts *without* registering a span exporter or claiming OTel's global provider — for runtimes that already own one.

---

For detailed usage, see the individual module documentation pages linked above.
