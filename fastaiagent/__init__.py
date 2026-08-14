"""FastAIAgent SDK — Build, debug, evaluate, and operate AI agents."""

from fastaiagent._internal.errors import StopAgent
from fastaiagent._version import __version__
from fastaiagent.agent import (
    Agent,
    AgentConfig,
    AgentMemory,
    AgentMiddleware,
    AgentResult,
    ComposableMemory,
    FactExtractionBlock,
    FewShotBlock,
    Memory,
    MemoryBlock,
    MiddlewareContext,
    PersistentFactBlock,
    RedactPII,
    Reflect,
    RunContext,
    StaticBlock,
    SummaryBlock,
    Supervisor,
    Swarm,
    SwarmError,
    SwarmState,
    ToolBudget,
    TrimLongMessages,
    VectorBlock,
    Worker,
)
from fastaiagent.chain import Chain, ChainResult, ChainState
from fastaiagent.chain.idempotent import IdempotencyError, idempotent
from fastaiagent.chain.interrupt import (
    AlreadyResumed,
    InterruptSignal,
    Resume,
    interrupt,
)
from fastaiagent.chain.node import node
from fastaiagent.checkpointers import Checkpointer, PendingInterrupt, SQLiteCheckpointer
from fastaiagent.client import connect, disconnect, push, refresh_policy
from fastaiagent.eval import (
    Dataset,
    EvalResults,
    HardeningReport,
    Scenario,
    Scorecard,
    Scorer,
    SimulatedUser,
    SimulationResults,
    agenerate_scenarios,
    aharden,
    asimulate,
    evaluate,
    generate_scenarios,
    harden,
    simulate,
)
from fastaiagent.guardrail import (
    Guardrail,
    GuardrailResult,
    allowed_topics,
    banned_topics,
    grounded,
    guardrail_from_policy_rule,
    json_valid,
    no_hallucination,
    no_pii,
    no_prompt_injection,
    no_secrets,
    openai_moderation,
    plane_guardrails_for_agent,
    responsible_ai,
    run_guardrail,
    toxicity_check,
)
from fastaiagent.kb import KeywordStore, LocalKB, MetadataStore, PlatformKB, VectorStore
from fastaiagent.llm import LLMClient, Message, StreamEvent, TextDelta
from fastaiagent.llm.stream import HandoffEvent
from fastaiagent.multimodal import PDF, ContentPart, File, Image, normalize_input
from fastaiagent.optimize import (
    OptimizationReport,
    OptimizeConfig,
    aoptimize,
    optimize,
)
from fastaiagent.prompt import Prompt, PromptRegistry
from fastaiagent.runtime import job_scope
from fastaiagent.tool import FunctionTool, MCPTool, RESTTool, Tool, ToolRegistry, tool
from fastaiagent.trace import (
    RedactionPolicy,
    TraceStore,
    disable_otel_capture,
    emit_evaluation,
    emit_guardrail,
    enable_otel_capture,
    get_redaction_policy,
    set_evaluation_attributes,
    set_guardrail_attributes,
    set_redaction_policy,
    trace_context,
)
from fastaiagent.trace.replay import Replay


def __getattr__(name: str) -> object:
    if name == "is_connected":
        from fastaiagent.client import _connection

        return _connection.is_connected
    if name == "FastAIAgentMCPServer":
        # Lazy: avoid pulling in the optional ``mcp`` package at import time.
        # Requires ``pip install 'fastaiagent[mcp-server]'``.
        from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

        return FastAIAgentMCPServer
    if name == "integrations":
        # The README's "Trace any agent" snippet calls
        # ``fastaiagent.integrations.langchain.enable()``. Lazy-import the
        # subpackage so optional integration deps don't load eagerly.
        import fastaiagent.integrations as integrations  # noqa: PLC0415

        return integrations
    raise AttributeError(f"module 'fastaiagent' has no attribute {name!r}")


__all__ = [
    "__version__",
    "connect",
    "disconnect",
    "refresh_policy",
    "push",
    "job_scope",
    "is_connected",
    # Agent
    "Agent",
    "AgentConfig",
    "AgentResult",
    "RunContext",
    "Supervisor",
    "Worker",
    # Swarm
    "Swarm",
    "SwarmError",
    "SwarmState",
    "HandoffEvent",
    # Middleware
    "AgentMiddleware",
    "MiddlewareContext",
    "StopAgent",
    "TrimLongMessages",
    "ToolBudget",
    "RedactPII",
    "Reflect",
    # Memory
    "AgentMemory",
    "ComposableMemory",
    "Memory",
    "MemoryBlock",
    "StaticBlock",
    "FewShotBlock",
    "SummaryBlock",
    "VectorBlock",
    "FactExtractionBlock",
    "PersistentFactBlock",
    # Chain
    "Chain",
    "ChainResult",
    "ChainState",
    # Durability
    "Checkpointer",
    "SQLiteCheckpointer",
    "PendingInterrupt",
    "interrupt",
    "Resume",
    "InterruptSignal",
    "AlreadyResumed",
    "idempotent",
    "IdempotencyError",
    # LLM
    "LLMClient",
    "Message",
    "StreamEvent",
    "TextDelta",
    # Multimodal
    "Image",
    "PDF",
    "File",
    "ContentPart",
    "normalize_input",
    # Tool
    "Tool",
    "FunctionTool",
    "RESTTool",
    "MCPTool",
    "ToolRegistry",
    "tool",
    "node",
    # Guardrail
    "Guardrail",
    "GuardrailResult",
    "no_pii",
    "no_prompt_injection",
    "openai_moderation",
    "json_valid",
    "toxicity_check",
    # Responsible-AI "Trust Layer"
    "no_secrets",
    "grounded",
    "no_hallucination",
    "banned_topics",
    "allowed_topics",
    "responsible_ai",
    # Guardrail primitives — compute only, no span emission. Pair with
    # emit_guardrail when the runtime isn't the SDK's own agent loop.
    "run_guardrail",
    "guardrail_from_policy_rule",
    "plane_guardrails_for_agent",
    # Prompt
    "PromptRegistry",
    "Prompt",
    # KB
    "LocalKB",
    "PlatformKB",
    "VectorStore",
    "KeywordStore",
    "MetadataStore",
    # Eval
    "evaluate",
    "Dataset",
    "Scorer",
    "EvalResults",
    "Scorecard",
    # Simulation
    "simulate",
    "asimulate",
    "Scenario",
    "SimulatedUser",
    "SimulationResults",
    "generate_scenarios",
    "agenerate_scenarios",
    # Agent hardening
    "harden",
    "aharden",
    "HardeningReport",
    # Optimization (eval-driven self-improvement loop)
    "optimize",
    "aoptimize",
    "OptimizeConfig",
    "OptimizationReport",
    # Trace
    "TraceStore",
    "trace_context",
    "enable_otel_capture",
    "disable_otel_capture",
    "Replay",
    "RedactionPolicy",
    "set_redaction_policy",
    "get_redaction_policy",
    # OpenInference emitters — the standard wire shape for guardrail outcomes
    # and inline eval scores, emitted on whichever tracer you own.
    "set_guardrail_attributes",
    "set_evaluation_attributes",
    "emit_guardrail",
    "emit_evaluation",
]
