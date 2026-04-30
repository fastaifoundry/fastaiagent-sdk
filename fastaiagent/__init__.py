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
    MemoryBlock,
    MiddlewareContext,
    RedactPII,
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
from fastaiagent.checkpointers import Checkpointer, PendingInterrupt, SQLiteCheckpointer
from fastaiagent.client import connect, disconnect
from fastaiagent.eval import Dataset, EvalResults, Scorer, evaluate
from fastaiagent.guardrail import Guardrail, GuardrailResult, json_valid, no_pii, toxicity_check
from fastaiagent.kb import KeywordStore, LocalKB, MetadataStore, PlatformKB, VectorStore
from fastaiagent.llm import LLMClient, Message, StreamEvent, TextDelta
from fastaiagent.llm.stream import HandoffEvent
from fastaiagent.prompt import Prompt, PromptRegistry
from fastaiagent.tool import FunctionTool, MCPTool, RESTTool, Tool, ToolRegistry, tool
from fastaiagent.trace import TraceStore, trace_context
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
    raise AttributeError(f"module 'fastaiagent' has no attribute {name!r}")


__all__ = [
    "__version__",
    "connect",
    "disconnect",
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
    # Memory
    "AgentMemory",
    "ComposableMemory",
    "MemoryBlock",
    "StaticBlock",
    "SummaryBlock",
    "VectorBlock",
    "FactExtractionBlock",
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
    # Tool
    "Tool",
    "FunctionTool",
    "RESTTool",
    "MCPTool",
    "ToolRegistry",
    "tool",
    # Guardrail
    "Guardrail",
    "GuardrailResult",
    "no_pii",
    "json_valid",
    "toxicity_check",
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
    # Trace
    "TraceStore",
    "trace_context",
    "Replay",
]
