"""FastAIAgent SDK — Build, debug, evaluate, and operate AI agents."""

from fastaiagent._internal.errors import StopAgent
from fastaiagent._version import __version__
from fastaiagent.agent import (
    Agent,
    AgentConfig,
    AgentMiddleware,
    AgentResult,
    MiddlewareContext,
    RedactPII,
    RunContext,
    Supervisor,
    ToolBudget,
    TrimLongMessages,
    Worker,
)
from fastaiagent.chain import Chain, ChainResult, ChainState
from fastaiagent.client import connect, disconnect
from fastaiagent.eval import Dataset, EvalResults, Scorer, evaluate
from fastaiagent.guardrail import Guardrail, GuardrailResult, json_valid, no_pii, toxicity_check
from fastaiagent.kb import LocalKB
from fastaiagent.llm import LLMClient, Message, StreamEvent, TextDelta
from fastaiagent.prompt import Prompt, PromptRegistry
from fastaiagent.tool import FunctionTool, MCPTool, RESTTool, Tool, ToolRegistry, tool
from fastaiagent.trace import TraceStore, trace_context
from fastaiagent.trace.replay import Replay


def __getattr__(name: str) -> object:
    if name == "is_connected":
        from fastaiagent.client import _connection

        return _connection.is_connected
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
    # Middleware
    "AgentMiddleware",
    "MiddlewareContext",
    "StopAgent",
    "TrimLongMessages",
    "ToolBudget",
    "RedactPII",
    # Chain
    "Chain",
    "ChainResult",
    "ChainState",
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
