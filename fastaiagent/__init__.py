"""FastAIAgent SDK — Build, debug, evaluate, and operate AI agents."""

from fastaiagent._version import __version__
from fastaiagent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.chain import Chain, ChainResult, ChainState
from fastaiagent.eval import Dataset, EvalResults, Scorer, evaluate
from fastaiagent.guardrail import Guardrail, GuardrailResult, no_pii, json_valid, toxicity_check
from fastaiagent.kb import LocalKB
from fastaiagent.llm import LLMClient, Message
from fastaiagent.prompt import Prompt, PromptRegistry
from fastaiagent.tool import FunctionTool, MCPTool, RESTTool, Tool, tool
from fastaiagent.trace import TraceStore, trace_context
from fastaiagent.trace.replay import Replay
from fastaiagent.client import FastAI


def init(
    api_key: str | None = None,
    target: str = "https://app.fastaiagent.net",
    project: str | None = None,
) -> FastAI | None:
    """Quick setup: connect to platform + enable tracing."""
    if api_key:
        return FastAI(api_key=api_key, target=target, project=project)
    return None


__all__ = [
    "__version__",
    "init",
    # Agent
    "Agent",
    "AgentConfig",
    "AgentResult",
    # Chain
    "Chain",
    "ChainResult",
    "ChainState",
    # LLM
    "LLMClient",
    "Message",
    # Tool
    "Tool",
    "FunctionTool",
    "RESTTool",
    "MCPTool",
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
    # Platform
    "FastAI",
]
