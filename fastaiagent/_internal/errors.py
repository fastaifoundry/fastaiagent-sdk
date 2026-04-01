"""All custom exception classes for the FastAIAgent SDK."""


class FastAIAgentError(Exception):
    """Base exception for all FastAIAgent SDK errors."""


# --- Agent errors ---


class AgentError(FastAIAgentError):
    """Error during agent execution."""


class AgentTimeoutError(AgentError):
    """Agent execution exceeded timeout."""


class MaxIterationsError(AgentError):
    """Agent exceeded maximum iteration count in tool-calling loop."""


# --- Chain errors ---


class ChainError(FastAIAgentError):
    """Error during chain execution."""


class ChainCycleError(ChainError):
    """Cycle limit exceeded in chain execution."""


class ChainCheckpointError(ChainError):
    """Error saving or loading a chain checkpoint."""


class ChainStateValidationError(ChainError):
    """Chain state failed schema validation."""


# --- Tool errors ---


class ToolError(FastAIAgentError):
    """Error related to tool operations."""


class ToolExecutionError(ToolError):
    """Error executing a tool."""


class ToolSchemaError(ToolError):
    """Tool schema is invalid or cannot be parsed."""


class SchemaDriftError(ToolError):
    """Tool response schema has drifted from the declared schema."""


# --- LLM errors ---


class LLMError(FastAIAgentError):
    """Error related to LLM operations."""


class LLMProviderError(LLMError):
    """Error from an LLM provider."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""


class LLMAuthError(LLMError):
    """LLM provider authentication failed."""


# --- Guardrail errors ---


class GuardrailError(FastAIAgentError):
    """Error related to guardrail operations."""


class GuardrailBlockedError(GuardrailError):
    """A blocking guardrail rejected the input/output."""

    def __init__(self, guardrail_name: str, message: str = "", results: list | None = None):
        self.guardrail_name = guardrail_name
        self.results = results or []
        super().__init__(message or f"Blocked by guardrail: {guardrail_name}")


# --- Trace errors ---


class TraceError(FastAIAgentError):
    """Error related to tracing operations."""


class ReplayError(FastAIAgentError):
    """Error during Agent Replay operations."""


# --- Platform errors ---


class PlatformError(FastAIAgentError):
    """Error communicating with the FastAIAgent platform."""


class PlatformAuthError(PlatformError):
    """Platform authentication failed."""


class PlatformTierLimitError(PlatformError):
    """Platform tier limit reached."""


class PlatformNotFoundError(PlatformError):
    """Resource not found on the platform."""


class PlatformConnectionError(PlatformError):
    """Cannot connect to the platform."""


class PlatformRateLimitError(PlatformError):
    """Platform rate limit exceeded."""


# --- Prompt errors ---


class PromptError(FastAIAgentError):
    """Error related to prompt operations."""


class PromptNotFoundError(PromptError):
    """Prompt not found in registry."""


class FragmentNotFoundError(PromptError):
    """Prompt fragment not found."""


# --- KB errors ---


class KBError(FastAIAgentError):
    """Error related to knowledge base operations."""


# --- Eval errors ---


class EvalError(FastAIAgentError):
    """Error related to evaluation operations."""
