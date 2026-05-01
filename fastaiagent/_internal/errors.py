"""All custom exception classes for the FastAIAgent SDK."""

from typing import Any


class FastAIAgentError(Exception):
    """Base exception for all FastAIAgent SDK errors."""


# --- Agent errors ---


class AgentError(FastAIAgentError):
    """Error during agent execution."""


class AgentTimeoutError(AgentError):
    """Agent execution exceeded timeout."""


class MaxIterationsError(AgentError):
    """Agent exceeded maximum iteration count in tool-calling loop."""


class StopAgent(AgentError):  # noqa: N818  # cooperative-stop signal, not a policy error
    """Raised by middleware to short-circuit an agent run with a final message.

    When raised inside a middleware hook, the tool-calling loop exits cleanly
    and the agent returns an ``AgentResult`` whose ``output`` is the exception
    message. Unlike ``GuardrailBlockedError``, this is a cooperative stop, not
    a policy rejection — use it for budget limits, completion signals, etc.
    """

    def __init__(self, message: str, reason: str = ""):
        self.reason = reason
        super().__init__(message)


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

    def __init__(self, message: str = "", status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""


class LLMAuthError(LLMError):
    """LLM provider authentication failed."""


# --- Guardrail errors ---


class GuardrailError(FastAIAgentError):
    """Error related to guardrail operations."""


class GuardrailBlockedError(GuardrailError):
    """A blocking guardrail rejected the input/output."""

    def __init__(self, guardrail_name: str, message: str = "", results: list[Any] | None = None):
        self.guardrail_name = guardrail_name
        self.results = results or []
        super().__init__(message or f"Blocked by guardrail: {guardrail_name}")

    def __repr__(self) -> str:
        return (
            f"GuardrailBlockedError(guardrail_name={self.guardrail_name!r}, "
            f"results={self.results!r})"
        )


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

    def __init__(self, message: str = "", resource_type: str = "", limit: int = 0):
        self.resource_type = resource_type
        self.limit = limit
        super().__init__(message or f"Tier limit reached for {resource_type} (limit: {limit})")

    def __repr__(self) -> str:
        return f"PlatformTierLimitError(resource_type={self.resource_type!r}, limit={self.limit!r})"


class PlatformNotFoundError(PlatformError):
    """Resource not found on the platform."""


class PlatformConnectionError(PlatformError):
    """Cannot connect to the platform."""


class PlatformNotConnectedError(PlatformError):
    """SDK is not connected to the platform. Call fa.connect() first."""


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


# --- Multimodal errors ---


class MultimodalError(FastAIAgentError):
    """Error related to multimodal input (image, PDF) handling."""


class UnsupportedFormatError(MultimodalError):
    """The provided media format / scheme is not supported."""


class NonVisionModelError(MultimodalError):
    """The configured LLM model does not support vision input."""

    def __init__(self, message: str = "", *, provider: str = "", model: str = ""):
        self.provider = provider
        self.model = model
        if not message:
            message = (
                f"model {model!r} on provider {provider!r} does not support image input; "
                f"use a vision-capable model (e.g. gpt-4o, claude-sonnet-4-6) "
                f"or pass text-only input"
            )
        super().__init__(message)
