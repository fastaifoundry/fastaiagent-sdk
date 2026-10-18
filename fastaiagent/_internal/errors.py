"""All custom exception classes for the FastAIAgent SDK."""

from typing import Any


class FastAIAgentError(Exception):
    """Base exception for all FastAIAgent SDK errors."""


# --- Agent errors ---


class AgentError(FastAIAgentError):
    """Error during agent execution."""


class AgentTimeoutError(AgentError):
    """The run passed ``AgentConfig.timeout_seconds`` (1.91.0)."""


class RunLimitError(AgentError):
    """A run limit could not be checked (1.91.0): ``max_cost_usd`` on a run whose
    spend cannot be priced. Raised rather than letting the run go unbudgeted."""


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


class ChainRoutingError(ChainError):
    """Conditional routing produced no matching edge under ``strict_routing=True``.

    Raised by :func:`fastaiagent.chain.executor._select_outgoing_edges` when a
    chain is built with ``Chain(..., strict_routing=True)`` and a node finishes
    with no edge whose condition matches and no unconditional fallback.
    """


class ChainResumeError(ChainCheckpointError):
    """Resume contract violated — ``resume_value`` does not match checkpoint status.

    Raised by :meth:`Chain.resume` when the caller passes a
    :class:`Resume` value but the latest checkpoint is not in
    ``status="interrupted"``, or when an interrupted checkpoint is resumed
    without a ``resume_value``.

    Subclasses :class:`ChainCheckpointError` so existing
    ``except ChainCheckpointError:`` handlers continue to catch this
    (backwards-compatible with v1.13.x).
    """


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
    """Error from an LLM provider.

    Since 1.89.0 the message built from a provider's response body is scrubbed
    of credentials (a provider may echo a rejected key back). The structured
    fields are parsed from that already-scrubbed body: ``provider_message`` is
    the provider's own explanation, ``provider_code`` its error code or type,
    and ``request_id`` its request id. Each is ``None`` when the provider did
    not send it.
    """

    def __init__(
        self,
        message: str = "",
        status_code: int | None = None,
        *,
        provider: str | None = None,
        provider_message: str | None = None,
        provider_code: str | None = None,
        request_id: str | None = None,
    ):
        self.status_code = status_code
        self.provider = provider
        self.provider_message = provider_message
        self.provider_code = provider_code
        self.request_id = request_id
        super().__init__(message)


class LLMRateLimitError(LLMProviderError):
    """LLM provider rate limit exceeded (HTTP 429).

    Raised since 1.91.0 (it was defined and never raised). A subclass of
    :class:`LLMProviderError`, so ``except LLMProviderError`` still catches it.
    ``retry_after`` is the wait in seconds the provider asked for —
    ``Retry-After`` or ``retry-after-ms`` — or ``None`` when it named none; the
    client's own retries wait that long instead of the exponential backoff.
    """

    def __init__(self, message: str = "", *, retry_after: float | None = None, **kwargs: Any):
        kwargs.setdefault("status_code", 429)
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


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


class MissingPDFBackendError(MultimodalError, ImportError):
    """Local PDF decoding was requested but no PDF engine is installed.

    Also inherits :class:`ImportError` on purpose. Before the ``pdf`` extra
    existed, this path failed with a bare ``ModuleNotFoundError`` (an
    ``ImportError`` subclass) from ``import pymupdf``. Inheriting both keeps
    every existing ``except ImportError:`` handler working while new code can
    catch a typed SDK error and read an actionable message.
    """


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
