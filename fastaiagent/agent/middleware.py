"""Agent middleware — composable pre/post model hooks and tool wrappers.

Middleware runs between the agent's guardrail ring and the LLM/tool loop::

    input guardrail
        -> before_model
        -> LLM
        -> after_model
        -> wrap_tool chain -> tool
        -> output guardrail

Three hook types, all optional:

- ``before_model``   — transform the message list before the LLM call
- ``after_model``    — inspect or rewrite the LLM response before tool dispatch
- ``wrap_tool``      — onion-wrap each tool invocation (must call ``call_next``)

Ordering:

- ``before_model`` runs in declaration order (first middleware first)
- ``after_model`` runs in reverse declaration order (last middleware first) —
  mirrors the onion pattern
- ``wrap_tool`` is onion: the first middleware is outermost, sees the full
  downstream chain via ``call_next``

Middleware is orthogonal to ``Guardrail``. Middleware *transforms*; guardrails
*assert*. To block execution cooperatively from middleware, raise
``StopAgent``.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastaiagent._internal.errors import StopAgent
from fastaiagent.agent.context import RunContext
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import Message, MessageRole
from fastaiagent.tool.base import Tool, ToolResult

__all__ = [
    "AgentMiddleware",
    "MiddlewareContext",
    "RedactPII",
    "StopAgent",
    "ToolBudget",
    "ToolCallNext",
    "TrimLongMessages",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class MiddlewareContext:
    """Per-turn context passed to every middleware callback.

    Attributes:
        run_context: The user-supplied RunContext, if any.
        turn: Zero-indexed LLM iteration within the current ``agent.arun()``.
        tool_call_index: Zero-indexed tool call within the current turn.
        scratch: Mutable dict shared across middleware callbacks for this run.
            Use this to pass data between ``before_model`` and ``after_model``,
            or to accumulate state across tool calls within a run.
        agent_name: Name of the agent running.
    """

    run_context: RunContext[Any] | None = None
    turn: int = 0
    tool_call_index: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""


ToolCallNext = Callable[[Tool, dict[str, Any]], Awaitable[ToolResult]]


# ---------------------------------------------------------------------------
# Middleware base class
# ---------------------------------------------------------------------------


class AgentMiddleware:
    """Base class for agent middleware.

    Subclass and override one or more of the hook methods. All hooks are
    optional; the default implementations are no-ops.

    Hooks must be idempotent and side-effect-safe — a retry or fork may
    re-invoke them on the same input.

    Example:
        class LogPrompts(AgentMiddleware):
            async def before_model(self, ctx, messages):
                print(f"Turn {ctx.turn}: {len(messages)} messages")
                return messages
    """

    name: str = ""

    async def before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        """Transform messages before the LLM call.

        Return the (possibly modified) messages. May return a new list or
        mutate in place. Raising propagates; use ``StopAgent`` to end cleanly.
        """
        return messages

    async def after_model(
        self, ctx: MiddlewareContext, response: LLMResponse
    ) -> LLMResponse:
        """Inspect or rewrite the LLM response before tool dispatch.

        Return the (possibly modified) response. Raising propagates; use
        ``Guardrail`` for block-semantics.
        """
        return response

    async def wrap_tool(
        self,
        ctx: MiddlewareContext,
        tool: Tool,
        args: dict[str, Any],
        call_next: ToolCallNext,
    ) -> ToolResult:
        """Wrap a single tool invocation.

        Must ``await call_next(tool, args)`` to proceed. Short-circuit by
        returning a ``ToolResult`` directly without calling ``call_next``.
        """
        return await call_next(tool, args)


# ---------------------------------------------------------------------------
# Pipeline composer
# ---------------------------------------------------------------------------


class _MiddlewarePipeline:
    """Runs a middleware stack in the correct order for each hook.

    Internal — constructed by ``Agent`` from the user's middleware list.
    """

    def __init__(self, middleware: list[AgentMiddleware]):
        self._middleware = list(middleware)

    @property
    def middleware(self) -> list[AgentMiddleware]:
        return list(self._middleware)

    def __len__(self) -> int:
        return len(self._middleware)

    def __bool__(self) -> bool:
        return bool(self._middleware)

    async def apply_before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        for mw in self._middleware:
            messages = await mw.before_model(ctx, messages)
        return messages

    async def apply_after_model(
        self, ctx: MiddlewareContext, response: LLMResponse
    ) -> LLMResponse:
        for mw in reversed(self._middleware):
            response = await mw.after_model(ctx, response)
        return response

    async def invoke_tool(
        self,
        ctx: MiddlewareContext,
        tool: Tool,
        args: dict[str, Any],
        terminal: ToolCallNext,
    ) -> ToolResult:
        """Run the tool through the onion.

        ``terminal`` is the real tool-invocation closure. Middleware wraps
        around it; the first middleware is outermost.
        """
        if not self._middleware:
            return await terminal(tool, args)

        chain: ToolCallNext = terminal
        for mw in reversed(self._middleware):
            prev = chain

            async def _wrapper(
                t: Tool,
                a: dict[str, Any],
                _mw: AgentMiddleware = mw,
                _prev: ToolCallNext = prev,
            ) -> ToolResult:
                return await _mw.wrap_tool(ctx, t, a, _prev)

            chain = _wrapper
        return await chain(tool, args)


# ---------------------------------------------------------------------------
# Built-in middleware
# ---------------------------------------------------------------------------


class TrimLongMessages(AgentMiddleware):
    """Keep only the most recent ``keep_last`` messages plus any leading
    SystemMessage.

    Useful for long-running agents to avoid context-window blowouts without
    the cost of summarization.
    """

    name = "trim_long_messages"

    def __init__(self, keep_last: int = 20):
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.keep_last = keep_last

    async def before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        if len(messages) <= self.keep_last:
            return messages

        # Preserve a leading system message (if present) and keep the tail.
        head: list[Message] = []
        body = messages
        if messages and messages[0].role == MessageRole.system:
            head = [messages[0]]
            body = messages[1:]

        tail = body[-self.keep_last :]
        return head + tail


class ToolBudget(AgentMiddleware):
    """Cap the number of tool invocations per agent run.

    When the budget is exhausted, raises ``StopAgent`` which ends the run
    cooperatively with a final message.
    """

    name = "tool_budget"

    def __init__(self, max_calls: int = 10, message: str = "Tool budget exhausted."):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.message = message

    async def wrap_tool(
        self,
        ctx: MiddlewareContext,
        tool: Tool,
        args: dict[str, Any],
        call_next: ToolCallNext,
    ) -> ToolResult:
        key = f"_tool_budget_used_{id(self)}"
        used = ctx.scratch.get(key, 0)
        if used >= self.max_calls:
            raise StopAgent(self.message, reason="tool_budget")
        ctx.scratch[key] = used + 1
        return await call_next(tool, args)


# Default PII patterns: US-style email, phone, SSN, and credit-card-ish
# digit runs. Not exhaustive — users should supply domain-specific patterns.
_DEFAULT_PII_PATTERNS = [
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",          # email
    r"\b(?:\+?1[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}\b",  # US phone
    r"\b\d{3}-\d{2}-\d{4}\b",                                      # SSN
    r"\b(?:\d[ \-]?){13,19}\b",                                    # card-ish
]


class RedactPII(AgentMiddleware):
    """Redact common PII patterns from outbound prompts and inbound responses.

    Patterns are matched with regex. By default covers email, US phone, SSN,
    and long digit runs (credit-card-ish). Override via the ``patterns`` arg.

    Redactions use a placeholder that preserves the match length roughly for
    display continuity; the full pre-redaction text is not retained.
    """

    name = "redact_pii"

    def __init__(
        self,
        patterns: list[str] | None = None,
        placeholder: str = "[REDACTED]",
    ):
        raw_patterns = patterns if patterns is not None else _DEFAULT_PII_PATTERNS
        self.patterns = [re.compile(p) for p in raw_patterns]
        self.placeholder = placeholder

    def _redact(self, text: str) -> str:
        if not text:
            return text
        result = text
        for pat in self.patterns:
            result = pat.sub(self.placeholder, result)
        return result

    async def before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        # Redact in-place on message content. Messages are Pydantic BaseModels
        # with mutable fields — safe to mutate.
        for msg in messages:
            if msg.content:
                msg.content = self._redact(msg.content)
        return messages

    async def after_model(
        self, ctx: MiddlewareContext, response: LLMResponse
    ) -> LLMResponse:
        if content := response.content:
            response.content = self._redact(content)
        return response
