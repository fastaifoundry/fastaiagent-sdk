"""Agent middleware — composable hooks around the run, each model call and each tool.

Where each hook sits::

    before_run                      (once per run, before the input guardrails)
    input guardrail
        -> before_model
        -> wrap_model chain -> LLM  (every model call, re-asks included)
        -> after_model
        -> wrap_tool chain -> tool
        -> output guardrail
    after_run                       (once, on success)
    on_error                        (once, when the run fails)
    on_stream_event                 (each event ``astream`` yields)

All hooks are optional:

- ``before_run``      — rewrite the run's input, or ``raise StopAgent`` to end it
- ``before_model``    — transform the message list before the LLM call
- ``wrap_model``      — onion-wrap each model call (must call ``call_next`` to
  reach the model; may retry it, time it, or answer without it)
- ``after_model``     — inspect or rewrite the LLM response before tool dispatch
- ``wrap_tool``       — onion-wrap each tool invocation (must call ``call_next``)
- ``after_run``       — inspect or replace the finished ``AgentResult``
- ``on_error``        — observe the exception a run failed with (cannot swallow it)
- ``on_stream_event`` — rewrite or drop an event on its way to the caller

Ordering:

- ``before_run`` / ``before_model`` run in declaration order (first middleware
  first)
- ``after_model`` / ``after_run`` / ``on_error`` / ``on_stream_event`` run in
  reverse declaration order (last middleware first) — mirrors the onion pattern
- ``wrap_model`` / ``wrap_tool`` are onions: the first middleware is outermost
  and sees the full downstream chain via ``call_next``

Middleware is orthogonal to ``Guardrail``. Middleware *transforms*; guardrails
*assert*. To block execution cooperatively from middleware, raise
``StopAgent``. Any other exception from a hook fails the run (fail-closed), and
``on_error`` sees it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import StopAgent
from fastaiagent.agent.context import RunContext, RunUsage, current_run_usage_view
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import Message, MessageRole, balance_tool_messages
from fastaiagent.llm.stream import Paused, StreamDone, StreamEvent
from fastaiagent.tool.base import Tool, ToolResult

if TYPE_CHECKING:
    from fastaiagent.agent.agent import AgentResult

logger = logging.getLogger(__name__)

__all__ = [
    "AgentMiddleware",
    "MiddlewareContext",
    "ModelCallNext",
    "Reflect",
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

    One context lives for the whole run — a resume included — so ``scratch``
    carries from ``before_run`` to ``after_run``.

    Attributes:
        run_context: The user-supplied RunContext, if any.
        turn: Zero-indexed LLM iteration within the current ``agent.arun()``.
        tool_call_index: Zero-indexed tool call within the current turn.
        scratch: Mutable dict shared across middleware callbacks for this run.
            Use this to pass data between ``before_model`` and ``after_model``,
            or to accumulate state across tool calls within a run.
        agent_name: Name of the agent running.
        execution_id: The run's id — the one its checkpoints and resumes use.
        llm: The client THIS model call will use. Set it in ``before_model`` or
            ``wrap_model`` to send one call to another model; it is reset to the
            agent's client before every turn, so the change lasts one call.
        tools: The tools THIS model call offers. Filter or extend the list in
            ``before_model`` / ``wrap_model``; a tool added here can be called
            on this turn. Reset to the agent's tools before every turn. A
            structured-output or guardrail re-ask offers no tools.
        state: (property) The CURRENT middleware's own JSON-safe dict — each
            middleware sees its own. Unlike ``scratch`` it is saved in
            checkpoints and restored when the run is resumed or forked.
    """

    run_context: RunContext[Any] | None = None
    turn: int = 0
    tool_call_index: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    execution_id: str = ""
    llm: Any = None
    tools: list[Tool] = field(default_factory=list)
    # Per-middleware state, keyed by the pipeline's key for each middleware;
    # ``_state_key`` names the one whose hook is running.
    _states: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _state_key: str = field(default="", repr=False)

    @property
    def usage(self) -> RunUsage:
        """The run's tokens and cost so far (read-only, from the run's running
        total — the number ``AgentResult.cost`` and ``cost_limit`` read)."""
        return current_run_usage_view()

    @property
    def state(self) -> dict[str, Any]:
        """This middleware's saved state (1.91.0, 1B). A JSON-safe dict: it goes
        into every checkpoint under ``state_snapshot["middleware_state"]``, so a
        budget or a counter kept here survives a resume. Use ``scratch`` for
        anything that should not be saved."""
        return self._states.setdefault(self._state_key, {})

    def saved_state(self) -> dict[str, Any] | None:
        """Every middleware's state as checkpoints store it — ``None`` when no
        middleware has any, so a checkpoint keeps its shape. Raises
        ``TypeError`` (naming the middleware) for state that is not JSON:
        silently dropping it would reset a budget on the next resume."""
        kept = {key: value for key, value in self._states.items() if value}
        if not kept:
            return None
        for key, value in kept.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"middleware {key!r} keeps state that cannot be checkpointed "
                    f"({exc}). ctx.state must be JSON-safe; keep other objects in "
                    "ctx.scratch, which is not saved."
                ) from None
        copied: dict[str, Any] = json.loads(json.dumps(kept))
        return copied

    def restore_state(self, saved: dict[str, Any] | None) -> None:
        """Load state saved by :meth:`saved_state` (a resume or a fork)."""
        if saved:
            self._states = json.loads(json.dumps(saved))


ToolCallNext = Callable[[Tool, dict[str, Any]], Awaitable[ToolResult]]
#: What ``wrap_model`` calls to reach the next middleware, and finally the model.
ModelCallNext = Callable[[list[Message]], Awaitable[LLMResponse]]


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
    #: Tools this middleware brings (1.91.0, 1B) — merged into the run's tools.
    #: A name that clashes with the agent's tools or another middleware's is an
    #: error when the ``Agent`` is built.
    tools: Sequence[Tool] = ()
    #: Text for the model (1.91.0, 1B): a string, or ``callable(ctx) -> str``
    #: (a method works). All middleware's instructions form ONE system message
    #: right after the system prompt, before memory, in declaration order.
    instructions: str | Callable[[MiddlewareContext], str] | None = None

    async def before_run(self, ctx: MiddlewareContext, input: Any) -> Any:
        """Once per run, before the input guardrails — so they judge what this
        returns. ``input`` is what the run was given (a string, or a multimodal
        list). Return it, or a replacement. ``raise StopAgent`` to end the run
        before any model call.

        Also fires when a run is **resumed**, with the run's original input
        text; there the return value is not used — the saved history is what
        the run continues from.
        """
        return input

    async def after_run(self, ctx: MiddlewareContext, result: AgentResult) -> AgentResult:
        """Once, when the run finishes — not when it pauses or fails. Return the
        result, or a replacement. On ``astream`` the reply has already been
        streamed; what this returns is what ``stream()`` returns.
        """
        return result

    async def wrap_model(
        self,
        ctx: MiddlewareContext,
        messages: list[Message],
        call_next: ModelCallNext,
    ) -> LLMResponse:
        """Wrap one model call — every one, the structured-output and guardrail
        re-asks included.

        ``await call_next(messages)`` reaches the next middleware and finally the
        model (``ctx.llm``, offering ``ctx.tools``). Retry by calling it again,
        answer without the model by returning an ``LLMResponse`` yourself, or
        time it. On ``astream`` the model's deltas stream to the caller as they
        arrive; a response returned without calling the model is streamed as one
        ``TextDelta``.
        """
        return await call_next(messages)

    async def on_error(self, ctx: MiddlewareContext, error: BaseException) -> None:
        """Once, when the run fails — any exception except ``StopAgent``, which
        ends a run cooperatively, and a pause. Observe it (log, count, alert);
        the run still fails with ``error``. An exception raised here is logged
        and never replaces ``error``.
        """
        return None

    async def on_stream_event(
        self, ctx: MiddlewareContext, event: StreamEvent
    ) -> StreamEvent | None:
        """Each event ``astream`` is about to yield. Return it, a replacement, or
        ``None`` to drop it. ``StreamDone`` and ``Paused`` are always delivered
        — a ``None`` for those is ignored, since a consumer waits on them.
        """
        return event

    async def before_model(self, ctx: MiddlewareContext, messages: list[Message]) -> list[Message]:
        """Transform messages before the LLM call.

        Return the (possibly modified) messages. May return a new list or
        mutate in place. Raising propagates; use ``StopAgent`` to end cleanly.
        """
        return messages

    async def after_model(self, ctx: MiddlewareContext, response: LLMResponse) -> LLMResponse:
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
        # The key each middleware's ``ctx.state`` is saved under: its ``name``
        # (or class name), with ``#2``, ``#3``… for repeats. Stable across a
        # resume as long as the middleware list is the same.
        self._keys: list[str] = []
        seen: dict[str, int] = {}
        for mw in self._middleware:
            base = mw.name or type(mw).__name__
            seen[base] = seen.get(base, 0) + 1
            self._keys.append(base if seen[base] == 1 else f"{base}#{seen[base]}")

    @property
    def middleware(self) -> list[AgentMiddleware]:
        return list(self._middleware)

    def _each(self, *, reverse: bool = False) -> list[tuple[str, AgentMiddleware]]:
        pairs = list(zip(self._keys, self._middleware, strict=True))
        return list(reversed(pairs)) if reverse else pairs

    @property
    def tools(self) -> list[Tool]:
        """Every tool the middleware brings, in declaration order."""
        return [t for mw in self._middleware for t in (mw.tools or ())]

    def instructions_text(self, ctx: MiddlewareContext) -> str:
        """All middleware instructions, joined in declaration order ('' if none)."""
        parts: list[str] = []
        for key, mw in self._each():
            text = mw.instructions
            if callable(text):
                ctx._state_key = key
                text = text(ctx)
            if text and str(text).strip():
                parts.append(str(text).strip())
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return len(self._middleware)

    def __bool__(self) -> bool:
        return bool(self._middleware)

    def overrides(self, hook: str) -> bool:
        """Does any middleware implement ``hook``? The loops take today's exact
        path when none does — ``wrap_model`` on a stream, in particular, costs a
        task and a queue."""
        base = getattr(AgentMiddleware, hook)
        return any(getattr(type(mw), hook, base) is not base for mw in self._middleware)

    async def apply_before_run(self, ctx: MiddlewareContext, input: Any) -> Any:
        for key, mw in self._each():
            ctx._state_key = key
            input = await mw.before_run(ctx, input)
        return input

    async def apply_after_run(self, ctx: MiddlewareContext, result: AgentResult) -> AgentResult:
        for key, mw in self._each(reverse=True):
            ctx._state_key = key
            result = await mw.after_run(ctx, result)
        return result

    async def notify_error(self, ctx: MiddlewareContext, error: BaseException) -> None:
        """Tell every middleware the run failed. Never raises: the caller is
        about to re-raise ``error``, and a hook's own bug must not replace it."""
        for key, mw in self._each(reverse=True):
            ctx._state_key = key
            try:
                await mw.on_error(ctx, error)
            except Exception as hook_error:  # noqa: BLE001 - see docstring
                logger.warning(
                    "middleware %s on_error raised %s: %s (the run's own error stands)",
                    mw.name or type(mw).__name__,
                    type(hook_error).__name__,
                    hook_error,
                )

    async def apply_stream_event(
        self, ctx: MiddlewareContext, event: StreamEvent
    ) -> StreamEvent | None:
        for key, mw in self._each(reverse=True):
            ctx._state_key = key
            rewritten = await mw.on_stream_event(ctx, event)
            if rewritten is None:
                if isinstance(event, (StreamDone, Paused)):
                    continue  # always delivered: a consumer waits on these
                return None
            event = rewritten
        return event

    async def call_model(
        self,
        ctx: MiddlewareContext,
        messages: list[Message],
        terminal: ModelCallNext,
    ) -> LLMResponse:
        """Run one model call through the ``wrap_model`` onion; ``terminal`` is
        the real call. The first middleware is outermost."""
        chain: ModelCallNext = terminal
        for key, mw in self._each(reverse=True):
            prev = chain

            async def _wrapper(
                m: list[Message],
                _key: str = key,
                _mw: AgentMiddleware = mw,
                _prev: ModelCallNext = prev,
            ) -> LLMResponse:
                async def _next(m2: list[Message]) -> LLMResponse:
                    try:
                        return await _prev(m2)
                    finally:
                        ctx._state_key = _key  # back to this middleware's state

                ctx._state_key = _key
                return await _mw.wrap_model(ctx, m, _next)

            chain = _wrapper
        return await chain(messages)

    async def apply_before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        for key, mw in self._each():
            ctx._state_key = key
            messages = await mw.before_model(ctx, messages)
        return messages

    async def apply_after_model(self, ctx: MiddlewareContext, response: LLMResponse) -> LLMResponse:
        for key, mw in self._each(reverse=True):
            ctx._state_key = key
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
        for key, mw in self._each(reverse=True):
            prev = chain

            async def _wrapper(
                t: Tool,
                a: dict[str, Any],
                _key: str = key,
                _mw: AgentMiddleware = mw,
                _prev: ToolCallNext = prev,
            ) -> ToolResult:
                async def _next(t2: Tool, a2: dict[str, Any]) -> ToolResult:
                    try:
                        return await _prev(t2, a2)
                    finally:
                        ctx._state_key = _key  # back to this middleware's state

                ctx._state_key = _key
                return await _mw.wrap_tool(ctx, t, a, _next)

            chain = _wrapper
        return await chain(tool, args)


# ---------------------------------------------------------------------------
# Built-in middleware
# ---------------------------------------------------------------------------

#: What the model is told about a tool call whose result the trimmer cut away.
#: The tool DID run — its result is simply no longer in the window — so this
#: must not read like a failure the model should retry.
TRIMMED_NOTE = "this tool result is no longer in the conversation window"


class TrimLongMessages(AgentMiddleware):
    """Keep only the most recent ``keep_last`` messages plus any leading
    SystemMessage.

    Useful for long-running agents to avoid context-window blowouts without
    the cost of summarization.

    The window is a message count, so it can land *between* an assistant message
    carrying ``tool_calls`` and the results that answer them. Both halves of that
    split are rejected by OpenAI and Anthropic, so the tail is re-balanced before
    it is returned: a tool result whose parent call was trimmed away is dropped,
    and a call whose results were trimmed away is answered with a note saying the
    result is no longer in the history. That can leave slightly more than
    ``keep_last`` messages — a provider 400 is the more expensive of the two.
    """

    name = "trim_long_messages"

    def __init__(self, keep_last: int = 20):
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.keep_last = keep_last

    async def before_model(self, ctx: MiddlewareContext, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.keep_last:
            return messages

        # Preserve every LEADING system message and keep the tail. Memory blocks
        # render their context (a pinned StaticBlock, a summary, known facts) as
        # system messages right after the system prompt; keeping only
        # ``messages[0]`` trimmed the user's pinned memory away on any long run
        # (1.90.0, F11). The leading run is the prompt's preamble, not history.
        n_head = 0
        while n_head < len(messages) and messages[n_head].role == MessageRole.system:
            n_head += 1
        head: list[Message] = list(messages[:n_head])
        body = messages[n_head:]
        if len(body) <= self.keep_last:
            return messages

        tail = balance_tool_messages(body[-self.keep_last :], note=TRIMMED_NOTE)
        return head + tail


class ToolBudget(AgentMiddleware):
    """Cap the number of tool invocations per agent run.

    When the budget is exhausted, raises ``StopAgent`` which ends the run
    cooperatively with a final message.

    The count lives in ``ctx.state``, so it is checkpointed: a run that pauses
    and is resumed keeps counting from where it was (1.91.0, decision B1-5).
    Before, it lived in ``ctx.scratch`` and every resume started a fresh budget.
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
        used = int(ctx.state.get("used", 0))
        if used >= self.max_calls:
            raise StopAgent(self.message, reason="tool_budget")
        ctx.state["used"] = used + 1
        return await call_next(tool, args)


#: The built-in tool a deferred toolset is reached through (1.91.0, 1G).
LOAD_TOOLS = "load_tools"


class _LoadToolsTool(Tool):
    """``load_tools(names)`` — answered by :class:`_DeferredTools` in
    ``wrap_tool``, never executed itself."""

    origin = "builtin"

    def __init__(self, deferred: list[Tool]) -> None:
        listing = "\n".join(
            f"- {t.name}: {((t.description or '').strip().splitlines() or [''])[0]}"
            for t in deferred
        )
        super().__init__(
            name=LOAD_TOOLS,
            description=(
                "Load tools before you call them. These tools are available but not "
                f"loaded yet:\n{listing}\nCall load_tools with the names you need; "
                "they can be called from your next step on."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of the tools to load.",
                    }
                },
                "required": ["names"],
            },
        )

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        return ToolResult(error="load_tools runs inside the agent loop only")


class _DeferredTools(AgentMiddleware):
    """Keeps ``Tool(deferred=True)`` tools out of the request until the model
    loads them (1.91.0, 1G). Added by ``Agent`` itself — innermost, so user
    middleware sees every tool — only when some tool is deferred.

    The loaded names are this middleware's ``ctx.state``, so they are
    checkpointed and a resumed run keeps them.
    """

    name = "deferred_tools"

    def __init__(self, deferred: list[Tool]) -> None:
        self._deferred = {t.name: t for t in deferred}
        self.tools = [_LoadToolsTool(deferred)]

    def _loaded(self, ctx: MiddlewareContext) -> list[str]:
        return list(ctx.state.get("loaded", []))

    async def before_model(self, ctx: MiddlewareContext, messages: list[Message]) -> list[Message]:
        loaded = set(self._loaded(ctx))
        pending = [n for n in self._deferred if n not in loaded]
        ctx.tools = [
            t
            for t in ctx.tools
            if (t.name not in self._deferred or t.name in loaded)
            and (t.name != LOAD_TOOLS or pending)
        ]
        return messages

    async def wrap_tool(
        self,
        ctx: MiddlewareContext,
        tool: Tool,
        args: dict[str, Any],
        call_next: ToolCallNext,
    ) -> ToolResult:
        loaded = self._loaded(ctx)
        if tool.name == LOAD_TOOLS:
            names = args.get("names") or []
            if isinstance(names, str):
                names = [names]
            known = [n for n in names if n in self._deferred]
            unknown = [n for n in names if n not in self._deferred]
            for n in known:
                if n not in loaded:
                    loaded.append(n)
            ctx.state["loaded"] = loaded
            parts = []
            if known:
                parts.append(f"Loaded: {', '.join(known)}. You can call them now.")
            if unknown:
                parts.append(
                    f"Not available: {', '.join(unknown)}. Tools you can load: "
                    f"{', '.join(self._deferred)}."
                )
            return ToolResult(output=" ".join(parts) or "No tools named.")
        if tool.name in self._deferred and tool.name not in loaded and ctx.llm is not None:
            # A MODEL called it without loading: refuse rather than run a tool
            # whose definition it never saw. ``ctx.llm`` is set by every turn of
            # the agent loop; a direct call (MCP ``expose_tools``) has no model
            # to hide definitions from and runs as any exposed tool does.
            return ToolResult(
                output=f"{tool.name} is not loaded. Call load_tools with "
                f'["{tool.name}"] first, then call it.'
            )
        return await call_next(tool, args)


class RedactPII(AgentMiddleware):
    """Redact PII from outbound prompts and inbound responses.

    By default this delegates to
    :func:`fastaiagent._internal.safety_detectors.detect_pii` — the **same**
    detector behind the ``no_pii()`` builtin, the ``pii`` guardrail type and the
    ``PIILeakage`` scorer. Pass ``patterns=`` to use your own regexes instead;
    that path is unchanged and does exactly what you tell it to.

    **Why the default moved to the shared detector.** This class used to carry a
    second, private copy of the PII regexes, and the copy had drifted: its
    card-ish pattern was a bare ``\\b(?:\\d[ \\-]?){13,19}\\b``, with **no Luhn
    check**. So any 13-19 digit run — an order number, an invoice id, an IMEI,
    a concatenated timestamp — was redacted as a credit card, and that
    corruption is what the model saw and what was replayed in a guardrail
    re-ask. `detect_pii` Luhn-validates card candidates for exactly this
    reason, and it was one import away.

    Redacted spans are replaced with ``placeholder`` in what the model is sent.
    ``before_model`` redacts **copies** of the messages: the objects it is
    handed can be the ones stored in memory, so rewriting them in place used to
    change conversation history after the fact. Memory keeps what was said; to
    keep PII out of memory, redact before it reaches memory.
    """

    name = "redact_pii"

    def __init__(
        self,
        patterns: list[str] | None = None,
        placeholder: str = "[REDACTED]",
        entities: tuple[str, ...] | None = None,
    ):
        # ``patterns`` keeps its old meaning: your regexes, applied verbatim.
        # Only the *default* path changed, and only to stop being wrong.
        self.patterns = [re.compile(p) for p in patterns] if patterns is not None else None
        self.placeholder = placeholder
        self.entities = entities

    def _redact(self, text: Any) -> Any:
        """Redact one message body.

        Returns non-string content untouched. ``msg.content`` is a
        ``list[ContentPart]`` for a multimodal message, and the old
        implementation called ``pat.sub`` on it and raised
        ``TypeError: expected string or bytes-like object`` — a middleware that
        crashes the run it was added to protect.
        """
        if not text or not isinstance(text, str):
            return text

        if self.patterns is not None:
            result = text
            for pat in self.patterns:
                result = pat.sub(self.placeholder, result)
            return result

        from fastaiagent._internal.safety_detectors import (
            DEFAULT_PII_ENTITIES,
            detect_pii,
            mask_spans,
        )

        matches = detect_pii(text, entities=list(self.entities or DEFAULT_PII_ENTITIES))
        # ``mask_spans`` merges overlapping spans and replaces right-to-left, so
        # one value matched by two patterns is redacted once and cleanly rather
        # than leaving fragments of itself behind.
        return mask_spans(text, [(m.start, m.end) for m in matches], self.placeholder)

    async def before_model(self, ctx: MiddlewareContext, messages: list[Message]) -> list[Message]:
        # Replace each message with a redacted copy. Never edit the objects
        # themselves: they can be the ones memory stores. Keep the list itself
        # (``[:]``) — the executor and the agent hold this same list and keep
        # appending to it.
        messages[:] = [
            m.model_copy(update={"content": self._redact(m.content)}) if m.content else m
            for m in messages
        ]
        return messages

    async def after_model(self, ctx: MiddlewareContext, response: LLMResponse) -> LLMResponse:
        if content := response.content:
            response.content = self._redact(content)
        return response


def _record_reflect_skip(ctx: MiddlewareContext, exc: BaseException) -> None:
    """``Reflect`` could not review the draft: a warning, a count in
    ``ctx.scratch["reflect_skipped"]`` and a ``reflect.skipped`` event on the
    current span (the error's type only — its text may quote the draft)."""
    ctx.scratch["reflect_skipped"] = ctx.scratch.get("reflect_skipped", 0) + 1
    logger.warning(
        "Reflect could not review the answer (%s) and kept the draft unreviewed",
        type(exc).__name__,
    )
    try:
        from opentelemetry import trace as _otel_trace

        _otel_trace.get_current_span().add_event(
            "reflect.skipped", {"reflect.error_type": type(exc).__name__}
        )
    except Exception:  # pragma: no cover - tracing must never break the run
        logger.debug("reflect.skipped span event failed", exc_info=True)


class Reflect(AgentMiddleware):
    """Self-critique the agent's final answer and revise it.

    On each **terminal** text response (one carrying no tool calls), the model is
    asked to critique its own answer against optional ``facts`` (non-negotiable
    truths / policies) and ``criteria``, then return a corrected answer.
    Responses that carry tool calls are passed through untouched — only the final
    answer is reflected.

    Opt-in via ``Agent(middleware=[Reflect(facts=[...])])``. Costs one extra LLM
    call per final answer (more if ``max_revisions > 1`` and the reviewer is not
    yet satisfied). Fails open: any reviewer error keeps the original answer —
    and is recorded as a warning, ``ctx.scratch["reflect_skipped"]`` and a
    ``reflect.skipped`` span event, so an unreviewed answer is never mistaken for
    an approved one.

    Example:
        agent = Agent(
            name="grounded-writer",
            llm=llm,
            middleware=[Reflect(facts=["Refunds are only valid within 30 days."])],
        )
    """

    name = "reflect"

    def __init__(
        self,
        *,
        llm: Any = None,
        facts: list[str] | None = None,
        criteria: str | None = None,
        max_revisions: int = 1,
    ):
        self._llm = llm
        self.facts = facts or []
        self.criteria = criteria
        self.max_revisions = max_revisions

    async def after_model(self, ctx: MiddlewareContext, response: LLMResponse) -> LLMResponse:
        # Only reflect on terminal text answers — skip tool-call turns.
        if response.tool_calls or not (response.content and response.content.strip()):
            return response

        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()

        facts_block = ""
        if self.facts:
            joined = "\n".join(f"- {f}" for f in self.facts)
            facts_block = f"Non-negotiable facts/policies the answer MUST respect:\n{joined}\n\n"
        criteria_block = f"Quality criteria: {self.criteria}\n\n" if self.criteria else ""

        answer = response.content
        revisions = 0
        for _ in range(max(1, self.max_revisions)):
            prompt = (
                "Critique the draft answer below, then return an improved version.\n\n"
                f"{facts_block}{criteria_block}"
                f"Draft answer:\n{answer}\n\n"
                "If the draft is already correct and compliant, return it unchanged.\n"
                'Respond with JSON only: {"ok": <true|false>, "revised": "<final answer>"}'
            )
            try:
                resp = await llm.acomplete(
                    [
                        SystemMessage(
                            "You are a meticulous self-reviewer. Respond with JSON only."
                        ),
                        UserMessage(prompt),
                    ]
                )
                raw = (resp.content or "").strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
                data = json.loads(raw)
            except Exception as exc:
                # Fail open — keep the latest good answer if reflection errors —
                # but say so (1.91.0, 1E): a review that silently did not happen
                # read exactly like one that approved the draft.
                _record_reflect_skip(ctx, exc)
                break

            revised = str(data.get("revised") or answer)
            if revised.strip() and revised.strip() != answer.strip():
                answer = revised
                revisions += 1
            if bool(data.get("ok", True)):
                break

        response.content = answer
        ctx.scratch["reflect_revisions"] = ctx.scratch.get("reflect_revisions", 0) + revisions
        return response
