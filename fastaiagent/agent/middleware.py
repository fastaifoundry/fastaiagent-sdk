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
from collections.abc import Awaitable, Callable
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
    """

    run_context: RunContext[Any] | None = None
    turn: int = 0
    tool_call_index: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    execution_id: str = ""
    llm: Any = None
    tools: list[Tool] = field(default_factory=list)

    @property
    def usage(self) -> RunUsage:
        """The run's tokens and cost so far (read-only, from the run's running
        total — the number ``AgentResult.cost`` and ``cost_limit`` read)."""
        return current_run_usage_view()


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

    @property
    def middleware(self) -> list[AgentMiddleware]:
        return list(self._middleware)

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
        for mw in self._middleware:
            input = await mw.before_run(ctx, input)
        return input

    async def apply_after_run(self, ctx: MiddlewareContext, result: AgentResult) -> AgentResult:
        for mw in reversed(self._middleware):
            result = await mw.after_run(ctx, result)
        return result

    async def notify_error(self, ctx: MiddlewareContext, error: BaseException) -> None:
        """Tell every middleware the run failed. Never raises: the caller is
        about to re-raise ``error``, and a hook's own bug must not replace it."""
        for mw in reversed(self._middleware):
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
        for mw in reversed(self._middleware):
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
        for mw in reversed(self._middleware):
            prev = chain

            async def _wrapper(
                m: list[Message],
                _mw: AgentMiddleware = mw,
                _prev: ModelCallNext = prev,
            ) -> LLMResponse:
                return await _mw.wrap_model(ctx, m, _prev)

            chain = _wrapper
        return await chain(messages)

    async def apply_before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        for mw in self._middleware:
            messages = await mw.before_model(ctx, messages)
        return messages

    async def apply_after_model(self, ctx: MiddlewareContext, response: LLMResponse) -> LLMResponse:
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
