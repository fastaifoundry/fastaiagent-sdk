"""Agent tool-calling loop executor."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fastaiagent._internal.errors import MaxIterationsError, StopAgent, ToolExecutionError
from fastaiagent.agent.middleware import MiddlewareContext, _MiddlewarePipeline
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.chain.interrupt import InterruptSignal, _agent_path
from fastaiagent.checkpointers.protocol import Checkpointer, PendingInterrupt
from fastaiagent.guardrail.executor import GuardrailOutcome, execute_guardrails
from fastaiagent.guardrail.guardrail import GuardrailPosition
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import (
    AssistantMessage,
    Message,
    ReasoningBlock,
    ToolCall,
    ToolMessage,
    balance_tool_messages,
)
from fastaiagent.llm.stream import (
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from fastaiagent.tool.base import Tool, ToolResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail

#: What the model is told about a tool call the agent stopped short of running.
#: Middleware ``StopAgent`` ends a turn *between* the assistant message that
#: declared the calls and the results that answer them; the calls it skipped are
#: never dispatched, so the honest record is that they produced nothing.
STOPPED_NOTE = "the agent stopped before this tool ran, so it was not run"


class _AgentInterrupted(Exception):  # noqa: N818  (internal sentinel, mirrors InterruptSignal)
    """Internal sentinel — raised after the executor persists an interrupted
    agent checkpoint, so :meth:`Agent._arun_core` can return a paused
    :class:`AgentResult` without itself knowing the suspension shape.
    """

    def __init__(
        self, *, reason: str, context: dict[str, Any], node_id: str, agent_path: str | None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.context = context
        self.node_id = node_id
        self.agent_path = agent_path


def _is_multimodal_part(obj: Any) -> bool:
    """True iff ``obj`` is a multimodal part (Image / PDF) carrying raw bytes
    that ``Message.model_dump(mode="json")`` cannot natively serialize.

    The check is import-light because this function runs on every
    checkpoint write — only the modules' top-level classes are touched.
    """
    from fastaiagent.multimodal.file import File
    from fastaiagent.multimodal.image import Image
    from fastaiagent.multimodal.pdf import PDF

    return isinstance(obj, (Image, PDF, File))


def _serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Serialize messages to JSON-safe dicts.

    Multimodal parts (``Image``, ``PDF``) carry raw bytes that Pydantic's
    ``model_dump(mode="json")`` cannot encode — it raises ``UnicodeDecodeError``
    when the content list contains an Image. We pre-process by replacing
    each multimodal part with its ``to_dict()`` form (base64-encoded bytes
    in a typed envelope); the matching :func:`_deserialize_messages`
    rebuilds the dataclass instances on resume.

    Plain text-only messages take the same fast path as before.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m.content, list) and any(_is_multimodal_part(p) for p in m.content):
            # Convert Image / PDF parts into their dict form before model_dump.
            normalized_parts: list[Any] = [
                p.to_dict() if _is_multimodal_part(p) else p for p in m.content
            ]
            m_copy = m.model_copy(update={"content": normalized_parts})
            dumped = m_copy.model_dump(mode="json")
        else:
            dumped = m.model_dump(mode="json")
        # A call's provider signature is saved only when there is one, so a
        # checkpoint from every provider that issues none — and checkpoints
        # replicate to the plane — keeps the exact shape it had before 1.91.0.
        for tc in dumped.get("tool_calls") or []:
            if tc.get("signature") is None:
                tc.pop("signature", None)
        # Reasoning blocks (1.91.0) likewise: saved only when the model issued some.
        if dumped.get("reasoning_blocks") is None:
            dumped.pop("reasoning_blocks", None)
        out.append(dumped)
    return out


def _deserialize_messages(raw: list[dict[str, Any]]) -> list[Message]:
    """Inverse of :func:`_serialize_messages`: rebuild Image / PDF instances
    from their dict form before ``Message.model_validate``. Used by
    ``Agent.aresume`` so a resumed run sees the original multimodal
    content rather than dict stand-ins.
    """
    from fastaiagent.multimodal.file import File
    from fastaiagent.multimodal.image import Image
    from fastaiagent.multimodal.pdf import PDF

    out: list[Message] = []
    for raw_m in raw:
        content = raw_m.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") in ("image", "pdf", "file") for p in content
        ):
            rebuilt: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    rebuilt.append(Image.from_dict(part))
                elif isinstance(part, dict) and part.get("type") == "pdf":
                    rebuilt.append(PDF.from_dict(part))
                elif isinstance(part, dict) and part.get("type") == "file":
                    rebuilt.append(File.from_dict(part))
                else:
                    rebuilt.append(part)
            new_raw = {**raw_m, "content": rebuilt}
            out.append(Message.model_validate(new_raw))
        else:
            out.append(Message.model_validate(raw_m))
    return out


def _record_agent_interrupt(
    *,
    checkpointer: Checkpointer,
    execution_id: str,
    agent_name: str,
    iteration: int,
    tool_name: str,
    tool_call_id: str,
    tool_args: dict[str, Any],
    messages: list[Message],
    sig: InterruptSignal,
    middleware_state: dict[str, Any] | None = None,
) -> _AgentInterrupted:
    """Persist the interrupted-agent checkpoint + pending row in one txn.

    ``middleware_state`` is the state as of the paused call's pre-tool
    checkpoint, not as of now: the resume re-runs that call through
    ``wrap_tool``, so a counter it already bumped would be counted twice.

    Returns the sentinel exception the caller should raise so
    :class:`Agent` can convert it into a paused :class:`AgentResult`.
    """
    base_path = _agent_path.get() or f"agent:{agent_name}"
    full_path = f"{base_path}/tool:{tool_name}"
    node_id = f"turn:{iteration}/tool:{tool_name}"

    interrupted_ckpt = Checkpoint(
        checkpoint_id=str(uuid.uuid4()),
        chain_name=agent_name,
        execution_id=execution_id,
        node_id=node_id,
        node_index=iteration,
        step_type="hitl_pause",
        status="interrupted",
        state_snapshot=_with_middleware_state(
            {
                "messages": _serialize_messages(messages),
                "turn": iteration,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
            },
            middleware_state,
        ),
        node_input=dict(tool_args),
        interrupt_reason=sig.reason,
        interrupt_context=sig.context,
        agent_path=full_path,
    )
    pending = PendingInterrupt(
        execution_id=execution_id,
        chain_name=agent_name,
        node_id=node_id,
        reason=sig.reason,
        context=sig.context,
        agent_path=full_path,
    )
    checkpointer.record_interrupt(interrupted_ckpt, pending)
    # Best-effort: report the pause to a connected plane (no-op when not
    # connected; never blocks/raises into the agent hot path).
    try:
        from fastaiagent.governance import hitl_kind
        from fastaiagent.trace.hitl_export import record_pause_event

        record_pause_event(
            run_id=execution_id,
            node=node_id,
            reason=sig.reason,
            agent_id=agent_name,
            kind=hitl_kind(sig.reason),
        )
    except Exception:
        logger.debug("HITL pause emit failed", exc_info=True)
    return _AgentInterrupted(
        reason=sig.reason,
        context=sig.context,
        node_id=node_id,
        agent_path=full_path,
    )


def _with_middleware_state(
    snapshot: dict[str, Any], middleware_state: dict[str, Any] | None
) -> dict[str, Any]:
    """Add ``middleware_state`` (1.91.0, 1B) only when there is some, so a
    checkpoint from an agent whose middleware keeps none — and checkpoints
    replicate to the plane — has exactly the shape it had before."""
    if middleware_state:
        snapshot["middleware_state"] = middleware_state
    return snapshot


def _mw_state(mw_ctx: MiddlewareContext | None) -> dict[str, Any] | None:
    return mw_ctx.saved_state() if mw_ctx is not None else None


def _put_turn_checkpoint(
    *,
    checkpointer: Checkpointer,
    execution_id: str,
    agent_name: str,
    iteration: int,
    messages: list[Message],
    middleware_state: dict[str, Any] | None = None,
) -> None:
    """Pre-LLM turn-boundary checkpoint — the resume point for crash recovery."""
    base_path = _agent_path.get() or f"agent:{agent_name}"
    checkpointer.put(
        Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            chain_name=agent_name,
            execution_id=execution_id,
            node_id=f"turn:{iteration}",
            node_index=iteration,
            step_type="llm_call",
            status="completed",
            state_snapshot=_with_middleware_state(
                {
                    "messages": _serialize_messages(messages),
                    "turn": iteration,
                },
                middleware_state,
            ),
            agent_path=base_path,
        )
    )


def _checkpoint_exhausted_loop(
    checkpointer: Checkpointer | None,
    execution_id: str | None,
    agent_name: str,
    max_iterations: int,
    messages: list[Message],
    middleware_state: dict[str, Any] | None = None,
) -> None:
    """A turn row past the last turn, written as the loop runs out of iterations.

    The last turn's tools have run and their results are in ``messages``, but the
    newest row is still that turn's pre-tool checkpoint — so crash recovery
    re-invoked a tool that had already finished: one charge became two (1.91.0,
    X12, both loops). A turn row here makes recovery re-ask the model instead;
    resumed with the same ``max_iterations`` it stops again at once.
    """
    if checkpointer is None or execution_id is None:
        return
    _put_turn_checkpoint(
        checkpointer=checkpointer,
        execution_id=execution_id,
        agent_name=agent_name,
        iteration=max_iterations,
        messages=messages,
        middleware_state=middleware_state,
    )


def _put_tool_checkpoint(
    *,
    checkpointer: Checkpointer,
    execution_id: str,
    agent_name: str,
    iteration: int,
    tool_name: str,
    tool_call_id: str,
    tool_args: dict[str, Any],
    messages: list[Message],
    middleware_state: dict[str, Any] | None = None,
) -> None:
    """Pre-tool checkpoint — captures tool args at the moment of dispatch."""
    base_path = _agent_path.get() or f"agent:{agent_name}"
    checkpointer.put(
        Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            chain_name=agent_name,
            execution_id=execution_id,
            node_id=f"turn:{iteration}/tool:{tool_name}",
            node_index=iteration,
            step_type="tool_call",
            status="completed",
            state_snapshot=_with_middleware_state(
                {
                    "messages": _serialize_messages(messages),
                    "turn": iteration,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                },
                middleware_state,
            ),
            node_input=dict(tool_args),
            agent_path=f"{base_path}/tool:{tool_name}",
        )
    )


class _StubTool(Tool):
    """Placeholder Tool used when the LLM requests a tool that is not registered.

    Passed into ``wrap_tool`` middleware so that middleware always sees a Tool
    object. The ``aexecute`` path is never reached — the terminal closure
    handles the unknown-tool branch directly.
    """

    def __init__(self, name: str):
        super().__init__(name=name, description="(unknown tool)")

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        return ToolResult(error=f"Unknown tool '{self.name}'")


def _coerce_tool_output_to_message_content(
    output: Any,
) -> tuple[str | list[Any], str]:
    """Map a tool's return value to ``(message_content, summary_text)``.

    ``message_content`` becomes the next ``ToolMessage.content`` — for
    multimodal returns it's a ``list[ContentPart]`` so images flow back to
    the LLM verbatim. ``summary_text`` is always a string and is used for
    tracing, guardrails, and memory.
    """
    from fastaiagent.multimodal.image import Image as MMImage
    from fastaiagent.multimodal.pdf import PDF as MMPDF

    if isinstance(output, MMImage):
        summary = (
            f"[tool returned image: media_type={output.media_type}, "
            f"size_bytes={output.size_bytes()}]"
        )
        return [summary, output], summary
    if isinstance(output, MMPDF):
        try:
            page_count = output.page_count()
        except Exception:
            page_count = -1
        summary = f"[tool returned pdf: size_bytes={output.size_bytes()}, pages={page_count}]"
        return [summary, output], summary
    if isinstance(output, list) and any(isinstance(p, (MMImage, MMPDF)) for p in output):
        summary = "[tool returned multimodal content with " + str(len(output)) + " parts]"
        return list(output), summary
    if isinstance(output, str):
        return output, output
    from pydantic import BaseModel

    if isinstance(output, BaseModel):
        # Covers tools that return a Pydantic model (e.g. via ``output_type``
        # validation) — emit clean JSON rather than a Python repr.
        text = output.model_dump_json()
        return text, text
    text = json.dumps(output, default=str)
    return text, text


def _rewritten_tool_arguments(
    original: dict[str, Any], outcome: GuardrailOutcome
) -> dict[str, Any]:
    """Take a masked/overridden tool call back apart into its arguments.

    Tool-call guardrails judge a ``{"tool", "arguments"}`` JSON envelope, so a
    rewrite comes back as a rewritten envelope. Anything that no longer parses
    as that envelope cannot be handed to the tool — an ``override`` replaces the
    whole payload with prose, for instance — so it degrades to a block rather
    than invoking the tool with arguments nobody vetted.

    (The plane refuses ``mask`` on tool rules for a related reason: its hosted
    MCP gate acts on the verdict while passing the caller's original arguments
    through. A locally-defined guardrail can still ask for one, so handle it.)
    """
    from fastaiagent._internal.errors import GuardrailBlockedError

    try:
        envelope = json.loads(str(outcome.data))
        rewritten = envelope["arguments"]
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        rewritten = None

    if not isinstance(rewritten, dict):
        outcome.refuse("the rewritten tool call is not an arguments object")
        raise GuardrailBlockedError(
            guardrail_name="tool_call",
            message=(
                "A guardrail rewrote this tool call, but the result is not a usable "
                "arguments object; blocked instead."
            ),
            results=outcome.results,
        )
    return rewritten


def _rewritten_tool_result(
    message_content: str | list[Any], outcome: GuardrailOutcome
) -> tuple[str | list[Any], str]:
    """Apply a masked/overridden tool result to both values the caller carries.

    ``_coerce_tool_output_to_message_content`` returns a pair: the content that
    goes into the next ``ToolMessage``, and the text form used for traces and
    guardrails. Only the text form was judged. When the two are the same string
    the rewrite covers both; when the message content is a list of parts (a tool
    that returned an Image or PDF) the model would still see the unmasked parts,
    so the rewrite cannot be applied and the call blocks.
    """
    from fastaiagent._internal.errors import GuardrailBlockedError

    rewritten = str(outcome.data)
    if isinstance(message_content, str):
        return rewritten, rewritten

    outcome.refuse("a tool result with image/PDF parts cannot be rewritten")
    raise GuardrailBlockedError(
        guardrail_name="tool_result",
        message=(
            "A guardrail rewrote this tool result, but it carries image/PDF content "
            "parts the rewrite cannot cover; blocked instead."
        ),
        results=outcome.results,
    )


async def _invoke_tool_with_span(
    tool: Tool | None,
    tool_name: str,
    arguments: dict[str, Any],
    context: Any | None,
    guardrails: list[Guardrail] | None,
    tool_call_record: dict[str, Any] | None = None,
    *,
    agent_name: str | None = None,
) -> tuple[str | list[Any], str]:
    """Execute a single tool call inside an OTel span.

    Returns a ``(message_content, summary_text)`` pair. ``message_content``
    is what goes into the next ``ToolMessage`` — a string for plain returns
    and a ``list[ContentPart]`` when a tool returns ``Image``/``PDF``.
    ``summary_text`` is the string form used for traces, guardrails, and
    the ``tool_call_record`` so existing dashboards keep working unchanged.

    Always creates a span (even for unknown tools) so dashboards see the
    attempt. ``tool.name`` and ``tool.status`` are captured unconditionally;
    ``tool.args`` / ``tool.result`` are gated by ``trace_payloads_enabled()``.
    """
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import trace_payloads_enabled

    tracer = get_tracer("fastaiagent.agent.executor")
    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        # Origin lets the UI group "function / kb / mcp / rest / custom"
        # without having to cross-reference agent.tools. "unknown" when the
        # LLM hallucinates a tool that isn't registered.
        span.set_attribute("tool.origin", getattr(tool, "origin", "unknown") if tool else "unknown")
        # Classify this as a tool span for the central Replay engine, which reads
        # attributes["fastaiagent.runner.type"] (∈ {tool, tool_call, worker_call})
        # to decide inject-vs-execute. Set explicitly rather than relying on the
        # normalize.py inference, which is off by default and is NOT run on the
        # connect() push path. Both attributes are unconditional (not payload-
        # gated) and set before the ``tool is None`` branch, so even an unknown/
        # hallucinated tool's span is classified and carries the safe default.
        span.set_attribute("fastaiagent.runner.type", "tool")
        span.set_attribute(
            "fastaiagent.tool.replay_class",
            getattr(tool, "replay_class", "side_effecting") if tool else "side_effecting",
        )
        if trace_payloads_enabled():
            try:
                span.set_attribute("tool.args", json.dumps(arguments, default=str))
            except Exception:
                logger.debug("Failed to serialize tool arguments for trace", exc_info=True)

        if tool is None:
            result_text = f"Error: Unknown tool '{tool_name}'"
            span.set_attribute("tool.status", "unknown")
            if tool_call_record is not None:
                tool_call_record["error"] = result_text
            return result_text, result_text

        try:
            # Tool-call guardrail: validate arguments before execution
            if guardrails:
                tc_data = json.dumps({"tool": tool_name, "arguments": arguments}, default=str)
                tc_outcome = await execute_guardrails(
                    guardrails, tc_data, GuardrailPosition.tool_call, agent_name=agent_name
                )
                if tc_outcome.modified:
                    arguments = _rewritten_tool_arguments(arguments, tc_outcome)

            result = await tool.ainvoke(arguments, context=context)
            if result.success:
                message_content, result_text = _coerce_tool_output_to_message_content(result.output)
                # Tool-result guardrail: validate output after execution
                if guardrails:
                    tr_outcome = await execute_guardrails(
                        guardrails,
                        result_text,
                        GuardrailPosition.tool_result,
                        agent_name=agent_name,
                    )
                    if tr_outcome.modified:
                        message_content, result_text = _rewritten_tool_result(
                            message_content, tr_outcome
                        )
                span.set_attribute("tool.status", "ok")
            else:
                result_text = f"Error: {result.error}"
                message_content = result_text
                span.set_attribute("tool.status", "error")
                span.set_attribute("tool.error", str(result.error))

            if trace_payloads_enabled():
                span.set_attribute("tool.result", result_text)
            if tool_call_record is not None:
                tool_call_record["output"] = result_text
            return message_content, result_text
        except ToolExecutionError as e:
            result_text = f"Error: {e}"
            span.set_attribute("tool.status", "error")
            span.set_attribute("tool.error", str(e))
            if tool_call_record is not None:
                tool_call_record["error"] = str(e)
            return result_text, result_text


class _StoppedResponse(LLMResponse):
    """The loop's answer when middleware raised ``StopAgent`` (1.90.0, F8).

    The content is the stop message, as before; ``stop_reason`` is what used to
    be lost, and what tells ``Agent`` not to re-drive a run its middleware ended.
    """

    stop_reason: str = "stopped"


def _stopped_response(stop: StopAgent) -> _StoppedResponse:
    return _StoppedResponse(
        content=str(stop), finish_reason="stop", stop_reason=stop.reason or "stopped"
    )


#: ``(agent_name, reason)`` pairs already warned about — once each per process.
_PARALLEL_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _parallel_ok(
    parallel_tools: bool,
    n_calls: int,
    *,
    checkpointer: Any,
    mw_pipeline: Any,
    agent_id: str | None,
    agent_name: str,
) -> bool:
    """Whether this turn's tool calls may run concurrently — and say so when not.

    ``AgentConfig(parallel_tools=True)`` runs sequentially whenever a
    checkpointer, middleware or a managed ``agent_id`` is in play (their
    ordering and pause semantics are per call). Until 1.90.0 that fallback was
    silent — and a Swarm always injects middleware, so its agents never ran in
    parallel without anyone knowing (F10). One warning per agent per reason.
    """
    if not parallel_tools or n_calls <= 1:
        return False
    reasons = [
        name
        for name, present in (
            ("middleware", mw_pipeline is not None),
            ("checkpointer", checkpointer is not None),
            ("agent_id", bool(agent_id)),
        )
        if present
    ]
    for reason in reasons:
        if (agent_name, reason) not in _PARALLEL_FALLBACK_WARNED:
            _PARALLEL_FALLBACK_WARNED.add((agent_name, reason))
            logger.warning(
                "Agent %r: parallel_tools=True is running tool calls sequentially "
                "because of its %s (parallel execution is not yet supported with "
                "middleware, a checkpointer or managed governance).",
                agent_name,
                reason,
            )
    return not reasons


def _stream_stop(stop: StopAgent, final_reply: dict[str, str] | None) -> TextDelta:
    """End a streamed run on ``StopAgent`` the way ``arun`` ends one (1.90.0, F8).

    ``arun`` returns the stop message as the output; the stream used to end with
    no text at all, so a streaming caller never saw why it stopped. The message
    becomes the reply (and the final text), and the reason goes to the caller.
    """
    if final_reply is not None:
        final_reply["text"] = str(stop)
        final_reply["stop_reason"] = stop.reason or "stopped"
    return TextDelta(text=str(stop))


def _turn_llm_kwargs(kwargs: dict[str, Any], tool_choice: str, iteration: int) -> dict[str, Any]:
    """This turn's LLM kwargs with ``AgentConfig.tool_choice`` applied (1.90.0, F6).

    ``"required"`` forces only the run's first turn: a model that must call a
    tool on every turn can never answer, and the loop would end at
    ``max_iterations``. ``"none"`` applies to every turn. ``"auto"`` adds nothing.
    A ``tool_choice`` the caller passed to ``arun`` themselves is left alone.
    """
    if "tool_choice" in kwargs or tool_choice in (None, "auto"):
        return kwargs
    if tool_choice == "required" and iteration != 0:
        return kwargs
    return {**kwargs, "tool_choice": tool_choice}


def _tool_defs(tools: list[Tool]) -> list[dict[str, Any]] | None:
    return [t.to_openai_format() for t in tools] if tools else None


def _begin_turn(llm: Any, tools: list[Tool], mw_ctx: MiddlewareContext | None) -> None:
    """Reset ``ctx.llm`` / ``ctx.tools`` to the agent's for a new turn, so a
    middleware that changes them changes ONE call (1.91.0, 1A)."""
    if mw_ctx is not None:
        mw_ctx.llm = llm
        mw_ctx.tools = list(tools)


def _turn_model(llm: Any, mw_ctx: MiddlewareContext | None) -> Any:
    """The client this call uses: ``ctx.llm`` when middleware set one."""
    if mw_ctx is not None and mw_ctx.llm is not None:
        return mw_ctx.llm
    return llm


def _wraps_model(mw_pipeline: _MiddlewarePipeline | None, mw_ctx: MiddlewareContext | None) -> bool:
    return bool(mw_pipeline) and mw_ctx is not None and mw_pipeline.overrides("wrap_model")  # type: ignore[union-attr]


async def _complete_turn(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    tool_defs: list[dict[str, Any]] | None,
    mw_pipeline: _MiddlewarePipeline | None,
    mw_ctx: MiddlewareContext | None,
    turn_kwargs: dict[str, Any],
) -> LLMResponse:
    """One non-streamed model call of the tool loop, through ``wrap_model``
    when any middleware implements it. The terminal reads ``ctx.llm`` and
    ``ctx.tools`` when it runs, so a ``wrap_model`` that changes them is heard."""
    if not _wraps_model(mw_pipeline, mw_ctx):
        offered = tool_defs if mw_ctx is None else _tool_defs(mw_ctx.tools)
        plain: LLMResponse = await _turn_model(llm, mw_ctx).acomplete(
            messages, tools=offered, **turn_kwargs
        )
        return plain
    assert mw_pipeline is not None and mw_ctx is not None

    async def _terminal(msgs: list[Message]) -> LLMResponse:
        wrapped: LLMResponse = await _turn_model(llm, mw_ctx).acomplete(
            msgs, tools=_tool_defs(mw_ctx.tools), **turn_kwargs
        )
        return wrapped

    return await mw_pipeline.call_model(mw_ctx, messages, _terminal)


async def _streamed_turn(
    llm: Any,
    messages: list[Message],
    mw_pipeline: _MiddlewarePipeline,
    mw_ctx: MiddlewareContext,
    turn_kwargs: dict[str, Any],
    out: dict[str, LLMResponse],
) -> AsyncGenerator[StreamEvent, None]:
    """One streamed model call through the ``wrap_model`` onion (1.91.0, 1A).

    The onion is a coroutine and the stream is a generator, so the call runs as
    a task whose terminal pushes the model's events onto a queue this generator
    drains — deltas reach the caller as they arrive, not after the onion
    returns. The final response (what the onion returned) is left in
    ``out["response"]``. A response produced without calling the model is
    streamed as one ``TextDelta`` (plus its tool-call events); a retry after
    deltas streams the second attempt too, as ``max_retries`` does.
    """
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    attempts = 0

    async def _terminal(msgs: list[Message]) -> LLMResponse:
        nonlocal attempts
        attempts += 1
        text = ""
        reasoning = ""
        blocks: list[ReasoningBlock] = []
        calls: list[ToolCall] = []
        prompt_tokens = completion_tokens = 0
        finish = ""
        async for event in _turn_model(llm, mw_ctx).astream(
            msgs, tools=_tool_defs(mw_ctx.tools), **turn_kwargs
        ):
            if isinstance(event, StreamDone):
                finish = event.finish_reason or finish
                continue  # the loop emits one StreamDone for the whole run
            if isinstance(event, TextDelta):
                text += event.text
            elif isinstance(event, ReasoningDelta):
                reasoning += event.text
                if event.block is not None:
                    blocks.append(event.block)
            elif isinstance(event, ToolCallEnd):
                calls.append(
                    ToolCall(
                        id=event.call_id,
                        name=event.tool_name,
                        arguments=event.arguments,
                        signature=event.signature,
                    )
                )
            elif isinstance(event, Usage):
                prompt_tokens += event.prompt_tokens
                completion_tokens += event.completion_tokens
            await queue.put(event)
        return LLMResponse(
            content=text or None,
            tool_calls=calls,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="tool_calls" if calls else (finish or "stop"),
            reasoning=reasoning or None,
            reasoning_blocks=blocks,
        )

    task = asyncio.ensure_future(mw_pipeline.call_model(mw_ctx, messages, _terminal))
    try:
        while True:
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
                continue
            getter.cancel()
            while not queue.empty():
                yield queue.get_nowait()
            break
        response = task.result()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - we are unwinding already
                pass
    if attempts == 0:
        # Answered without the model: the caller still has to see the answer.
        if response.content:
            yield TextDelta(text=response.content)
        for tc in response.tool_calls:
            yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
            yield ToolCallEnd(
                call_id=tc.id,
                tool_name=tc.name,
                arguments=dict(tc.arguments),
                signature=tc.signature,
            )
    out["response"] = response


#: How :func:`_governed_tool_call` asks managed governance about a call.
#:
#: * ``"gate"`` — :func:`governance.gate_tool_call`: ``deny`` refuses,
#:   ``require_approval`` calls ``interrupt()`` and pauses the run. The agent
#:   loops, and crash recovery in ``Agent.aresume`` (the pre-tool checkpoint is
#:   written *before* the gate, so a recovered call may never have been asked).
#: * ``"direct"`` — :func:`governance.check_direct_call`: there is no run to
#:   pause, so ``require_approval`` refuses. The MCP ``expose_tools`` surface.
#: * ``"resolved"`` — skip: the pause this call is resuming from *was* the
#:   decision. ``Agent.aresume`` with a ``Resume`` value; a rejected approval is
#:   refused before this is reached, so a rejected call is never executed.
GovernanceMode = Literal["gate", "direct", "resolved"]


@dataclass
class _GovernedCall:
    """What :func:`_governed_tool_call` produced for one call."""

    #: The next ``ToolMessage``'s content: a string, or content parts for a tool
    #: that returned an ``Image``/``PDF`` (the plain path keeps them).
    content: str | list[Any]
    #: Set when governance refused the call; ``content`` is then the refusal and
    #: the tool never ran.
    refusal: str | None = None


async def _governed_tool_call(
    tool: Tool | None,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    governance_mode: GovernanceMode,
    agent_id: str | None,
    run_id: str,
    context: Any | None,
    guardrails: list[Guardrail] | None,
    mw_pipeline: _MiddlewarePipeline | None = None,
    mw_ctx: MiddlewareContext | None = None,
    tool_call_index: int = 0,
    record: dict[str, Any] | None = None,
    agent_name: str | None = None,
) -> _GovernedCall:
    """Run one tool call the way the agent loop does — the one place that does.

    Order: managed governance → ``wrap_tool`` middleware → ``tool_call``
    guardrails → ``Tool.ainvoke`` (timeout, retries, ``output_type``) →
    ``tool_result`` guardrails, all inside the ``tool.<name>`` span.

    Before 1.90.0 the agent loops each carried a copy of this, and two callers
    skipped it: ``Agent.aresume`` re-ran a paused tool with a bare
    ``tool.aexecute`` (no guardrails, middleware, span, timeout or retries — and
    crash recovery asked no governance at all), and the MCP ``expose_tools``
    surface did the same after its governance check.

    Control flow propagates for the caller to handle: ``InterruptSignal`` (a
    pause), ``StopAgent`` (from middleware) and ``GuardrailBlockedError``.
    """
    if tool is not None and governance_mode != "resolved":
        from fastaiagent import governance

        if governance_mode == "direct":
            refusal = await governance.check_direct_call(tool_name, dict(arguments), agent_id)
        else:
            refusal = await governance.gate_tool_call(
                tool_name, dict(arguments), agent_id or "", run_id
            )
        if refusal is not None:
            if record is not None:
                record["error"] = refusal
            return _GovernedCall(content=refusal, refusal=refusal)

    if mw_pipeline and mw_ctx is not None:
        mw_ctx.tool_call_index = tool_call_index

        async def _terminal(t: Tool, a: dict[str, Any]) -> ToolResult:
            # Middleware path: multimodal tool returns flatten to the string
            # summary because ``ToolResult.output`` is text-only today. The plain
            # path below preserves ContentParts.
            _, summary_text = await _invoke_tool_with_span(
                tool=tool,
                tool_name=tool_name,
                arguments=a,
                context=context,
                guardrails=guardrails,
                tool_call_record=record,
                agent_name=agent_name,
            )
            return ToolResult(output=summary_text)

        wrap_target = tool if tool is not None else _StubTool(tool_name)
        tr = await mw_pipeline.invoke_tool(mw_ctx, wrap_target, dict(arguments), _terminal)
        if isinstance(tr.output, str):
            return _GovernedCall(content=tr.output)
        if tr.error is not None:
            return _GovernedCall(content=f"Error: {tr.error}")
        return _GovernedCall(content=json.dumps(tr.output, default=str))

    content, _ = await _invoke_tool_with_span(
        tool=tool,
        tool_name=tool_name,
        arguments=arguments,
        context=context,
        guardrails=guardrails,
        tool_call_record=record,
        agent_name=agent_name,
    )
    return _GovernedCall(content=content)


async def execute_tool_loop(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    max_iterations: int = 10,
    tool_choice: str = "auto",
    tracer: Any = None,
    context: Any | None = None,
    guardrails: list[Guardrail] | None = None,
    mw_pipeline: _MiddlewarePipeline | None = None,
    mw_ctx: MiddlewareContext | None = None,
    *,
    checkpointer: Checkpointer | None = None,
    execution_id: str | None = None,
    agent_name: str = "",
    agent_id: str | None = None,
    start_iteration: int = 0,
    parallel_tools: bool = False,
    max_parallel_tools: int = 4,
    governance_run_id: str | None = None,
    **kwargs: Any,
) -> tuple[LLMResponse, list[dict[str, Any]]]:
    """Execute the agent's tool-calling loop.

    Sends messages to the LLM. If the LLM requests tool calls,
    executes them, appends results, and loops. Stops when the LLM
    returns a final response (no tool calls) or max_iterations is reached.

    When ``mw_pipeline`` and ``mw_ctx`` are provided, middleware hooks fire:
    ``before_model`` before each LLM call, ``after_model`` after each LLM
    response, and ``wrap_tool`` around each tool invocation.

    When ``checkpointer`` and ``execution_id`` are provided, the loop writes
    a turn-boundary checkpoint before each LLM call and a pre-tool
    checkpoint before each tool runs. ``InterruptSignal`` raised from inside
    any tool is caught: the loop persists an ``status="interrupted"``
    checkpoint plus a row in ``pending_interrupts`` (atomically), then
    raises :class:`_AgentInterrupted` so the caller can return a paused
    :class:`AgentResult`.

    Returns:
        Tuple of (final LLM response, list of all tool call records)
    """
    tool_defs = [t.to_openai_format() for t in tools] if tools else None
    tools_by_name = {t.name: t for t in tools}
    all_tool_calls: list[dict[str, Any]] = []

    async def _run_plain_tool_call(
        idx: int, tc: ToolCall, iteration: int
    ) -> tuple[int, Message, dict[str, Any]]:
        """Execute one tool call via the plain (no checkpoint/middleware/
        governance) path and return its ordered index, message, and record.

        Used only by the parallel branch, which is gated on exactly those
        features being absent — so this mirrors the sequential ``else`` path
        below without the control-flow machinery.
        """
        record: dict[str, Any] = {
            "iteration": iteration,
            "tool_name": tc.name,
            "arguments": tc.arguments,
            "tool_call_id": tc.id,
        }
        content, _ = await _invoke_tool_with_span(
            tool=tools_by_name.get(tc.name),
            tool_name=tc.name,
            arguments=tc.arguments,
            context=context,
            guardrails=guardrails,
            tool_call_record=record,
            agent_name=agent_name,
        )
        return idx, ToolMessage(content=content, tool_call_id=tc.id), record

    for iteration in range(start_iteration, max_iterations):
        # Turn-boundary checkpoint — the resume point for crashes mid-LLM.
        if checkpointer is not None and execution_id is not None:
            _put_turn_checkpoint(
                checkpointer=checkpointer,
                execution_id=execution_id,
                agent_name=agent_name,
                iteration=iteration,
                messages=messages,
                middleware_state=_mw_state(mw_ctx),
            )
        _begin_turn(llm, tools, mw_ctx)
        # Middleware: before_model (may raise StopAgent)
        if mw_pipeline and mw_ctx is not None:
            mw_ctx.turn = iteration
            try:
                kept = await mw_pipeline.apply_before_model(mw_ctx, messages)
                if kept is not messages:
                    # Keep the ONE list everyone holds. ``Agent`` re-sends this
                    # object for the structured and guardrail re-asks; a hook
                    # that returned a new list (``TrimLongMessages`` did)
                    # rebound only the loop's name, so a re-ask after a trim
                    # re-sent the untrimmed history minus its latest turns
                    # (1.90.0, F11b). ``RedactPII`` already writes in place.
                    messages[:] = kept
            except StopAgent as stop:
                return _stopped_response(stop), all_tool_calls

        # Call LLM — through ``wrap_model`` when a middleware implements it.
        try:
            response = await _complete_turn(
                llm,
                messages,
                tools,
                tool_defs,
                mw_pipeline,
                mw_ctx,
                _turn_llm_kwargs(kwargs, tool_choice, iteration),
            )
        except StopAgent as stop:
            return _stopped_response(stop), all_tool_calls
        if mw_ctx is not None:
            # This turn's tools are what was offered: one a middleware added
            # for this call can be dispatched (1.91.0, 1A).
            tools_by_name = {t.name: t for t in mw_ctx.tools}

        # Middleware: after_model (may raise StopAgent)
        if mw_pipeline and mw_ctx is not None:
            try:
                response = await mw_pipeline.apply_after_model(mw_ctx, response)
            except StopAgent as stop:
                return _stopped_response(stop), all_tool_calls

        # No tool calls — we're done
        if not response.tool_calls:
            return response, all_tool_calls

        # Build assistant message with tool calls — and the model's reasoning
        # blocks, which Claude requires back within the tool-use turn (X17).
        messages.append(
            AssistantMessage(
                content=response.content,
                tool_calls=response.tool_calls,
                reasoning_blocks=response.reasoning_blocks,
            )
        )

        # Execute the turn's tool calls concurrently when it's provably safe —
        # no per-tool checkpointing, middleware, or managed governance in play
        # (those are order/identity sensitive and stay sequential). Results are
        # re-ordered by index so message history stays deterministic.
        if _parallel_ok(
            parallel_tools,
            len(response.tool_calls),
            checkpointer=checkpointer,
            mw_pipeline=mw_pipeline,
            agent_id=agent_id,
            agent_name=agent_name,
        ):
            sem = asyncio.Semaphore(max_parallel_tools)

            async def _guarded(
                idx: int, tc: ToolCall, _it: int = iteration
            ) -> tuple[int, Message, dict[str, Any]]:
                async with sem:
                    return await _run_plain_tool_call(idx, tc, _it)

            results = await asyncio.gather(
                *[_guarded(i, tc) for i, tc in enumerate(response.tool_calls)]
            )
            for _, tool_msg, record in sorted(results, key=lambda r: r[0]):
                messages.append(tool_msg)
                all_tool_calls.append(record)
            continue

        # Execute each tool call
        for idx, tc in enumerate(response.tool_calls):
            tool_call_record = {
                "iteration": iteration,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "tool_call_id": tc.id,
            }

            tool = tools_by_name.get(tc.name)

            # Pre-tool checkpoint — captures tool args at the moment of
            # dispatch. Resume re-enters the tool with these same args.
            pre_tool_state: dict[str, Any] | None = None
            if checkpointer is not None and execution_id is not None:
                pre_tool_state = _mw_state(mw_ctx)
                _put_tool_checkpoint(
                    checkpointer=checkpointer,
                    execution_id=execution_id,
                    agent_name=agent_name,
                    iteration=iteration,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    tool_args=dict(tc.arguments),
                    messages=messages,
                    middleware_state=pre_tool_state,
                )

            try:
                # Managed governance (Task C) gates the call against the policy
                # cached on connect(): a ``deny`` is fed back to the model as the
                # tool result; a ``require_approval`` calls interrupt(), which
                # pauses the agent (caught + checkpointed below). The pending run
                # is registered under the run that OWNS the pause
                # (``governance_run_id``, see ``Agent._pause_owner_run_id``).
                try:
                    tool_message_content: str | list[Any] = (
                        await _governed_tool_call(
                            tool,
                            tc.name,
                            tc.arguments,
                            governance_mode="gate",
                            agent_id=agent_id,
                            run_id=governance_run_id or execution_id or "",
                            context=context,
                            guardrails=guardrails,
                            mw_pipeline=mw_pipeline,
                            mw_ctx=mw_ctx,
                            tool_call_index=idx,
                            record=tool_call_record,
                            agent_name=agent_name,
                        )
                    ).content
                except StopAgent as stop:
                    # Include the in-flight tool-call record; the terminal
                    # closure already populated ``output``/``error`` via
                    # ``_invoke_tool_with_span`` before the stopper fired.
                    if tool_call_record not in all_tool_calls:
                        all_tool_calls.append(tool_call_record)
                    # The assistant message at the top of this turn already
                    # declares EVERY call in it. Returning here leaves this
                    # one — and every later sibling — unanswered, which is a
                    # history OpenAI and Anthropic both 400 on. Answer the
                    # in-flight call with whatever the terminal actually
                    # produced (a stopper that fires before ``call_next``,
                    # like ``ToolBudget``, produces nothing), then balance
                    # the siblings. They are NOT dispatched: their side
                    # effects never happen and the synthetic result says so.
                    in_flight = tool_call_record.get("error") or tool_call_record.get("output")
                    if in_flight is not None:
                        messages.append(ToolMessage(content=str(in_flight), tool_call_id=tc.id))
                    # ``messages[:]`` and not a rebind: ``Agent._arun_core``
                    # holds this same list object and is what re-sends it
                    # (structured-output re-ask, ``reask`` guardrail).
                    messages[:] = balance_tool_messages(messages, note=STOPPED_NOTE)
                    return (
                        _stopped_response(stop),
                        all_tool_calls,
                    )
            except InterruptSignal as sig:
                # A tool inside this turn called interrupt(). Persist the
                # suspension and bubble paused state up via _AgentInterrupted.
                if checkpointer is not None and execution_id is not None:
                    raise _record_agent_interrupt(
                        checkpointer=checkpointer,
                        execution_id=execution_id,
                        agent_name=agent_name,
                        iteration=iteration,
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        tool_args=dict(tc.arguments),
                        messages=messages,
                        sig=sig,
                        middleware_state=pre_tool_state,
                    ) from None
                # No checkpointer wired — let the InterruptSignal propagate
                # to whatever owns suspension (typically a parent Chain).
                raise

            messages.append(ToolMessage(content=tool_message_content, tool_call_id=tc.id))
            all_tool_calls.append(tool_call_record)

    _checkpoint_exhausted_loop(
        checkpointer,
        execution_id,
        agent_name,
        max_iterations,
        messages,
        middleware_state=_mw_state(mw_ctx) if checkpointer is not None else None,
    )
    raise MaxIterationsError(
        f"Agent exceeded maximum iterations ({max_iterations}). "
        f"The LLM continued requesting tool calls beyond the limit.\n"
        f"Options:\n"
        f"  1. Increase the limit: AgentConfig(max_iterations={max_iterations * 2})\n"
        f"  2. Review the system prompt to ensure the agent can reach a final answer\n"
        f"  3. Simplify the available tools to reduce unnecessary tool-calling loops"
    )


async def stream_tool_loop(
    llm: Any,
    messages: list[Message],
    tools: list[Tool],
    max_iterations: int = 10,
    tool_choice: str = "auto",
    context: Any | None = None,
    guardrails: list[Guardrail] | None = None,
    mw_pipeline: _MiddlewarePipeline | None = None,
    mw_ctx: MiddlewareContext | None = None,
    *,
    checkpointer: Checkpointer | None = None,
    execution_id: str | None = None,
    agent_name: str = "",
    agent_id: str | None = None,
    start_iteration: int = 0,
    parallel_tools: bool = False,
    max_parallel_tools: int = 4,
    governance_run_id: str | None = None,
    final_reply: dict[str, str] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[StreamEvent, None]:
    """Streaming version of execute_tool_loop.

    Yields StreamEvent objects as tokens arrive from the LLM.
    Handles tool execution between streaming iterations.

    When ``mw_pipeline`` and ``mw_ctx`` are provided, middleware hooks fire:
    ``before_model`` before each LLM stream, ``after_model`` after each
    streamed response is fully accumulated, and ``wrap_tool`` around each
    tool invocation.

    When ``checkpointer`` and ``execution_id`` are provided, the loop writes
    a turn-boundary checkpoint before each LLM stream and a pre-tool
    checkpoint before each tool runs. ``InterruptSignal`` raised from inside
    any tool is caught and handled identically to ``execute_tool_loop``.

    The final TextDelta events contain the agent's response text.
    ToolCallStart/ToolCallEnd events are emitted for both LLM-requested
    tool calls and their execution results.

    ``final_reply``, when given, receives ``{"text": ...}``: the last turn's
    reply after ``after_model`` — what ``execute_tool_loop`` returns as the
    answer. The deltas can't carry it: they include earlier turns' text (said
    before a tool call) and precede any middleware rewrite.
    """
    tool_defs = [t.to_openai_format() for t in tools] if tools else None
    tools_by_name = {t.name: t for t in tools}

    async def _run_plain_tool_call(idx: int, tc: ToolCall) -> tuple[int, Message]:
        """Execute one tool call via the plain path (see the ``execute_tool_loop``
        counterpart). Used only by the parallel branch, which is gated on the
        absence of checkpointer/middleware/governance."""
        content, _ = await _invoke_tool_with_span(
            tool=tools_by_name.get(tc.name),
            tool_name=tc.name,
            arguments=tc.arguments,
            context=context,
            guardrails=guardrails,
            agent_name=agent_name,
        )
        return idx, ToolMessage(content=content, tool_call_id=tc.id)

    for iteration in range(start_iteration, max_iterations):
        # Turn-boundary checkpoint — the resume point for crashes mid-LLM.
        if checkpointer is not None and execution_id is not None:
            _put_turn_checkpoint(
                checkpointer=checkpointer,
                execution_id=execution_id,
                agent_name=agent_name,
                iteration=iteration,
                messages=messages,
                middleware_state=_mw_state(mw_ctx),
            )
        _begin_turn(llm, tools, mw_ctx)

        # Middleware: before_model (may raise StopAgent)
        if mw_pipeline and mw_ctx is not None:
            mw_ctx.turn = iteration
            try:
                kept = await mw_pipeline.apply_before_model(mw_ctx, messages)
                if kept is not messages:
                    # Keep the ONE list everyone holds. ``Agent`` re-sends this
                    # object for the structured and guardrail re-asks; a hook
                    # that returned a new list (``TrimLongMessages`` did)
                    # rebound only the loop's name, so a re-ask after a trim
                    # re-sent the untrimmed history minus its latest turns
                    # (1.90.0, F11b). ``RedactPII`` already writes in place.
                    messages[:] = kept
            except StopAgent as stop:
                yield _stream_stop(stop, final_reply)
                return

        # Stream from LLM
        accumulated_text = ""
        pending_tool_calls: list[ToolCall] = []
        reasoning_blocks: list[ReasoningBlock] = []
        total_usage = Usage()
        turn_kwargs = _turn_llm_kwargs(kwargs, tool_choice, iteration)
        wrapped: LLMResponse | None = None

        if _wraps_model(mw_pipeline, mw_ctx):
            # Through the ``wrap_model`` onion; the deltas still stream live.
            assert mw_pipeline is not None and mw_ctx is not None
            out: dict[str, LLMResponse] = {}
            try:
                async for event in _streamed_turn(
                    llm, messages, mw_pipeline, mw_ctx, turn_kwargs, out
                ):
                    yield event
            except StopAgent as stop:
                yield _stream_stop(stop, final_reply)
                return
            wrapped = out["response"]
            accumulated_text = wrapped.content or ""
            pending_tool_calls = list(wrapped.tool_calls)
            reasoning_blocks = list(wrapped.reasoning_blocks)
        else:
            offered = tool_defs if mw_ctx is None else _tool_defs(mw_ctx.tools)
            async for event in _turn_model(llm, mw_ctx).astream(
                messages, tools=offered, **turn_kwargs
            ):
                if isinstance(event, TextDelta):
                    accumulated_text += event.text
                    yield event
                elif isinstance(event, ReasoningDelta):
                    # Reasoning reaches the caller (1.91.0, B2-5); its blocks go
                    # back with the turn's tool calls (X17).
                    if event.block is not None:
                        reasoning_blocks.append(event.block)
                    yield event
                elif isinstance(event, ToolCallStart):
                    yield event
                elif isinstance(event, ToolCallEnd):
                    pending_tool_calls.append(
                        ToolCall(
                            id=event.call_id,
                            name=event.tool_name,
                            arguments=event.arguments,
                            signature=event.signature,
                        )
                    )
                    yield event
                elif isinstance(event, Usage):
                    total_usage.prompt_tokens += event.prompt_tokens
                    total_usage.completion_tokens += event.completion_tokens
                    yield event
                # Don't yield StreamDone here — we may have more iterations
        if mw_ctx is not None:
            # This turn's tools are what was offered (see execute_tool_loop).
            tools_by_name = {t.name: t for t in mw_ctx.tools}

        # Middleware: after_model — build an LLMResponse from accumulated
        # stream data and let middleware inspect/rewrite it. This runs after
        # streaming completes but before tool execution, so the stream events
        # have already been yielded to the caller.
        if mw_pipeline and mw_ctx is not None:
            response = wrapped or LLMResponse(
                content=accumulated_text or None,
                tool_calls=list(pending_tool_calls),
                # ``total_tokens`` as a non-streamed response carries it — a
                # token-budget middleware reads it (1.91.0, X8).
                usage={
                    "prompt_tokens": total_usage.prompt_tokens,
                    "completion_tokens": total_usage.completion_tokens,
                    "total_tokens": total_usage.prompt_tokens + total_usage.completion_tokens,
                },
                finish_reason="tool_calls" if pending_tool_calls else "stop",
                reasoning_blocks=list(reasoning_blocks),
            )
            try:
                response = await mw_pipeline.apply_after_model(mw_ctx, response)
            except StopAgent as stop:
                yield _stream_stop(stop, final_reply)
                return
            # Reflect any middleware mutations back into the loop state.
            accumulated_text = response.content or ""
            pending_tool_calls = list(response.tool_calls)
            reasoning_blocks = list(response.reasoning_blocks)

        # No tool calls — final response, we're done
        if not pending_tool_calls:
            if final_reply is not None:
                final_reply["text"] = accumulated_text
            return

        # Append assistant message with accumulated content + tool calls
        messages.append(
            AssistantMessage(
                content=accumulated_text or None,
                tool_calls=pending_tool_calls,
                reasoning_blocks=reasoning_blocks,
            )
        )

        # Concurrent execution when it's provably safe (see execute_tool_loop).
        # ToolCallStart/End events were already yielded during streaming, so
        # parallel execution doesn't reorder anything the caller has seen.
        if _parallel_ok(
            parallel_tools,
            len(pending_tool_calls),
            checkpointer=checkpointer,
            mw_pipeline=mw_pipeline,
            agent_id=agent_id,
            agent_name=agent_name,
        ):
            sem = asyncio.Semaphore(max_parallel_tools)

            async def _guarded(idx: int, tc: ToolCall) -> tuple[int, Message]:
                async with sem:
                    return await _run_plain_tool_call(idx, tc)

            results = await asyncio.gather(
                *[_guarded(i, tc) for i, tc in enumerate(pending_tool_calls)]
            )
            for _, tool_msg in sorted(results, key=lambda r: r[0]):
                messages.append(tool_msg)
            continue

        # Execute each tool call
        for idx, tc in enumerate(pending_tool_calls):
            tool = tools_by_name.get(tc.name)

            # Pre-tool checkpoint — captures tool args at the moment of
            # dispatch. Resume re-enters the tool with these same args.
            pre_tool_state: dict[str, Any] | None = None
            if checkpointer is not None and execution_id is not None:
                pre_tool_state = _mw_state(mw_ctx)
                _put_tool_checkpoint(
                    checkpointer=checkpointer,
                    execution_id=execution_id,
                    agent_name=agent_name,
                    iteration=iteration,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    tool_args=dict(tc.arguments),
                    messages=messages,
                    middleware_state=pre_tool_state,
                )

            try:
                # Managed governance (Task C) — gate the tool call (see
                # execute_tool_loop). deny -> refusal to the model; require_approval
                # -> interrupt() pauses; the calling application resolves it.
                try:
                    tool_message_content: str | list[Any] = (
                        await _governed_tool_call(
                            tool,
                            tc.name,
                            tc.arguments,
                            governance_mode="gate",
                            agent_id=agent_id,
                            run_id=governance_run_id or execution_id or "",
                            context=context,
                            guardrails=guardrails,
                            mw_pipeline=mw_pipeline,
                            mw_ctx=mw_ctx,
                            tool_call_index=idx,
                            agent_name=agent_name,
                        )
                    ).content
                except StopAgent as stop:
                    # Streaming twin of the same gap — see the comment on
                    # ``execute_tool_loop``'s handler. The stream path keeps
                    # no per-call record, so the in-flight call is answered
                    # by the same synthetic note as its siblings.
                    messages[:] = balance_tool_messages(messages, note=STOPPED_NOTE)
                    yield _stream_stop(stop, final_reply)
                    return
            except InterruptSignal as sig:
                if checkpointer is not None and execution_id is not None:
                    raise _record_agent_interrupt(
                        checkpointer=checkpointer,
                        execution_id=execution_id,
                        agent_name=agent_name,
                        iteration=iteration,
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        tool_args=dict(tc.arguments),
                        messages=messages,
                        sig=sig,
                        middleware_state=pre_tool_state,
                    ) from None
                raise

            messages.append(ToolMessage(content=tool_message_content, tool_call_id=tc.id))

    _checkpoint_exhausted_loop(
        checkpointer,
        execution_id,
        agent_name,
        max_iterations,
        messages,
        middleware_state=_mw_state(mw_ctx) if checkpointer is not None else None,
    )
    raise MaxIterationsError(
        f"Agent exceeded maximum iterations ({max_iterations}). "
        f"The LLM continued requesting tool calls beyond the limit.\n"
        f"Options:\n"
        f"  1. Increase the limit: AgentConfig(max_iterations={max_iterations * 2})\n"
        f"  2. Review the system prompt to ensure the agent can reach a final answer\n"
        f"  3. Simplify the available tools to reduce unnecessary tool-calling loops"
    )
