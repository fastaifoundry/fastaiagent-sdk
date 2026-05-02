"""Agent class — the central component of the SDK."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.agent.context import RunContext
from fastaiagent.agent.executor import _AgentInterrupted, execute_tool_loop, stream_tool_loop
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.middleware import (
    AgentMiddleware,
    MiddlewareContext,
    _MiddlewarePipeline,
)
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.chain.idempotent import _current_checkpointer
from fastaiagent.chain.interrupt import (
    AlreadyResumed,
    Resume,
    _agent_path,
    _execution_id,
    _resume_value,
)
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition
from fastaiagent.llm.client import LLMClient, _strip_code_fences
from fastaiagent.llm.message import Message, SystemMessage, UserMessage
from fastaiagent.llm.stream import StreamEvent, TextDelta
from fastaiagent.multimodal.image import Image as MultimodalImage
from fastaiagent.multimodal.pdf import PDF as MultimodalPDF  # noqa: N811
from fastaiagent.multimodal.types import ContentPart, normalize_input
from fastaiagent.tool.base import Tool

AgentInput = str | MultimodalImage | MultimodalPDF | list[ContentPart]


def _input_summary_text(parts: list[ContentPart]) -> str:
    """Concatenate the text portions of a multimodal input.

    Used wherever a string is required (memory, guardrails, span attributes)
    but the user passed a multimodal list. Image/PDF parts are replaced
    with a short marker so the trace stays readable.
    """
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, str):
            pieces.append(part)
        elif isinstance(part, MultimodalImage):
            pieces.append(f"[image:{part.media_type}:{part.size_bytes()}b]")
        elif isinstance(part, MultimodalPDF):
            pieces.append(f"[pdf:{part.size_bytes()}b]")
    return " ".join(pieces)


class AgentConfig(BaseModel):
    """Agent execution configuration."""

    max_iterations: int = Field(default=10, ge=1, le=100)
    tool_choice: str = "auto"  # "auto", "required", "none"
    temperature: float | None = None
    max_tokens: int | None = None


class AgentResult(BaseModel):
    """Result of an agent execution.

    ``status`` is ``"completed"`` for a normal run or ``"paused"`` when a
    tool called :func:`interrupt`. In the paused case ``pending_interrupt``
    holds ``{reason, context, node_id, agent_path}`` — the same payload the
    ``/approvals`` UI reads from the ``pending_interrupts`` table.
    ``execution_id`` is always populated when a checkpointer is configured.
    """

    output: str = ""
    parsed: Any | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    trace_id: str | None = None
    execution_id: str = ""
    status: str = "completed"
    pending_interrupt: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}


class Agent:
    """An AI agent with tools, guardrails, and full tracing.

    Example:
        agent = Agent(
            name="support-bot",
            system_prompt="You are a helpful support agent.",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            tools=[search_tool, refund_tool],
            guardrails=[no_pii()],
        )
        result = agent.run("How do I get a refund?")
    """

    def __init__(
        self,
        name: str,
        system_prompt: str | Callable[..., str] = "",
        llm: LLMClient | None = None,
        tools: Sequence[Tool] | None = None,
        guardrails: Sequence[Guardrail] | None = None,
        memory: AgentMemory | ComposableMemory | None = None,
        config: AgentConfig | None = None,
        output_type: type | None = None,
        middleware: Sequence[AgentMiddleware] | None = None,
        checkpointer: Checkpointer | None = None,
        agent_path_label: str | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm or LLMClient()
        self.tools: list[Tool] = list(tools) if tools else []
        self.guardrails: list[Guardrail] = list(guardrails) if guardrails else []
        self.memory = memory
        self.config = config or AgentConfig()
        self.output_type = output_type
        self.middleware: list[AgentMiddleware] = list(middleware) if middleware else []
        self._mw_pipeline = _MiddlewarePipeline(self.middleware)
        self._checkpointer: Checkpointer | None = checkpointer
        # Override the segment this Agent contributes to ``_agent_path``.
        # Default ``"agent:<name>"``; ``Supervisor`` uses ``"supervisor:<name>"``,
        # delegated workers use ``"worker:<role>"``.
        self._agent_path_label: str = agent_path_label or f"agent:{self.name}"

    def _build_response_format(self) -> dict[str, Any] | None:
        """Build response_format dict from output_type for structured output."""
        if self.output_type is None:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_type.__name__,
                "schema": self.output_type.model_json_schema(),
            },
        }

    def _parse_output(self, text: str) -> Any | None:
        """Parse LLM text output into output_type Pydantic model."""
        if self.output_type is None or not text:
            return None
        try:
            clean = _strip_code_fences(text)
            data = json.loads(clean)
            return self.output_type.model_validate(data)
        except Exception:
            return None

    def run(
        self,
        input: AgentInput,
        *,
        context: RunContext | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous execution.

        ``input`` is a string, an :class:`Image`, a :class:`PDF`, or a list
        of those parts. The list form sends multimodal content to the LLM.
        """
        return run_sync(
            self.arun(
                input,
                context=context,
                trace=trace,
                execution_id=execution_id,
                **kwargs,
            )
        )

    async def arun(
        self,
        input: AgentInput,
        *,
        context: RunContext | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Async execution with tool-calling loop.

        ``input`` may be a string or a multimodal list (text + Image + PDF).
        ``execution_id`` (optional) names this run for resume. If omitted,
        a UUID is generated. Pair with a ``checkpointer`` on the Agent to
        get crash- and interrupt-recovery via :meth:`resume`.
        """
        if trace:
            return await self._arun_traced(
                input, context=context, execution_id=execution_id, **kwargs
            )
        return await self._arun_core(input, context=context, execution_id=execution_id, **kwargs)

    async def _arun_traced(
        self,
        input: AgentInput,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute with OTel tracing."""
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import trace_payloads_enabled

        tracer = get_tracer()
        with tracer.start_as_current_span(f"agent.{self.name}") as span:
            span.set_attribute("agent.name", self.name)
            # Span attributes must be primitives — coerce multimodal input
            # to a readable text summary.
            normalized_input_parts: list[ContentPart] = (
                [input] if isinstance(input, str) else normalize_input(input)
            )
            input_text = (
                input
                if isinstance(input, str)
                else _input_summary_text(normalized_input_parts)
            )
            span.set_attribute("agent.input", input_text)

            # Persist multimodal attachments (Image/PDF) to the
            # ``trace_attachments`` table so the UI / Replay can fetch
            # thumbnails and (optionally) the original bytes.
            try:
                from fastaiagent.multimodal.image import Image as _MMImage
                from fastaiagent.multimodal.pdf import PDF as _MMPDF

                if any(
                    isinstance(p, (_MMImage, _MMPDF)) for p in normalized_input_parts
                ):
                    from fastaiagent.trace.attachments import save_parts_for_span
                    from fastaiagent.trace.storage import TraceStore

                    span_ctx = span.get_span_context()
                    saved = save_parts_for_span(
                        db=TraceStore.default()._db,
                        trace_id=format(span_ctx.trace_id, "032x"),
                        span_id=format(span_ctx.span_id, "016x"),
                        parts=normalized_input_parts,
                        role="input",
                    )
                    if saved:
                        span.set_attribute(
                            "fastaiagent.input.attachment_ids",
                            json.dumps([r.attachment_id for r in saved]),
                        )
                        span.set_attribute(
                            "fastaiagent.input.media_count", len(saved)
                        )
            except Exception:
                # Trace persistence must never fail the agent run.
                pass

            # Reconstruction metadata for ForkedReplay.arerun (always captured —
            # structural, not payload).
            span.set_attribute("agent.config", json.dumps(self.config.model_dump()))
            span.set_attribute("agent.tools", json.dumps([t.to_dict() for t in self.tools]))
            span.set_attribute(
                "agent.guardrails", json.dumps([g.to_dict() for g in self.guardrails])
            )
            span.set_attribute("agent.llm.provider", self.llm.provider)
            span.set_attribute("agent.llm.model", self.llm.model)
            span.set_attribute("agent.llm.config", json.dumps(self.llm.to_dict()))

            # Resolved system prompt (payload-gated).
            if trace_payloads_enabled():
                try:
                    resolved_prompt = self._resolve_system_prompt(context)
                    if resolved_prompt:
                        span.set_attribute("agent.system_prompt", resolved_prompt)
                except Exception:
                    # Callable system_prompt that needs context — best-effort only.
                    pass

            result = await self._arun_core(
                input, context=context, execution_id=execution_id, **kwargs
            )

            span.set_attribute("agent.output", result.output)
            span.set_attribute("agent.tokens_used", result.tokens_used)
            span.set_attribute("agent.latency_ms", result.latency_ms)

            # Set trace_id on result
            ctx = span.get_span_context()
            result.trace_id = format(ctx.trace_id, "032x")
            return result

    async def _arun_core(
        self,
        input: AgentInput,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
        _resumed_messages: list[Message] | None = None,
        _start_iteration: int = 0,
        **kwargs: Any,
    ) -> AgentResult:
        """Core execution without tracing."""
        start = time.monotonic()
        exec_id = execution_id or str(uuid.uuid4())

        # Set up the execution-scoped ContextVars so interrupt() / @idempotent
        # can find the active execution + checkpointer. ``_agent_path`` is
        # **extended** rather than overwritten so a parent topology
        # (Swarm / Supervisor) can prefix it with its own segment.
        exec_token = _execution_id.set(exec_id)
        parent_path = _agent_path.get()
        new_path = (
            f"{parent_path}/{self._agent_path_label}" if parent_path else self._agent_path_label
        )
        ap_token = _agent_path.set(new_path)
        cp_token = _current_checkpointer.set(self._checkpointer)

        # Run the agent's checkpointer setup once — cheap on subsequent calls.
        if self._checkpointer is not None:
            self._checkpointer.setup()

        try:
            # Normalize once — every downstream step uses the same shape.
            normalized_parts: list[ContentPart] = (
                [input] if isinstance(input, str) else normalize_input(input)
            )
            input_text = (
                input if isinstance(input, str) else _input_summary_text(normalized_parts)
            )

            # Execute input guardrails (blocking). Guardrails are text-only
            # today; we pass the text summary so policies still trigger on
            # the textual portion of multimodal input.
            if self.guardrails:
                await execute_guardrails(self.guardrails, input_text, GuardrailPosition.input)

            # Build messages — when resuming, restore the saved history.
            messages = (
                _resumed_messages
                if _resumed_messages is not None
                else self._build_messages(input, context=context)
            )

            # Inject response_format for structured output.
            response_format = self._build_response_format()
            if response_format is not None:
                kwargs["response_format"] = response_format

            # Middleware context shared across the whole run.
            mw_ctx: MiddlewareContext | None = None
            if self._mw_pipeline:
                mw_ctx = MiddlewareContext(run_context=context, agent_name=self.name)

            try:
                response, tool_calls = await execute_tool_loop(
                    llm=self.llm,
                    messages=messages,
                    tools=self.tools,
                    max_iterations=self.config.max_iterations,
                    tool_choice=self.config.tool_choice,
                    context=context,
                    guardrails=self.guardrails or None,
                    mw_pipeline=self._mw_pipeline if self._mw_pipeline else None,
                    mw_ctx=mw_ctx,
                    checkpointer=self._checkpointer,
                    execution_id=exec_id,
                    agent_name=self.name,
                    start_iteration=_start_iteration,
                    **kwargs,
                )
            except _AgentInterrupted as susp:
                latency = int((time.monotonic() - start) * 1000)
                return AgentResult(
                    output="",
                    execution_id=exec_id,
                    status="paused",
                    pending_interrupt={
                        "reason": susp.reason,
                        "context": susp.context,
                        "node_id": susp.node_id,
                        "agent_path": susp.agent_path,
                    },
                    latency_ms=latency,
                )

            output = response.content or ""
            parsed = self._parse_output(output)

            # Execute output guardrails.
            if self.guardrails:
                await execute_guardrails(self.guardrails, output, GuardrailPosition.output)

            # Store in memory. Memory backends are text-only; record the
            # text summary so multimodal calls don't break the memory store.
            if self.memory:
                self.memory.add(UserMessage(input_text))
                from fastaiagent.llm.message import AssistantMessage

                self.memory.add(AssistantMessage(output))

            latency = int((time.monotonic() - start) * 1000)
            tokens = response.usage.get("total_tokens", 0)

            return AgentResult(
                output=output,
                parsed=parsed,
                tool_calls=tool_calls,
                tokens_used=tokens,
                latency_ms=latency,
                execution_id=exec_id,
                status="completed",
            )
        finally:
            _current_checkpointer.reset(cp_token)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)

    async def astream(
        self,
        input: AgentInput,
        *,
        context: RunContext | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming execution — yields StreamEvent objects as tokens arrive.

        ``input`` accepts the same shapes as :py:meth:`arun` (string,
        ``Image``, ``PDF``, or a mixed list).

        Runs input guardrails before streaming begins. Output guardrails
        run after streaming completes. Memory is updated at the end.

        When the agent has middleware or a checkpointer configured, those
        are forwarded to :func:`stream_tool_loop` so middleware hooks fire
        and checkpoints are written during streaming — matching the behavior
        of :meth:`arun`.

        Example:
            async for event in agent.astream("Hello"):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
        """
        exec_id = execution_id or str(uuid.uuid4())

        input_text = (
            input
            if isinstance(input, str)
            else _input_summary_text(normalize_input(input))
        )

        # Execute input guardrails (blocking) on the text portion.
        if self.guardrails:
            await execute_guardrails(self.guardrails, input_text, GuardrailPosition.input)

        messages = self._build_messages(input, context=context)

        # Inject response_format for structured output
        response_format = self._build_response_format()
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Middleware context shared across the whole run.
        mw_ctx: MiddlewareContext | None = None
        if self._mw_pipeline:
            mw_ctx = MiddlewareContext(run_context=context, agent_name=self.name)

        # Set up execution-scoped ContextVars so interrupt() / @idempotent
        # can find the active execution + checkpointer.
        exec_token = _execution_id.set(exec_id)
        parent_path = _agent_path.get()
        new_path = (
            f"{parent_path}/{self._agent_path_label}" if parent_path else self._agent_path_label
        )
        ap_token = _agent_path.set(new_path)
        cp_token = _current_checkpointer.set(self._checkpointer)

        if self._checkpointer is not None:
            self._checkpointer.setup()

        try:
            # Stream tool loop — yields events to caller
            accumulated_text = ""
            async for event in stream_tool_loop(
                llm=self.llm,
                messages=messages,
                tools=self.tools,
                max_iterations=self.config.max_iterations,
                tool_choice=self.config.tool_choice,
                context=context,
                guardrails=self.guardrails or None,
                mw_pipeline=self._mw_pipeline if self._mw_pipeline else None,
                mw_ctx=mw_ctx,
                checkpointer=self._checkpointer,
                execution_id=exec_id,
                agent_name=self.name,
                **kwargs,
            ):
                if isinstance(event, TextDelta):
                    accumulated_text += event.text
                yield event

            output = accumulated_text

            # Execute output guardrails
            if self.guardrails:
                await execute_guardrails(self.guardrails, output, GuardrailPosition.output)

            # Store in memory (text summary for multimodal inputs).
            if self.memory:
                self.memory.add(UserMessage(input_text))
                from fastaiagent.llm.message import AssistantMessage

                self.memory.add(AssistantMessage(output))
        finally:
            _current_checkpointer.reset(cp_token)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)

    def resume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous resume wrapper around :meth:`aresume`."""
        return run_sync(
            self.aresume(
                execution_id,
                resume_value=resume_value,
                context=context,
                **kwargs,
            )
        )

    async def aresume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        agent_path_prefix: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Resume a paused or crashed agent run.

        Three resume shapes:

        1. **Interrupted** (``status="interrupted"`` checkpoint, pending row
           in ``pending_interrupts``): pass ``resume_value=Resume(...)``.
           The pending row is atomically claimed; concurrent resumers see
           :class:`AlreadyResumed`. The suspended tool is re-invoked with
           ``_resume_value`` in scope so :func:`interrupt` returns the value,
           then the agent loop continues.
        2. **Tool-boundary crash** (latest checkpoint is ``turn:N/tool:X``,
           ``status="completed"``): the saved tool is re-invoked with the
           saved args; the agent loop continues afterwards. The LLM is NOT
           re-called — the assistant's tool_calls are already in messages.
        3. **Turn-boundary crash** (latest checkpoint is ``turn:N``): the
           agent loop re-enters at iteration N, re-issuing the LLM call.
           Wrap side-effectful tool functions with ``@idempotent`` to avoid
           double-execution.

        ``agent_path_prefix`` (advanced): when this Agent is a worker inside
        a Supervisor / Swarm, multiple agents share one ``execution_id`` and
        the global ``get_last`` returns whichever sibling wrote most
        recently. Passing the agent's own path prefix
        (e.g. ``"supervisor:planner/worker:worker_b"``) scopes resume to
        the worker's own subtree.
        """
        from fastaiagent._internal.errors import ChainCheckpointError

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()

        def _scoped_latest(*, status: str | None = None) -> Checkpoint | None:
            """Return the most recently committed checkpoint, optionally
            filtered to those whose ``agent_path`` starts with the prefix
            and (if given) whose ``status`` matches.
            """
            if agent_path_prefix is None and status is None:
                return store.get_last(execution_id)
            for cp in reversed(store.list(execution_id, limit=500)):
                if status is not None and cp.status != status:
                    continue
                if agent_path_prefix is not None and not (
                    (cp.agent_path or "").startswith(agent_path_prefix)
                ):
                    continue
                return cp
            return None

        if resume_value is not None:
            # Caller is resuming an interrupted run. Atomically claim the
            # pending row before doing anything else — concurrent resumers
            # / second clicks of "Approve" see AlreadyResumed.
            claimed = store.delete_pending_interrupt_atomic(execution_id)
            if claimed is None:
                raise AlreadyResumed(
                    f"Agent execution '{execution_id}' has no pending interrupt to "
                    "claim — either it was never suspended or another resumer won."
                )
            # Find the interrupted checkpoint corresponding to this pending
            # row (most recent ``interrupted`` row for this execution).
            interrupted = _scoped_latest(status="interrupted")
            if interrupted is None:
                raise AlreadyResumed(
                    f"Agent execution '{execution_id}' had a pending interrupt row "
                    "but no matching interrupted checkpoint — storage is inconsistent."
                )
            latest = interrupted
        else:
            latest_opt = _scoped_latest()
            if latest_opt is None:
                raise ChainCheckpointError(
                    f"No checkpoint found for agent execution '{execution_id}'"
                )
            if latest_opt.status == "interrupted":
                raise ChainCheckpointError(
                    f"Agent execution '{execution_id}' is suspended on interrupt(); "
                    "pass resume_value=Resume(...) to agent.resume()."
                )
            latest = latest_opt

        raw_msgs = latest.state_snapshot.get("messages", [])
        messages: list[Message] = [Message.model_validate(m) for m in raw_msgs]
        start_iteration = int(latest.state_snapshot.get("turn", 0))
        is_tool_boundary = "/tool:" in latest.node_id

        # Original input from the first user message — re-passed only so
        # ``_arun_core`` can record it in memory at the end of the run.
        # Multimodal resumes: the first user message may be a list, in which
        # case the saved messages already carry the multimodal content. We
        # only need a text summary here for memory.add() at the end.
        original_input: str = ""
        for m in messages:
            if m.role.value == "user" and m.content:
                if isinstance(m.content, str):
                    original_input = m.content
                else:
                    original_input = _input_summary_text(list(m.content))
                break

        if not is_tool_boundary:
            # Pure turn-boundary resume — re-issue LLM at this iteration.
            return await self._arun_core(
                original_input,
                context=context,
                execution_id=execution_id,
                _resumed_messages=messages,
                _start_iteration=start_iteration,
                **kwargs,
            )

        # Tool-boundary or interrupted: re-invoke the saved tool, append
        # its ToolMessage, then continue the loop at iteration+1.
        tool_name = latest.state_snapshot.get("tool_name", "")
        tool_call_id = latest.state_snapshot.get("tool_call_id", "")
        tool_args = dict(latest.node_input or {})

        tool: Tool | None = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break
        if tool is None:
            raise ChainCheckpointError(
                f"Agent execution '{execution_id}' references tool {tool_name!r} "
                "which is not registered on this Agent. Pass the same tool list "
                "you used at run time."
            )

        rv_token = None
        if resume_value is not None:
            # Pending row already claimed above; just bind the resume value
            # so the suspended tool's interrupt() returns it on re-entry.
            rv_token = _resume_value.set(resume_value)

        # Bind execution-scoped ContextVars exactly like _arun_core would —
        # the suspended tool may itself call interrupt() / @idempotent. Match
        # _arun_core's nestable ``_agent_path`` semantics so a Swarm /
        # Supervisor prefix is preserved.
        exec_token = _execution_id.set(execution_id)
        parent_path = _agent_path.get()
        new_path = (
            f"{parent_path}/{self._agent_path_label}" if parent_path else self._agent_path_label
        )
        ap_token = _agent_path.set(new_path)
        cp_token = _current_checkpointer.set(self._checkpointer)
        try:
            try:
                tool_result = await tool.aexecute(tool_args, context=context)
            except _AgentInterrupted:
                # interrupt() was called again inside the resumed tool —
                # _record_agent_interrupt already wrote the new pending
                # row. Bubble paused state up.
                raise
            if tool_result.success:
                from fastaiagent.llm.message import ToolMessage

                if isinstance(tool_result.output, str):
                    result_text = tool_result.output
                else:
                    result_text = json.dumps(tool_result.output, default=str)
                messages.append(ToolMessage(content=result_text, tool_call_id=tool_call_id))
            else:
                from fastaiagent.llm.message import ToolMessage

                messages.append(
                    ToolMessage(
                        content=f"Error: {tool_result.error}",
                        tool_call_id=tool_call_id,
                    )
                )
        finally:
            _current_checkpointer.reset(cp_token)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)
            if rv_token is not None:
                _resume_value.reset(rv_token)

        # Continue the loop at the next iteration with the updated
        # message history.
        try:
            return await self._arun_core(
                original_input,
                context=context,
                execution_id=execution_id,
                _resumed_messages=messages,
                _start_iteration=start_iteration + 1,
                **kwargs,
            )
        except _AgentInterrupted as susp:
            return AgentResult(
                output="",
                execution_id=execution_id,
                status="paused",
                pending_interrupt={
                    "reason": susp.reason,
                    "context": susp.context,
                    "node_id": susp.node_id,
                    "agent_path": susp.agent_path,
                },
            )

    def stream(
        self, input: str, *, context: RunContext | None = None, trace: bool = True, **kwargs: Any
    ) -> AgentResult:
        """Synchronous streaming — collects stream into AgentResult.

        For true streaming, use ``astream()`` in an async context.
        """

        async def _collect() -> AgentResult:
            start = time.monotonic()
            text_parts: list[str] = []
            async for event in self.astream(input, context=context, trace=trace, **kwargs):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
            latency = int((time.monotonic() - start) * 1000)
            output = "".join(text_parts)
            parsed = self._parse_output(output)
            return AgentResult(
                output=output,
                parsed=parsed,
                latency_ms=latency,
            )

        return run_sync(_collect())

    def _resolve_system_prompt(self, context: RunContext | None = None) -> str:
        """Resolve system_prompt to a string. Calls it if callable."""
        if callable(self.system_prompt):
            return self.system_prompt(context)
        return self.system_prompt

    def _build_messages(
        self, input: AgentInput, context: RunContext | None = None
    ) -> list[Message]:
        """Build the message array for the LLM.

        Accepts string, ``Image``, ``PDF``, or a list of those parts. When
        the input is multimodal, the trailing user message carries a
        ``list[ContentPart]`` that ``LLMClient`` later renders per provider.
        """
        messages: list[Message] = []

        system_text = self._resolve_system_prompt(context)
        if system_text:
            messages.append(SystemMessage(system_text))

        if isinstance(input, str):
            user_content: str | list[ContentPart] = input
            query_text = input
        else:
            parts = normalize_input(input)
            # Single-string lists collapse back to plain strings so the wire
            # shape is identical to the legacy text-only path.
            if len(parts) == 1 and isinstance(parts[0], str):
                user_content = parts[0]
                query_text = parts[0]
            else:
                user_content = parts
                query_text = _input_summary_text(parts)

        # Add memory context. ComposableMemory uses the query text to run
        # query-conditioned blocks (e.g. VectorBlock); AgentMemory ignores
        # it. Multimodal queries fall back to their text summary.
        if self.memory:
            messages.extend(self.memory.get_context(query=query_text))

        messages.append(UserMessage(user_content))
        return messages

    def as_mcp_server(
        self,
        transport: str = "stdio",
        expose_tools: bool = False,
        expose_system_prompt: bool = True,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Any:
        """Expose this agent as an MCP server.

        Returns a :class:`fastaiagent.tool.mcp_server.FastAIAgentMCPServer`.
        Call ``await server.run()`` to start the stdio loop.

        Requires ``pip install 'fastaiagent[mcp-server]'``.

        Example::

            agent.as_mcp_server(transport="stdio").run()
        """
        from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

        return FastAIAgentMCPServer(
            target=self,
            transport=transport,  # type: ignore[arg-type]
            expose_tools=expose_tools,
            expose_system_prompt=expose_system_prompt,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format for platform push."""
        if callable(self.system_prompt):
            raise ValueError(
                f"Agent '{self.name}' has a callable system_prompt which cannot be "
                f"serialized. Use a static string for agents pushed to the platform."
            )
        d: dict[str, Any] = {
            "name": self.name,
            "agent_type": "single",
            "system_prompt": self.system_prompt,
            "llm_endpoint": self.llm.to_dict(),
            "tools": [t.to_dict() for t in self.tools],
            "guardrails": [g.to_dict() for g in self.guardrails],
            "config": self.config.model_dump(),
        }
        # Include response_format schema if output_type is set.
        # Note: output_type (a Python class) cannot be restored from JSON.
        response_format = self._build_response_format()
        if response_format is not None:
            d["config"]["response_format"] = response_format
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Deserialize from canonical format (platform pull).

        Note: output_type cannot be restored from serialized data because
        it is a Python class. The response_format schema in config is
        informational and will be passed through to the LLM if present.
        """
        return cls(
            name=data["name"],
            system_prompt=data.get("system_prompt", ""),
            llm=LLMClient.from_dict(data.get("llm_endpoint", {})),
            tools=[Tool.from_dict(t) for t in data.get("tools", [])],
            guardrails=[Guardrail.from_dict(g) for g in data.get("guardrails", [])],
            config=AgentConfig(**data.get("config", {})),
        )
