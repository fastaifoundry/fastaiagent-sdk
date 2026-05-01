"""Swarm — peer-to-peer multi-agent topology.

In a ``Swarm``, each agent can hand off control to any allowed peer (or all
peers by default). Unlike :class:`fastaiagent.agent.Supervisor`, there is no
central LLM making routing decisions — the currently active agent itself
calls a ``handoff_to_<peer>`` tool to pass control. Useful for
research→write→critique loops, triage-then-specialist patterns, and any
workflow where the "who should handle this next" decision belongs to the
specialist rather than a coordinator.

Handoffs are registered via an allowlist (``handoffs`` dict) so agents can
only hand off to pre-declared peers. A cycle guard (``max_handoffs``) caps
the total number of handoffs per run.

Public API::

    from fastaiagent import Agent, Swarm

    researcher = Agent(name="researcher", llm=llm, tools=[search_tool], ...)
    writer     = Agent(name="writer",     llm=llm, ...)
    critic     = Agent(name="critic",     llm=llm, ...)

    swarm = Swarm(
        name="content_team",
        agents=[researcher, writer, critic],
        entrypoint="researcher",
        handoffs={
            "researcher": ["writer"],
            "writer":     ["critic", "researcher"],
            "critic":     ["writer"],
        },
        max_handoffs=8,
    )

    result = swarm.run("Write a 500-word brief on large language models.")

``Swarm`` implements the same ``run`` / ``arun`` / ``astream`` / ``stream``
surface as :class:`fastaiagent.agent.Agent`, so it drops into a
:class:`fastaiagent.chain.Chain` node or wraps inside another ``Swarm``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.errors import AgentError, StopAgent
from fastaiagent.agent.agent import Agent, AgentResult
from fastaiagent.agent.context import RunContext
from fastaiagent.agent.middleware import AgentMiddleware
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.chain.idempotent import _current_checkpointer
from fastaiagent.chain.interrupt import (
    AlreadyResumed,
    Resume,
    _agent_path,
    _execution_id,
)
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer
from fastaiagent.llm.message import AssistantMessage, UserMessage
from fastaiagent.llm.stream import (
    HandoffEvent,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
)
from fastaiagent.tool.base import Tool
from fastaiagent.tool.function import FunctionTool

__all__ = ["Swarm", "SwarmError", "SwarmState"]

_HANDOFF_SENTINEL = "__HANDOFF__"


class _ExitAfterHandoff(AgentMiddleware):
    """Internal middleware that stops an agent's tool loop as soon as a
    ``handoff_to_<peer>`` tool executes. Without this, the agent's LLM
    tends to re-call the handoff tool on the next iteration (since the
    sentinel string looks like legitimate tool output), which burns the
    MaxIterations budget.
    """

    name = "_exit_after_handoff"

    async def wrap_tool(self, ctx, tool, args, call_next):
        result = await call_next(tool, args)
        if tool.name.startswith("handoff_to_"):
            # StopAgent short-circuits the remaining tool calls in this
            # iteration and the rest of the inner tool loop. Gap 3's
            # executor preserves ``all_tool_calls`` so the swarm outer
            # loop can still see the handoff record.
            raise StopAgent(_HANDOFF_SENTINEL)
        return result


class SwarmError(AgentError):
    """Raised when a swarm run violates its structural constraints
    (missing agents, unknown entrypoint, cycle-guard exhausted, etc.).
    """


@dataclass
class SwarmState:
    """Shared state across agents in a swarm run.

    Attributes:
        shared: Free-form blackboard any agent can read/write via a handoff's
            ``context=`` argument. Persists across handoffs within a single run.
        handoff_count: Number of handoffs so far (used by the cycle guard).
        path: Ordered list of agent names visited this run.
        last_reason: Reason string from the most recent handoff call.
    """

    shared: dict[str, Any] = field(default_factory=dict)
    handoff_count: int = 0
    path: list[str] = field(default_factory=list)
    last_reason: str = ""


def _encode_handoff(target: str, reason: str) -> str:
    """Serialize a handoff signal as the tool's string return value."""
    return f"{_HANDOFF_SENTINEL}:{target}:{reason}"


def _decode_handoff(tool_output: Any) -> tuple[str, str] | None:
    """Inverse of :func:`_encode_handoff`. Returns (target, reason) or None."""
    if not isinstance(tool_output, str):
        return None
    if not tool_output.startswith(_HANDOFF_SENTINEL + ":"):
        return None
    # Format: __HANDOFF__:<target>:<reason>
    parts = tool_output.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1], parts[2]


def _build_handoff_input(
    *,
    source: str,
    reason: str,
    original_input: Any,
    state: SwarmState,
) -> Any:
    """Build the input the handoff target receives.

    Wraps the handoff context (who handed off, why, what's in shared state)
    around the *original* user request — so multimodal payloads (Image,
    PDF) flow through to the target intact rather than being collapsed to
    a string. Pure text inputs keep the legacy single-string template.
    """
    from fastaiagent.multimodal.image import Image as MMImage
    from fastaiagent.multimodal.pdf import PDF as MMPDF

    handoff_text = (
        f"{source} handed off to you with reason: {reason!r}. "
        f"Current shared state: {state.shared!r}. "
        f"The original request follows — please continue."
    )
    if isinstance(original_input, str):
        return (
            f"{source} handed off to you with reason: {reason!r}. "
            f"Earlier request: {original_input!r}. Current shared state: "
            f"{state.shared!r}. Please continue."
        )
    if isinstance(original_input, (MMImage, MMPDF)):
        return [handoff_text, original_input]
    if isinstance(original_input, list):
        return [handoff_text, *original_input]
    return f"{source} handed off to you with reason: {reason!r}. Please continue."


def _swarm_snapshot(
    *,
    iteration: int,
    current: str,
    current_input: Any,
    original_input: Any,
    state: SwarmState,
    accumulated_tool_calls: list[dict[str, Any]],
    total_tokens: int,
) -> dict[str, Any]:
    """Serialize the swarm loop's hot state into a JSON-safe dict.

    Reuses the Chain-side multimodal walker so ``Image``/``PDF`` instances
    placed in ``state.shared`` (or in ``current_input`` / ``original_input``
    when the swarm was started with multimodal content) survive the JSON
    round-trip to SQLite and rehydrate on resume.
    """
    from fastaiagent.chain.state import _serialize_for_checkpoint

    return {
        "iteration": iteration,
        "active_agent": current,
        "current_input": _serialize_for_checkpoint(current_input),
        "original_input": _serialize_for_checkpoint(original_input),
        "shared_context": _serialize_for_checkpoint(dict(state.shared)),
        "handoff_count": state.handoff_count,
        "path": list(state.path),
        "last_reason": state.last_reason,
        "accumulated_tool_calls": list(accumulated_tool_calls),
        "total_tokens": total_tokens,
    }


def _restore_state(snapshot: dict[str, Any]) -> SwarmState:
    """Inverse of :func:`_swarm_snapshot` — rebuild a :class:`SwarmState`."""
    from fastaiagent.chain.state import _hydrate_from_checkpoint

    state = SwarmState()
    state.shared = _hydrate_from_checkpoint(dict(snapshot.get("shared_context", {})))
    state.handoff_count = int(snapshot.get("handoff_count", 0))
    state.path = list(snapshot.get("path", []))
    state.last_reason = str(snapshot.get("last_reason", ""))
    return state


def _parse_active_agent(swarm_name: str, agent_path: str | None) -> str | None:
    """Extract the ``<agent>`` segment from a ``swarm:<s>/agent:<a>/...`` path."""
    if not agent_path:
        return None
    prefix = f"swarm:{swarm_name}/agent:"
    if not agent_path.startswith(prefix):
        return None
    rest = agent_path[len(prefix) :]
    return rest.split("/", 1)[0] if rest else None


class Swarm:
    """A set of peer agents that hand off to each other via tool calls.

    Args:
        name: Swarm identifier (shown in traces and ``AgentResult``).
        agents: All agents in the swarm. Each must have a unique ``name``.
        entrypoint: Name of the agent that receives the initial input.
        handoffs: Allowlist mapping ``agent_name -> [allowed_targets]``.
            If ``None``, every agent may hand off to every other agent
            (full mesh).
        max_handoffs: Cycle guard. Raises :class:`SwarmError` if exceeded.
        config: Not used directly — agents' own ``config`` is honored.
    """

    def __init__(
        self,
        name: str,
        agents: Sequence[Agent],
        entrypoint: str,
        handoffs: dict[str, list[str]] | None = None,
        max_handoffs: int = 8,
        checkpointer: Checkpointer | None = None,
    ):
        if not agents:
            raise SwarmError("Swarm requires at least one agent")
        if max_handoffs < 0:
            raise SwarmError("max_handoffs must be >= 0")

        self.name = name
        self.agents: dict[str, Agent] = {}
        for agent in agents:
            if agent.name in self.agents:
                raise SwarmError(f"Duplicate agent name {agent.name!r} in swarm {name!r}")
            self.agents[agent.name] = agent

        if entrypoint not in self.agents:
            raise SwarmError(f"entrypoint {entrypoint!r} not in agents: {list(self.agents)}")
        self.entrypoint = entrypoint

        if handoffs is None:
            handoffs = {src: [dst for dst in self.agents if dst != src] for src in self.agents}
        for src, targets in handoffs.items():
            if src not in self.agents:
                raise SwarmError(f"handoffs key {src!r} not in agents: {list(self.agents)}")
            for dst in targets:
                if dst not in self.agents:
                    raise SwarmError(f"handoffs[{src!r}] references unknown agent {dst!r}")
        self.handoffs = handoffs
        self.max_handoffs = max_handoffs
        self._checkpointer: Checkpointer | None = checkpointer

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        input: Any,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
    ) -> AgentResult:
        """Synchronous execution. ``input`` may be a string or any of the
        multimodal shapes ``Agent.run`` accepts (``Image``, ``PDF``, or a
        list of content parts)."""
        return run_sync(self.arun(input, context=context, execution_id=execution_id))

    async def arun(
        self,
        input: Any,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the swarm until an agent produces a final answer with no handoff.

        Returns the final agent's :class:`AgentResult`, enriched with path
        metadata under ``tool_calls`` so callers can inspect the handoff chain.

        ``execution_id`` (optional) names this run for resume; pair with a
        ``checkpointer`` on the Swarm to get crash- and interrupt-recovery
        via :meth:`resume`.
        """
        from fastaiagent.trace.otel import get_tracer

        tracer = get_tracer()
        # Root span wraps the whole swarm run so every child agent span is a
        # descendant — the UI renders one trace with the handoff path.
        with tracer.start_as_current_span(f"swarm.{self.name}") as span:
            span.set_attribute("swarm.name", self.name)
            span.set_attribute("swarm.entrypoint", self.entrypoint)
            span.set_attribute("swarm.agent_count", len(getattr(self, "agents", []) or []))
            span.set_attribute("fastaiagent.runner.type", "swarm")
            # OTel only accepts primitive attribute values — coerce
            # multimodal lists to a readable text summary the same way
            # ``Agent._arun_traced`` does.
            from fastaiagent.agent.agent import _input_summary_text
            from fastaiagent.multimodal.types import normalize_input

            swarm_input_text = (
                input if isinstance(input, str) else _input_summary_text(normalize_input(input))
            )
            span.set_attribute("swarm.input", swarm_input_text)

            result = await self._arun_swarm(
                input, context=context, execution_id=execution_id, **kwargs
            )

            span.set_attribute("swarm.output", result.output)
            span.set_attribute("swarm.handoff_count", len(result.tool_calls or []))
        return result

    async def _arun_swarm(
        self,
        input: Any,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        exec_id = execution_id or str(uuid.uuid4())

        # Bind execution-scoped ContextVars so child agents inherit and
        # extend ``_agent_path`` with their own segment.
        exec_token = _execution_id.set(exec_id)
        ap_token = _agent_path.set(f"swarm:{self.name}")
        cp_token = _current_checkpointer.set(self._checkpointer)

        if self._checkpointer is not None:
            self._checkpointer.setup()

        try:
            state = SwarmState()
            state.path.append(self.entrypoint)
            return await self._run_loop(
                exec_id=exec_id,
                current=self.entrypoint,
                current_input=input,
                original_input=input,
                state=state,
                accumulated_tool_calls=[],
                total_tokens=0,
                start_iter=0,
                start=time.monotonic(),
                context=context,
                kwargs=kwargs,
            )
        finally:
            _current_checkpointer.reset(cp_token)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)

    async def _run_loop(
        self,
        *,
        exec_id: str,
        current: str,
        current_input: Any,
        original_input: Any,
        state: SwarmState,
        accumulated_tool_calls: list[dict[str, Any]],
        total_tokens: int,
        start_iter: int,
        start: float,
        context: RunContext[Any] | None,
        kwargs: dict[str, Any],
        skip_first_checkpoint: bool = False,
        first_agent_result: AgentResult | None = None,
    ) -> AgentResult:
        """Core swarm loop, factored so :meth:`aresume` can re-enter it.

        ``first_agent_result`` lets ``aresume`` inject the result of the
        first (resumed) agent without re-running it. Subsequent iterations
        run normally.
        """
        iteration = start_iter
        injected = first_agent_result

        while True:
            # Handoff-boundary checkpoint — captures everything needed to
            # resume right before the active agent runs. ``skip_first_checkpoint``
            # lets aresume re-enter without writing a duplicate.
            if self._checkpointer is not None and not (
                skip_first_checkpoint and iteration == start_iter
            ):
                self._checkpointer.put(
                    Checkpoint(
                        checkpoint_id=str(uuid.uuid4()),
                        chain_name=self.name,
                        execution_id=exec_id,
                        node_id=f"handoff:{iteration}",
                        node_index=iteration,
                        status="completed",
                        state_snapshot=_swarm_snapshot(
                            iteration=iteration,
                            current=current,
                            current_input=current_input,
                            original_input=original_input,
                            state=state,
                            accumulated_tool_calls=accumulated_tool_calls,
                            total_tokens=total_tokens,
                        ),
                        agent_path=f"swarm:{self.name}",
                    )
                )

            if injected is not None:
                result = injected
                injected = None
            else:
                active = self._active_agent(current, state, context=context)
                # Pass the same execution_id to child agents so all
                # checkpoints land under one umbrella execution.
                result = await active.arun(
                    current_input,
                    context=context,
                    execution_id=exec_id,
                    **kwargs,
                )

            # Bubble paused state up — the active agent already wrote its
            # interrupted checkpoint with the swarm-prefixed agent_path.
            if result.status == "paused":
                latency = int((time.monotonic() - start) * 1000)
                return AgentResult(
                    output="",
                    tool_calls=accumulated_tool_calls,
                    tokens_used=total_tokens + result.tokens_used,
                    latency_ms=latency,
                    execution_id=exec_id,
                    status="paused",
                    pending_interrupt=result.pending_interrupt,
                )

            total_tokens += result.tokens_used
            handoff = self._find_handoff(result)
            for call in result.tool_calls:
                call_copy = dict(call)
                call_copy.setdefault("agent", current)
                accumulated_tool_calls.append(call_copy)

            if handoff is None:
                latency = int((time.monotonic() - start) * 1000)
                return AgentResult(
                    output=result.output,
                    tool_calls=accumulated_tool_calls,
                    tokens_used=total_tokens,
                    latency_ms=latency,
                    execution_id=exec_id,
                    status="completed",
                )

            target, reason = handoff
            if target not in self.handoffs.get(current, []):
                raise SwarmError(
                    f"Agent {current!r} tried to hand off to {target!r}, "
                    f"which is not in its allowlist {self.handoffs.get(current, [])}"
                )
            state.handoff_count += 1
            if state.handoff_count > self.max_handoffs:
                raise SwarmError(
                    f"Swarm {self.name!r} exceeded max_handoffs={self.max_handoffs}. "
                    f"Path so far: {' -> '.join(state.path)} -> {target}"
                )
            state.last_reason = reason
            state.path.append(target)
            current_input = _build_handoff_input(
                source=current,
                reason=reason,
                original_input=original_input,
                state=state,
            )
            current = target
            iteration += 1

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
        **kwargs: Any,
    ) -> AgentResult:
        """Resume a paused or crashed swarm run.

        Determines the active agent from the latest checkpoint's
        ``agent_path``, recovers :class:`SwarmState` from the most recent
        ``handoff:N`` boundary, then either resumes the active agent's own
        interrupted run (when ``resume_value`` is given) or re-runs it
        fresh (crash recovery). After the active agent returns, the swarm
        loop continues normally — handoffs, allowlists, max_handoffs all
        still apply.
        """
        from fastaiagent._internal.errors import ChainCheckpointError

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()

        latest = store.get_last(execution_id)
        if latest is None:
            raise ChainCheckpointError(f"No checkpoint found for swarm execution '{execution_id}'")

        # Find the most recent handoff:N boundary to recover SwarmState.
        # Walk the checkpoint list in reverse — handoff rows always carry
        # ``agent_path == swarm:<self.name>``.
        all_cps = store.list(execution_id, limit=500)
        handoff_cp = None
        for cp in reversed(all_cps):
            if cp.node_id.startswith("handoff:") and cp.agent_path == f"swarm:{self.name}":
                handoff_cp = cp
                break
        if handoff_cp is None:
            raise ChainCheckpointError(
                f"Swarm execution '{execution_id}' has no handoff checkpoints — "
                "did this swarm ever run?"
            )

        snapshot = handoff_cp.state_snapshot
        active_name = _parse_active_agent(self.name, latest.agent_path) or str(
            snapshot.get("active_agent", self.entrypoint)
        )
        if active_name not in self.agents:
            raise ChainCheckpointError(
                f"Swarm execution '{execution_id}' references active agent "
                f"{active_name!r}, which is not registered on this Swarm."
            )

        # Reject mismatched resume shapes early.
        if resume_value is None and latest.status == "interrupted":
            raise ChainCheckpointError(
                f"Swarm execution '{execution_id}' is suspended on interrupt(); "
                "pass resume_value=Resume(...) to swarm.resume()."
            )

        state = _restore_state(snapshot)
        current_input = str(snapshot.get("current_input", ""))
        original_input = str(snapshot.get("original_input", current_input))
        accumulated = list(snapshot.get("accumulated_tool_calls", []))
        total_tokens = int(snapshot.get("total_tokens", 0))
        iteration = int(snapshot.get("iteration", 0))

        if self._checkpointer is not None:
            self._checkpointer.setup()

        # Bind ContextVars so the resumed agent inherits the swarm prefix.
        exec_token = _execution_id.set(execution_id)
        ap_token = _agent_path.set(f"swarm:{self.name}")
        cp_token = _current_checkpointer.set(self._checkpointer)
        try:
            active_agent = self._active_agent(active_name, state, context=context)
            # Re-run the active agent. ``aresume`` handles interrupt-claim
            # + suspended-tool re-entry; plain ``arun`` is the crash-recovery
            # path (re-issues LLM, re-runs tools).
            if resume_value is not None or latest.status == "interrupted":
                first_result = await active_agent.aresume(
                    execution_id,
                    resume_value=resume_value,
                    context=context,
                    **kwargs,
                )
            else:
                first_result = await active_agent.arun(
                    current_input,
                    context=context,
                    execution_id=execution_id,
                    **kwargs,
                )

            return await self._run_loop(
                exec_id=execution_id,
                current=active_name,
                current_input=current_input,
                original_input=original_input,
                state=state,
                accumulated_tool_calls=accumulated,
                total_tokens=total_tokens,
                start_iter=iteration,
                start=time.monotonic(),
                context=context,
                kwargs=kwargs,
                skip_first_checkpoint=True,
                first_agent_result=first_result,
            )
        except AlreadyResumed:
            # Re-raise so callers can distinguish "already resumed" from
            # generic checkpoint errors.
            raise
        finally:
            _current_checkpointer.reset(cp_token)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)

    async def astream(
        self, input: Any, *, context: RunContext | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream events from the currently active agent. Handoffs emit a
        :class:`HandoffEvent` before the target agent starts streaming.
        """
        state = SwarmState()
        state.path.append(self.entrypoint)
        current = self.entrypoint
        current_input = input

        while True:
            active = self._active_agent(current, state, context=context)
            handoff_from_stream: tuple[str, str] | None = None

            inner_stream = active.astream(current_input, context=context, **kwargs)
            try:
                async for event in inner_stream:
                    # Detect handoff intent live from the stream — no second
                    # LLM call needed. Break out as soon as it fires so the
                    # active agent's tool loop doesn't keep looping (the
                    # sync-path stop-after-handoff middleware does not fire
                    # in the streaming path in 0.5.0).
                    if isinstance(event, ToolCallEnd) and event.tool_name.startswith("handoff_to_"):
                        target = event.tool_name[len("handoff_to_") :]
                        reason = str(event.arguments.get("reason", "")) if event.arguments else ""
                        handoff_from_stream = (target, reason)
                        ctx_delta = event.arguments.get("context") if event.arguments else None
                        if isinstance(ctx_delta, dict):
                            state.shared.update(ctx_delta)
                        yield event
                        break
                    yield event
            finally:
                await inner_stream.aclose()

            if handoff_from_stream is None:
                return
            target, reason = handoff_from_stream
            if target not in self.handoffs.get(current, []):
                raise SwarmError(
                    f"Agent {current!r} tried to hand off to {target!r}, "
                    f"not in allowlist {self.handoffs.get(current, [])}"
                )
            state.handoff_count += 1
            if state.handoff_count > self.max_handoffs:
                raise SwarmError(f"Swarm {self.name!r} exceeded max_handoffs={self.max_handoffs}")
            yield HandoffEvent(from_agent=current, to_agent=target, reason=reason)
            state.path.append(target)
            state.last_reason = reason
            current_input = _build_handoff_input(
                source=current,
                reason=reason,
                original_input=input,
                state=state,
            )
            current = target

    def stream(self, input: Any, *, context: RunContext | None = None) -> AgentResult:
        """Synchronous streaming — collects into an :class:`AgentResult`."""

        async def _collect() -> AgentResult:
            start = time.monotonic()
            text_parts: list[str] = []
            async for event in self.astream(input, context=context):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
            latency = int((time.monotonic() - start) * 1000)
            return AgentResult(output="".join(text_parts), latency_ms=latency)

        return run_sync(_collect())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _active_agent(
        self,
        name: str,
        state: SwarmState,
        context: RunContext | None = None,
    ) -> Agent:
        """Return a clone of ``agents[name]`` with handoff tools injected.

        The clone shares everything with the original (llm, system_prompt,
        memory, guardrails, middleware) but gets additional per-run handoff
        tools. The original agent is left untouched so multiple swarms can
        share the same agent instances without cross-contamination.
        """
        base = self.agents[name]
        handoff_tools = self._build_handoff_tools(name, state)
        merged_tools: list[Tool] = list(base.tools) + handoff_tools
        # Prepend the internal exit-after-handoff middleware so the inner
        # tool loop stops as soon as a handoff executes. User-supplied
        # middleware still runs (around the real tool) because ours wraps
        # outermost and only short-circuits after the real tool returns.
        merged_middleware = [_ExitAfterHandoff(), *base.middleware]
        return Agent(
            name=base.name,
            system_prompt=base.system_prompt,
            llm=base.llm,
            tools=merged_tools,
            guardrails=base.guardrails,
            memory=base.memory,
            config=base.config,
            output_type=base.output_type,
            middleware=merged_middleware,
            # Forward the swarm's checkpointer so the cloned agent writes
            # turn / tool / interrupted checkpoints under the same execution
            # with the nested ``swarm:.../agent:...`` path.
            checkpointer=self._checkpointer,
        )

    def _build_handoff_tools(self, current: str, state: SwarmState) -> list[Tool]:
        """Create one ``handoff_to_<peer>`` FunctionTool per allowed peer."""
        tools: list[Tool] = []
        for peer in self.handoffs.get(current, []):
            peer_agent = self.agents[peer]
            peer_desc = (
                peer_agent.system_prompt[:160] if isinstance(peer_agent.system_prompt, str) else ""
            )

            def _handoff(
                reason: str = "",
                context: dict[str, Any] | None = None,
                _peer: str = peer,
                _state: SwarmState = state,
            ) -> str:
                # Only *record* the context delta on the shared blackboard;
                # the actual routing happens in arun/astream once the tool
                # result reaches it.
                if context:
                    _state.shared.update(context)
                return _encode_handoff(_peer, reason)

            tools.append(
                FunctionTool(
                    name=f"handoff_to_{peer}",
                    fn=_handoff,
                    description=(
                        f"Hand control off to {peer}. Use when {peer} is better "
                        f"suited to proceed. About {peer}: {peer_desc}"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why you are handing off to this peer.",
                            },
                            "context": {
                                "type": "object",
                                "description": (
                                    "Optional key/value pairs to stash on the "
                                    "swarm's shared blackboard for the next agent."
                                ),
                            },
                        },
                        "required": ["reason"],
                    },
                )
            )
        return tools

    @staticmethod
    def _find_handoff(result: AgentResult) -> tuple[str, str] | None:
        """Scan a result's tool calls for the most recent handoff sentinel."""
        return Swarm._find_handoff_from_calls(result.tool_calls)

    @staticmethod
    def _find_handoff_from_calls(
        tool_calls: list[dict[str, Any]],
    ) -> tuple[str, str] | None:
        for call in reversed(tool_calls):
            decoded = _decode_handoff(call.get("output"))
            if decoded is not None:
                return decoded
            name = call.get("tool_name") or ""
            if name.startswith("handoff_to_"):
                # Handoff tool was called but output not surfaced; derive from
                # the tool name itself. ``reason`` is unknown here.
                target = name[len("handoff_to_") :]
                return target, ""
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the swarm structure. Handoff callables and agent
        instances are captured via their names; rebuilding ``Swarm`` from a
        dict requires the caller to pass the live ``Agent`` instances back
        in (see :meth:`from_dict`)."""
        return {
            "name": self.name,
            "agent_names": list(self.agents),
            "entrypoint": self.entrypoint,
            "handoffs": {k: list(v) for k, v in self.handoffs.items()},
            "max_handoffs": self.max_handoffs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], agents: Sequence[Agent]) -> Swarm:
        """Restore a swarm from a dict + live agents. Caller supplies the
        reconstructed :class:`Agent` instances; names must match ``data["agent_names"]``.
        """
        expected = set(data.get("agent_names", []))
        got = {a.name for a in agents}
        if expected != got:
            raise SwarmError(f"from_dict: expected agents {sorted(expected)}, got {sorted(got)}")
        return cls(
            name=data["name"],
            agents=list(agents),
            entrypoint=data["entrypoint"],
            handoffs=data.get("handoffs"),
            max_handoffs=data.get("max_handoffs", 8),
        )


# Silence unused-import complaints for AssistantMessage / UserMessage — they
# are retained for type checkers referenced via other modules.
_ = (AssistantMessage, UserMessage)
