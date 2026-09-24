"""Chain class — directed graph workflow with cycles, typed state, and checkpointing."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.chain.checkpoint import (
    latest_forkable,
    latest_resumable,
    write_run_end,
    write_run_end_once,
)
from fastaiagent.chain.executor import execute_chain
from fastaiagent.chain.interrupt import AlreadyResumed, Resume
from fastaiagent.chain.node import Edge, Node, NodeConfig, NodeType
from fastaiagent.chain.validator import validate_chain
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer

logger = logging.getLogger(__name__)

#: Config keys the executor and the validator actually read. Anything else in
#: ``**config`` still rides through untouched — a chain may legitimately stash
#: its own metadata on a node — but it earns a warning, because the far more
#: common cause is a misspelled argument that silently became inert config and
#: left the node with nothing to run.
_KNOWN_CONFIG_KEYS = frozenset(
    {
        "agent_name",
        "agents",
        "auto_approve",
        "conditions",
        "input_mapping",
        "input_schema",
        "output_key",
        "output_schema",
        "reachable",
        "template",
        "tool_name",
    }
)

#: Names that *look* like they attach a callable to a node, and do not. Every
#: one of them lands in ``**config`` — ``add_node("fetch", fn=my_tool)`` builds
#: an **agent** node with no agent, which until 1.67.0 ran to
#: ``status="completed"`` with an error string as its output.
_CALLABLE_LOOKALIKE_KWARGS = ("fn", "function", "func", "callable")


def _type_name(obj: object) -> str:
    """``type(obj).__name__`` — a free function because ``add_node`` shadows
    the builtin ``type`` with its own parameter."""
    return obj.__class__.__name__


class ChainResult(BaseModel):
    """Result of a chain execution.

    ``status`` is ``"completed"`` for a normal run, or ``"paused"`` when a
    node called :func:`interrupt`. In the paused case ``pending_interrupt``
    holds ``{reason, context, node_id, agent_path}`` — the same payload the
    ``/approvals`` UI reads from the ``pending_interrupts`` table.
    """

    output: Any = None
    final_state: dict[str, Any] = Field(default_factory=dict)
    execution_id: str = ""
    node_results: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    pending_interrupt: dict[str, Any] | None = None
    #: The OTel trace this run emitted, or ``None`` when ``trace=False`` was
    #: passed. Additive in 1.67.0: a chain has opened a ``chain.<name>`` root
    #: span since long before, and ``ChainResult`` had no field to name it —
    #: so eval-to-trace linking and the UI's open-trace affordance had nothing
    #: to point at for a chain.
    trace_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Chain:
    """A directed graph workflow with cycles, typed state, and checkpointing.

    Example:
        chain = Chain("support-pipeline")
        chain.add_node("research", agent=researcher)
        chain.add_node("evaluate", agent=evaluator)
        chain.connect("research", "evaluate")
        chain.connect("evaluate", "research", max_iterations=3, exit_condition="quality >= 0.8")
        result = chain.execute({"message": "My order is late"})
    """

    def __init__(
        self,
        name: str,
        state_schema: dict[str, Any] | None = None,
        checkpoint_enabled: bool = True,
        checkpointer: Checkpointer | None = None,
        *,
        strict_routing: bool = False,
    ):
        """Construct a chain.

        ``strict_routing`` (default ``False``) controls fall-through behavior
        when no outgoing edge matches a node's result:

        * ``False`` (legacy / default): the branch silently terminates.
        * ``True``: raise :class:`fastaiagent._internal.errors.ChainRoutingError`
          so misconfigured chains fail loudly. See ``docs/chains/spec.md``.
        """
        self.name = name
        self.state_schema = state_schema
        self.nodes: list[NodeConfig] = []
        self.edges: list[Edge] = []
        self.checkpoint_enabled = checkpoint_enabled
        self._checkpointer = checkpointer
        self.strict_routing = strict_routing

    def add_node(
        self,
        id: str,
        agent: Any = None,
        tool: Any = None,
        type: NodeType | str | None = None,
        name: str = "",
        *,
        node: Node | None = None,
        output_key: str | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        **config: Any,
    ) -> Chain:
        """Add a node to the chain.

        **The node's type is inferred from what you attach** when you do not
        pass ``type=`` yourself: ``tool=`` builds a tool node, ``node=`` builds a
        tool node, ``agent=`` (or attaching nothing) builds an agent node. An
        explicit ``type=`` always wins, and ``agent=`` alongside ``tool=`` still
        means an agent node — the tool is ignored there exactly as before.

        Before 1.67.0 only ``node=`` inferred anything, so
        ``add_node("fetch", tool=my_tool)`` built an **agent** node with no
        agent, and the run completed with ``{"error": "No agent attached…"}`` as
        that node's output. The docs' own Node Types table taught that form.

        Two mistakes are refused here rather than at run time, because a
        ``TypeError`` naming the right form costs nothing and a node that cannot
        run costs a whole execution:

        * ``fn=`` / ``function=`` / ``func=`` / ``callable=`` — none of them is a
          parameter of this method, so each is absorbed into ``**config`` and
          attaches nothing.
        * a ``tool=`` that is not a Tool — a bare function has no ``aexecute``
          and used to surface as ``AttributeError: 'function' object has no
          attribute 'aexecute'`` from deep inside the executor.

        ``**config`` stays open (``input_mapping``, ``template``, ``conditions``,
        ``agents`` and a chain's own metadata all ride in it); an unrecognised
        key is logged at WARNING rather than refused.

        Pass ``node=`` a :func:`fastaiagent.node`-decorated function to add a
        typed, code-first node. ``output_key`` stores the node's output under a
        named state key (instead of the legacy ``_<id>_output`` wrap), and
        ``input_schema`` / ``output_schema`` (optional JSON schemas) validate the
        node's resolved inputs / output at its boundary. All additive — a node
        without any of these behaves exactly as before.
        """
        lookalikes = [k for k in _CALLABLE_LOOKALIKE_KWARGS if k in config]
        if lookalikes:
            bad = lookalikes[0]
            raise TypeError(
                f"add_node({id!r}, {bad}=...) attaches nothing: {bad!r} is not a "
                f"parameter of add_node, so it was absorbed into the node's config and "
                f"the node was left with nothing to run. Wrap the function as a tool — "
                f"chain.add_node({id!r}, tool=FunctionTool(name=..., fn=...)) — or "
                f"decorate it with @node and pass chain.add_node({id!r}, node=<fn>)."
            )

        explicit_type = type is not None
        if node is not None:
            tool = node.tool
            type = NodeType.tool
            name = name or node.name
            if output_key is None:
                output_key = node.output_key
            if input_schema is None:
                input_schema = node.input_schema
            if output_schema is None:
                output_schema = node.output_schema
        elif not explicit_type:
            # Infer from what was attached. ``agent`` wins when both are given,
            # which is what an un-typed ``agent=`` + ``tool=`` has always meant.
            type = NodeType.tool if (tool is not None and agent is None) else NodeType.agent

        if tool is not None and not hasattr(tool, "aexecute"):
            if isinstance(tool, Node):
                raise TypeError(
                    f"add_node({id!r}, tool=<@node function>) — a @node-decorated "
                    f"function is not a Tool. Pass it as chain.add_node({id!r}, "
                    f"node={tool.name}) so its schemas and output_key come with it."
                )
            raise TypeError(
                f"add_node({id!r}, tool={_type_name(tool)}) — a tool node needs a Tool, "
                f"not a bare {_type_name(tool)}: it has no aexecute(), so the node "
                f"raised AttributeError mid-run. Wrap it — "
                f"chain.add_node({id!r}, tool=FunctionTool(name='...', fn=<fn>)) — or "
                f"decorate it with @node and pass node=<fn>."
            )

        # Stash the 2.4b extras into ``config`` so they ride the existing
        # NodeConfig serialization and the executor can read them per node.
        if output_key is not None:
            config["output_key"] = output_key
        if input_schema is not None:
            config["input_schema"] = input_schema
        if output_schema is not None:
            config["output_schema"] = output_schema
        unknown = sorted(set(config) - _KNOWN_CONFIG_KEYS)
        if unknown:
            logger.warning(
                "Chain '%s' node '%s': unrecognised add_node keyword(s) %s were stored "
                "as node config and will not be read by the executor. If one of them was "
                "meant to attach something, see add_node's signature.",
                self.name,
                id,
                unknown,
            )
        # ``type`` is settled by here — explicit, inferred, or the legacy agent
        # default. Coerced through the enum so the string form callers have
        # always been able to pass (``type="transformer"``) keeps working.
        resolved_type = NodeType(type) if type is not None else NodeType.agent
        node_config = NodeConfig(
            id=id,
            type=resolved_type,
            name=name or id,
            agent=agent,
            agent_name=agent.name if agent and hasattr(agent, "name") else None,
            tool=tool,
            tool_name=tool.name if tool and hasattr(tool, "name") else None,
            config=config,
        )
        self.nodes.append(node_config)
        return self

    def connect(
        self,
        source: str,
        target: str,
        condition: str | None = None,
        max_iterations: int | None = None,
        exit_condition: str | None = None,
        label: str = "",
    ) -> Chain:
        """Connect two nodes with an edge."""
        is_cyclic = max_iterations is not None
        cycle_config: dict[str, Any] = {}
        if is_cyclic:
            cycle_config = {
                "max_iterations": max_iterations,
                "exit_condition": exit_condition,
                "on_max_reached": "error",
            }

        edge = Edge(
            id=f"e_{source}_{target}",
            source=source,
            target=target,
            condition=condition,
            label=label,
            is_cyclic=is_cyclic,
            cycle_config=cycle_config,
        )
        self.edges.append(edge)
        return self

    def validate(self) -> list[str]:
        """Validate chain structure. Returns list of errors."""
        return validate_chain(self.nodes, self.edges)

    def execute(
        self,
        initial_state: dict[str, Any] | None = None,
        trace: bool = True,
        *,
        context: Any | None = None,
        **kwargs: Any,
    ) -> ChainResult:
        """Synchronous execution.

        ``context`` is an optional :class:`fastaiagent.agent.context.RunContext`
        forwarded to every tool and agent node so dependency-injected tools
        (functions declaring a ``ctx: RunContext[Deps]`` parameter) work
        identically inside a Chain and inside an Agent.
        """
        return run_sync(self.aexecute(initial_state, trace=trace, context=context, **kwargs))

    async def aexecute(
        self,
        initial_state: dict[str, Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        hitl_handler: Any = None,
        *,
        context: Any | None = None,
        **kwargs: Any,
    ) -> ChainResult:
        """Async execution of the chain.

        ``context`` is an optional :class:`fastaiagent.agent.context.RunContext`
        propagated to every tool and agent node — see :meth:`execute`.

        ``trace=False`` skips the ``chain.<name>`` OTel root span (and the
        per-chain attributes). Child agent/tool spans are still created;
        they just don't nest under a chain-level parent. Matches the
        ``Agent.run(trace=False)`` contract.
        """
        store: Checkpointer | None = None
        if self.checkpoint_enabled:
            store = self._checkpointer or SQLiteCheckpointer()
            store.setup()

        # Wrap the whole chain in a root span so every child agent span is a
        # descendant of it — the UI can then render a chain as one trace with
        # a tree of agents, rather than N orphan agent traces. When the
        # caller passes ``trace=False`` we skip the span entirely (e.g. when
        # a Chain is invoked as a sub-step from inside another traced
        # workflow that already owns the root span).
        from contextlib import nullcontext

        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import trace_id_of

        if trace:
            span_ctx = get_tracer().start_as_current_span(f"chain.{self.name}")
        else:
            span_ctx = nullcontext(None)

        with span_ctx as span:
            if span is not None:
                span.set_attribute("chain.name", self.name)
                span.set_attribute("chain.node_count", len(self.nodes))
                span.set_attribute("chain.node_ids", ",".join(n.id for n in self.nodes))
                span.set_attribute("fastaiagent.runner.type", "chain")
                span.set_attribute("fastaiagent.framework", "fastaiagent")
                # Captured locally unconditionally so the UI/Replay have full
                # fidelity; payload privacy is enforced at export (N3 egress
                # model), not here.
                if initial_state:
                    import json

                    try:
                        span.set_attribute("chain.input", json.dumps(initial_state, default=str))
                    except (TypeError, ValueError):
                        logger.debug("Failed to serialize chain input for trace", exc_info=True)

            try:
                raw = await execute_chain(
                    nodes=self.nodes,
                    edges=self.edges,
                    initial_state=initial_state or {},
                    state_schema=self.state_schema,
                    checkpointer=store,
                    chain_name=self.name,
                    execution_id=execution_id,
                    hitl_handler=hitl_handler,
                    run_context=context,
                    strict_routing=self.strict_routing,
                )
            except BaseException as exc:
                # Terminal marker for a run that died (audit D5). Written HERE
                # and not in ``execute_chain`` because that function recurses
                # into itself for cycles with the same execution_id — a write
                # there would fire once per loop iteration. ``aexecute`` is
                # provably called once per run.
                if store is not None and execution_id:
                    write_run_end(
                        store,
                        execution_id=execution_id,
                        chain_name=self.name,
                        status="failed",
                        error=exc,
                    )
                raise
            # Only a run that actually ENDED gets a marker. A paused chain has
            # not ended, and a row after its ``interrupted`` one would hide the
            # pause from ``resume``'s status guard.
            if store is not None and raw.get("status") == "completed":
                write_run_end(
                    store,
                    execution_id=raw.get("execution_id") or execution_id or "",
                    chain_name=self.name,
                    status="completed",
                    state_snapshot=raw.get("final_state"),
                )

            if span is not None:
                try:
                    import json as _json

                    span.set_attribute("chain.output", _json.dumps(raw.get("output"), default=str))
                except (TypeError, ValueError):
                    logger.debug("Failed to serialize chain output for trace", exc_info=True)
                span.set_attribute("chain.execution_id", raw.get("execution_id") or "")
                # Read inside the ``with`` — the span context is only valid
                # while the span is current.
                trace_id = trace_id_of(span)
            else:
                trace_id = None

        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
            status=raw.get("status", "completed"),
            pending_interrupt=raw.get("pending_interrupt"),
            trace_id=trace_id,
        )

    async def resume(
        self,
        execution_id: str,
        modified_state: dict[str, Any] | None = None,
        *,
        resume_value: Resume | None = None,
        context: Any | None = None,
    ) -> ChainResult:
        """Resume a failed/paused chain execution from the last checkpoint.

        For an *interrupted* checkpoint (``interrupt()`` was called), pass a
        :class:`Resume` value. The resumer atomically claims the
        ``pending_interrupts`` row before starting; concurrent resumers see
        :class:`AlreadyResumed`.

        For a *failed* checkpoint, ``modified_state`` lets you patch state
        before the next node runs (the existing v0.x behavior).

        ``context`` is forwarded to every tool/agent node executed during the
        resume — supply the same :class:`RunContext` you originally passed to
        :meth:`aexecute` so dependency-injected tools work after a pause.
        """
        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()
        # Restore-anywhere (audit D4): when this machine has never seen the run
        # but the plane is holding it, pull it down before deciding there is
        # nothing to resume. No-op when disconnected or already present.
        from fastaiagent.checkpointers.platform_replica import restore_if_missing

        restore_if_missing(store, execution_id)
        # Refuses a finished run and steps past a `failed` tombstone (audit D5).
        latest = latest_resumable(store, execution_id, runner="Execution")
        if latest is None:
            from fastaiagent._internal.errors import ChainCheckpointError

            raise ChainCheckpointError(f"No checkpoint found for execution '{execution_id}'")

        state = latest.state_snapshot
        if modified_state:
            state.update(modified_state)

        from fastaiagent.chain.executor import _topological_sort

        order = _topological_sort(self.nodes, self.edges)

        start_node: str | None
        if resume_value is not None:
            # Caller is resuming an interrupted workflow. Atomically claim
            # the pending row — concurrent resumers and resumes-after-success
            # both see :class:`AlreadyResumed`, which is the long-standing
            # signal that "there is no pending interrupt to claim" (whether
            # because it was already claimed by a prior resume call, or
            # because the chain was never interrupted in the first place).
            claimed = store.delete_pending_interrupt_atomic(execution_id)
            if claimed is None:
                raise AlreadyResumed(
                    f"Execution '{execution_id}' has no pending interrupt to claim — "
                    "either it was never suspended or another resumer already won."
                )
            # Re-execute the interrupted node from the top so interrupt() can
            # return the resume_value.
            start_node = claimed.node_id
            # Best-effort: report the resolution to a connected plane (no-op when
            # not connected; never blocks/raises). Emitted only on the winning
            # claim, so a losing AlreadyResumed race reports no phantom resolution.
            try:
                from fastaiagent.governance import hitl_kind, resolution_context
                from fastaiagent.trace.hitl_export import record_resolution_event

                record_resolution_event(
                    run_id=execution_id,
                    node=claimed.node_id,
                    approved=resume_value.approved,
                    resolver=resume_value.metadata.get("resolver"),
                    reason=claimed.reason,
                    chain_id=self.name,
                    kind=hitl_kind(claimed.reason),
                    context=resolution_context(claimed.reason, claimed.context),
                )
            except Exception:
                logger.debug("HITL resolution emit failed", exc_info=True)
        elif latest.status == "interrupted":
            from fastaiagent._internal.errors import ChainResumeError

            raise ChainResumeError(
                f"Execution '{execution_id}' is suspended on interrupt(); "
                "pass resume_value=Resume(...) to chain.resume()."
            )
        else:
            # Existing failed/completed path: start at the *next* node.
            resume_idx = None
            for i, nid in enumerate(order):
                if nid == latest.node_id:
                    resume_idx = i + 1
                    break
            start_node = order[resume_idx] if resume_idx and resume_idx < len(order) else None

        # A resumed chain is a run, and a run gets a root span — the same
        # correction ``Swarm.aresume`` got in 1.67.0. ``resume`` calls
        # ``execute_chain`` directly, so without this its node spans were
        # emitted as orphan roots and the returned ``ChainResult`` had no trace
        # to name.
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import trace_id_of

        with get_tracer().start_as_current_span(f"chain.{self.name}") as span:
            span.set_attribute("chain.name", self.name)
            span.set_attribute("chain.resumed_execution_id", execution_id)
            span.set_attribute("fastaiagent.runner.type", "chain")
            span.set_attribute("fastaiagent.framework", "fastaiagent")
            trace_id = trace_id_of(span)
            try:
                raw = await execute_chain(
                    nodes=self.nodes,
                    edges=self.edges,
                    initial_state=state,
                    state_schema=self.state_schema,
                    checkpointer=store,
                    chain_name=self.name,
                    execution_id=execution_id,
                    resume_from_node=start_node,
                    resume_value=resume_value,
                    run_context=context,
                    strict_routing=self.strict_routing,
                )
            except BaseException as exc:
                write_run_end(
                    store,
                    execution_id=execution_id,
                    chain_name=self.name,
                    status="failed",
                    error=exc,
                )
                raise
            # A resumed run that reaches the end has ended just as much as one
            # that never paused, and earns the same marker. Missing this was the
            # gap that left every HITL-approved run looking unfinished on the
            # plane — ``resume`` calls ``execute_chain`` directly, bypassing
            # ``aexecute``.
            if raw.get("status") == "completed":
                write_run_end(
                    store,
                    execution_id=execution_id,
                    chain_name=self.name,
                    status="completed",
                    state_snapshot=raw.get("final_state"),
                )

        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
            status=raw.get("status", "completed"),
            pending_interrupt=raw.get("pending_interrupt"),
            trace_id=trace_id,
        )

    # Alias matching the ``aresume()`` contract that ``Agent`` / ``Swarm`` /
    # ``Supervisor`` expose. Lets the v1.0 HTTP / CLI resume entrypoints
    # treat all four runner types uniformly.
    async def aresume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        modified_state: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> ChainResult:
        """Async alias for :meth:`resume` (matches the Agent/Swarm/Supervisor surface)."""
        return await self.resume(
            execution_id,
            modified_state=modified_state,
            resume_value=resume_value,
            context=context,
        )

    async def afork(
        self,
        execution_id: str,
        *,
        checkpoint_id: str | None = None,
        input: Any | None = None,
        modified_state: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> ChainResult:
        """Fork a run from a saved checkpoint into a NEW, independent execution.

        Unlike :meth:`resume` (which continues the *same* ``execution_id``),
        ``afork`` branches from the chosen checkpoint's state under a **fresh**
        ``execution_id`` — the original run is left completely intact. Pass
        ``checkpoint_id`` to branch from a specific step (use
        ``checkpointer.list(execution_id)`` to find ids); omit it to branch from
        the last **executed step** — the run-end marker every finished run has
        written since 1.65.0 is stepped over, because it names no node to run
        forward from. ``input`` / ``modified_state`` patch the restored state so
        the branch diverges; the chain then runs forward from the node *after*
        the fork point.

        A run held only by the plane is pulled down first, so a fork works on a
        machine that never saw the original run — the same restore-anywhere
        behaviour :meth:`resume` has.

        Returns a :class:`ChainResult` whose ``execution_id`` is the new forked
        id. The fork's lineage links back to the source via
        ``parent_checkpoint_id``.

        This is the SDK's checkpoint-fork primitive. Trace-based counterfactual
        replay (re-deriving a run from ingested spans) is the Enterprise plane's
        job, not the SDK's — see :mod:`fastaiagent.trace.replay` for the local
        read-only inspect/diff surface.
        """
        from fastaiagent.chain.checkpoint import Checkpoint
        from fastaiagent.chain.executor import _topological_sort

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()
        # Restore-anywhere (audit D4). ``resume`` has done this since 1.65.0 and
        # ``fork`` never did, so forking a plane-held run on a fresh machine
        # failed even though the state was there to be had.
        from fastaiagent.checkpointers.platform_replica import restore_if_missing

        restore_if_missing(store, execution_id)
        # NOT ``latest_resumable``: that one raises AlreadyResumed on a finished
        # run, which is the case fork exists to serve. See ``latest_forkable``.
        base = latest_forkable(store, execution_id, checkpoint_id=checkpoint_id)
        if base is None:
            from fastaiagent._internal.errors import ChainCheckpointError

            raise ChainCheckpointError(
                f"No checkpoint found to fork for execution '{execution_id}'"
                + (f" / checkpoint '{checkpoint_id}'" if checkpoint_id else "")
                + " — the run holds no executed step to branch from."
            )

        # Restore the checkpoint's state, then apply the fork's modifications so
        # the branch diverges.
        state = dict(base.state_snapshot)
        if input is not None:
            state["input"] = input
        if modified_state:
            state.update(modified_state)

        # Run forward from the node AFTER the fork point — exactly like a
        # failed/completed resume, but under a fresh execution_id.
        order = _topological_sort(self.nodes, self.edges)
        start_node: str | None = None
        known_node = base.node_id in order
        if known_node:
            i = order.index(base.node_id)
            start_node = order[i + 1] if i + 1 < len(order) else None
        if start_node is None:
            from fastaiagent._internal.errors import ChainResumeError

            hint = (
                f"Pick an earlier step: chain.fork('{execution_id}', "
                f"checkpoint_id=<id>), with ids from "
                f"checkpointer.list('{execution_id}')."
            )
            if known_node:
                raise ChainResumeError(
                    f"Cannot fork execution '{execution_id}' from node "
                    f"'{base.node_id}': it is the last node in the chain, so nothing "
                    f"downstream would run. {hint}"
                )
            raise ChainResumeError(
                f"Cannot fork execution '{execution_id}' from checkpoint "
                f"'{base.node_id}': it does not name a node of chain '{self.name}', "
                f"so there is nothing to run forward from. {hint}"
            )

        fork_id = str(uuid.uuid4())
        # Lineage marker: one origin checkpoint under the new id pointing back at
        # the source. Its distinct node_id never collides with a real node or
        # confuses get_last()/resume of the fork.
        store.put(
            Checkpoint(
                checkpoint_id=str(uuid.uuid4()),
                parent_checkpoint_id=base.checkpoint_id or None,
                chain_name=self.name,
                execution_id=fork_id,
                node_id="__fork_origin__",
                node_index=base.node_index,
                step_type="fork_origin",
                status="completed",
                state_snapshot=dict(state),
            )
        )

        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import trace_id_of

        # Same reasoning as ``resume``: a fork is its own run under a fresh
        # execution id, so it gets its own root span and its own trace id.
        with get_tracer().start_as_current_span(f"chain.{self.name}") as span:
            span.set_attribute("chain.name", self.name)
            span.set_attribute("chain.forked_execution_id", fork_id)
            span.set_attribute("fastaiagent.runner.type", "chain")
            span.set_attribute("fastaiagent.framework", "fastaiagent")
            trace_id = trace_id_of(span)
            # A fork is a run, and a run that ends leaves a tombstone — the same
            # rule ``aexecute`` and ``resume`` have followed since the durability
            # audit (D5). ``afork`` had neither half: a branch that died left no
            # ``run_end`` row at all, so it was indistinguishable from one that
            # simply stopped and ``latest_resumable`` handed back its last real
            # checkpoint. Both markers are best-effort inside ``write_run_end``,
            # so a checkpointer problem can never replace the exception below.
            try:
                raw = await execute_chain(
                    nodes=self.nodes,
                    edges=self.edges,
                    initial_state=state,
                    state_schema=self.state_schema,
                    checkpointer=store,
                    chain_name=self.name,
                    execution_id=fork_id,
                    resume_from_node=start_node,
                    run_context=context,
                    strict_routing=self.strict_routing,
                )
            except BaseException as exc:
                write_run_end_once(
                    store,
                    execution_id=fork_id,
                    chain_name=self.name,
                    status="failed",
                    error=exc,
                )
                raise
            # Only a run that actually ENDED gets one. A paused branch has not
            # ended, and a row after its ``interrupted`` one would hide the pause
            # from ``resume``'s status guard.
            if raw.get("status") == "completed":
                write_run_end_once(
                    store,
                    execution_id=fork_id,
                    chain_name=self.name,
                    status="completed",
                    state_snapshot=raw.get("final_state"),
                )
        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
            status=raw.get("status", "completed"),
            pending_interrupt=raw.get("pending_interrupt"),
            trace_id=trace_id,
        )

    def fork(
        self,
        execution_id: str,
        *,
        checkpoint_id: str | None = None,
        input: Any | None = None,
        modified_state: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> ChainResult:
        """Sync wrapper for :meth:`afork`."""
        return run_sync(
            self.afork(
                execution_id,
                checkpoint_id=checkpoint_id,
                input=input,
                modified_state=modified_state,
                context=context,
            )
        )

    def as_mcp_server(
        self,
        transport: str = "stdio",
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Any:
        """Expose this chain as an MCP server.

        Returns a :class:`fastaiagent.tool.mcp_server.FastAIAgentMCPServer`.
        Call ``await server.run()`` to start the stdio loop.

        Requires ``pip install 'fastaiagent[mcp-server]'``.

        Example::

            chain.as_mcp_server(transport="stdio").run()
        """
        from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

        return FastAIAgentMCPServer(
            target=self,
            transport=transport,  # type: ignore[arg-type]
            expose_tools=False,
            expose_system_prompt=False,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format (ReactFlow-compatible)."""
        d: dict[str, Any] = {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.state_schema:
            d["state_schema"] = self.state_schema
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chain:
        """Deserialize from canonical format."""
        chain = cls(
            name=data["name"],
            state_schema=data.get("state_schema"),
        )
        chain.nodes = [NodeConfig.from_dict(n) for n in data.get("nodes", [])]
        chain.edges = [Edge.from_dict(e) for e in data.get("edges", [])]
        return chain
