"""Chain class — directed graph workflow with cycles, typed state, and checkpointing."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.chain.executor import execute_chain
from fastaiagent.chain.interrupt import AlreadyResumed, Resume
from fastaiagent.chain.node import Edge, Node, NodeConfig, NodeType
from fastaiagent.chain.validator import validate_chain
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer

logger = logging.getLogger(__name__)


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
        type: NodeType = NodeType.agent,
        name: str = "",
        *,
        node: Node | None = None,
        output_key: str | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        **config: Any,
    ) -> Chain:
        """Add a node to the chain.

        Pass ``node=`` a :func:`fastaiagent.node`-decorated function to add a
        typed, code-first node. ``output_key`` stores the node's output under a
        named state key (instead of the legacy ``_<id>_output`` wrap), and
        ``input_schema`` / ``output_schema`` (optional JSON schemas) validate the
        node's resolved inputs / output at its boundary. All additive — a node
        without any of these behaves exactly as before.
        """
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
        # Stash the 2.4b extras into ``config`` so they ride the existing
        # NodeConfig serialization and the executor can read them per node.
        if output_key is not None:
            config["output_key"] = output_key
        if input_schema is not None:
            config["input_schema"] = input_schema
        if output_schema is not None:
            config["output_schema"] = output_schema
        node_config = NodeConfig(
            id=id,
            type=type,
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
                if initial_state:
                    import json

                    try:
                        span.set_attribute("chain.input", json.dumps(initial_state, default=str))
                    except (TypeError, ValueError):
                        logger.debug("Failed to serialize chain input for trace", exc_info=True)

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

            if span is not None:
                try:
                    import json as _json

                    span.set_attribute("chain.output", _json.dumps(raw.get("output"), default=str))
                except (TypeError, ValueError):
                    logger.debug("Failed to serialize chain output for trace", exc_info=True)
                span.set_attribute("chain.execution_id", raw.get("execution_id") or "")

        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
            status=raw.get("status", "completed"),
            pending_interrupt=raw.get("pending_interrupt"),
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
        latest = store.get_last(execution_id)
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

        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
            status=raw.get("status", "completed"),
            pending_interrupt=raw.get("pending_interrupt"),
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
        the last checkpoint. ``input`` / ``modified_state`` patch the restored
        state so the branch diverges; the chain then runs forward from the node
        *after* the fork point.

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
        base = (
            store.get_by_id(execution_id, checkpoint_id)
            if checkpoint_id is not None
            else store.get_last(execution_id)
        )
        if base is None:
            from fastaiagent._internal.errors import ChainCheckpointError

            raise ChainCheckpointError(
                f"No checkpoint found to fork for execution '{execution_id}'"
                + (f" / checkpoint '{checkpoint_id}'" if checkpoint_id else "")
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
        for i, nid in enumerate(order):
            if nid == base.node_id:
                start_node = order[i + 1] if i + 1 < len(order) else None
                break
        if start_node is None:
            from fastaiagent._internal.errors import ChainResumeError

            raise ChainResumeError(
                f"Cannot fork execution '{execution_id}' from node "
                f"'{base.node_id}': it is the final node, so nothing downstream "
                "would run. Fork from an earlier checkpoint."
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
                status="completed",
                state_snapshot=dict(state),
            )
        )

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
        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
            status=raw.get("status", "completed"),
            pending_interrupt=raw.get("pending_interrupt"),
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
