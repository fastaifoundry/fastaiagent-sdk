"""Chain class — directed graph workflow with cycles, typed state, and checkpointing."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.chain.executor import execute_chain
from fastaiagent.chain.interrupt import AlreadyResumed, Resume
from fastaiagent.chain.node import Edge, NodeConfig, NodeType
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
    ):
        self.name = name
        self.state_schema = state_schema
        self.nodes: list[NodeConfig] = []
        self.edges: list[Edge] = []
        self.checkpoint_enabled = checkpoint_enabled
        self._checkpointer = checkpointer

    def add_node(
        self,
        id: str,
        agent: Any = None,
        tool: Any = None,
        type: NodeType = NodeType.agent,
        name: str = "",
        **config: Any,
    ) -> Chain:
        """Add a node to the chain."""
        node = NodeConfig(
            id=id,
            type=type,
            name=name or id,
            agent=agent,
            agent_name=agent.name if agent and hasattr(agent, "name") else None,
            tool=tool,
            tool_name=tool.name if tool and hasattr(tool, "name") else None,
            config=config,
        )
        self.nodes.append(node)
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
        **kwargs: Any,
    ) -> ChainResult:
        """Synchronous execution."""
        return run_sync(self.aexecute(initial_state, trace=trace, **kwargs))

    async def aexecute(
        self,
        initial_state: dict[str, Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        hitl_handler: Any = None,
        **kwargs: Any,
    ) -> ChainResult:
        """Async execution of the chain."""
        store: Checkpointer | None = None
        if self.checkpoint_enabled:
            store = self._checkpointer or SQLiteCheckpointer()
            store.setup()

        # Wrap the whole chain in a root span so every child agent span is a
        # descendant of it — the UI can then render a chain as one trace with
        # a tree of agents, rather than N orphan agent traces.
        from fastaiagent.trace.otel import get_tracer

        tracer = get_tracer()
        with tracer.start_as_current_span(f"chain.{self.name}") as span:
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
            )

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
    ) -> ChainResult:
        """Resume a failed/paused chain execution from the last checkpoint.

        For an *interrupted* checkpoint (``interrupt()`` was called), pass a
        :class:`Resume` value. The resumer atomically claims the
        ``pending_interrupts`` row before starting; concurrent resumers see
        :class:`AlreadyResumed`.

        For a *failed* checkpoint, ``modified_state`` lets you patch state
        before the next node runs (the existing v0.x behavior).
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
            # the pending row — concurrent resumers see AlreadyResumed.
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
            from fastaiagent._internal.errors import ChainCheckpointError

            raise ChainCheckpointError(
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
    ) -> ChainResult:
        """Async alias for :meth:`resume` (matches the Agent/Swarm/Supervisor surface)."""
        return await self.resume(
            execution_id, modified_state=modified_state, resume_value=resume_value
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
