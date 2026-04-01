"""Chain class — directed graph workflow with cycles, typed state, and checkpointing."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent.chain.checkpoint import CheckpointStore
from fastaiagent.chain.executor import execute_chain
from fastaiagent.chain.node import Edge, NodeConfig, NodeType
from fastaiagent.chain.validator import validate_chain


class ChainResult(BaseModel):
    """Result of a chain execution."""

    output: Any = None
    final_state: dict[str, Any] = Field(default_factory=dict)
    execution_id: str = ""
    node_results: dict[str, Any] = Field(default_factory=dict)

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
        state_schema: dict | None = None,
        checkpoint_enabled: bool = True,
        checkpoint_store: CheckpointStore | None = None,
    ):
        self.name = name
        self.state_schema = state_schema
        self.nodes: list[NodeConfig] = []
        self.edges: list[Edge] = []
        self.checkpoint_enabled = checkpoint_enabled
        self._checkpoint_store = checkpoint_store

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
        return asyncio.run(
            self.aexecute(initial_state, trace=trace, **kwargs)
        )

    async def aexecute(
        self,
        initial_state: dict[str, Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        hitl_handler: Any = None,
        **kwargs: Any,
    ) -> ChainResult:
        """Async execution of the chain."""
        store = None
        if self.checkpoint_enabled:
            store = self._checkpoint_store or CheckpointStore()

        raw = await execute_chain(
            nodes=self.nodes,
            edges=self.edges,
            initial_state=initial_state or {},
            state_schema=self.state_schema,
            checkpoint_store=store,
            chain_name=self.name,
            execution_id=execution_id,
            hitl_handler=hitl_handler,
        )

        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
        )

    async def resume(
        self, execution_id: str, modified_state: dict | None = None
    ) -> ChainResult:
        """Resume a failed/paused chain execution from the last checkpoint."""
        store = self._checkpoint_store or CheckpointStore()
        latest = store.get_latest(execution_id)
        if latest is None:
            from fastaiagent._internal.errors import ChainCheckpointError

            raise ChainCheckpointError(
                f"No checkpoint found for execution '{execution_id}'"
            )

        state = latest.state_snapshot
        if modified_state:
            state.update(modified_state)

        # Find the next node after the checkpoint
        from fastaiagent.chain.executor import _topological_sort

        order = _topological_sort(self.nodes, self.edges)
        resume_idx = None
        for i, nid in enumerate(order):
            if nid == latest.node_id:
                resume_idx = i + 1
                break

        next_node = order[resume_idx] if resume_idx and resume_idx < len(order) else None

        raw = await execute_chain(
            nodes=self.nodes,
            edges=self.edges,
            initial_state=state,
            state_schema=self.state_schema,
            checkpoint_store=store,
            chain_name=self.name,
            execution_id=execution_id,
            resume_from_node=next_node,
        )

        return ChainResult(
            output=raw["output"],
            final_state=raw["final_state"],
            execution_id=raw["execution_id"],
            node_results=raw["node_results"],
        )

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> Chain:
        """Deserialize from canonical format."""
        chain = cls(
            name=data["name"],
            state_schema=data.get("state_schema"),
        )
        chain.nodes = [NodeConfig.from_dict(n) for n in data.get("nodes", [])]
        chain.edges = [Edge.from_dict(e) for e in data.get("edges", [])]
        return chain
