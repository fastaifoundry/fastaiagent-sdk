"""Chain node types and configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

ParallelFailureMode = Literal["continue", "fail_fast", "any_success"]


class NodeType(str, Enum):
    """Type of a chain node."""

    agent = "agent"
    tool = "tool"
    condition = "condition"
    parallel = "parallel"
    hitl = "hitl"
    start = "start"
    end = "end"
    transformer = "transformer"


class NodeConfig(BaseModel):
    """Configuration for a chain node."""

    id: str
    type: NodeType = NodeType.agent
    name: str = ""
    agent: Any = None  # Agent instance (not serialized directly)
    agent_name: str | None = None
    tool: Any = None  # Tool instance
    tool_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    parallel_failure_mode: ParallelFailureMode = "continue"
    """Failure semantics for ``NodeType.parallel`` nodes.

    * ``"continue"`` (default, backwards-compatible): collect all results.
      Exceptions become ``{"error": str(e)}`` entries in ``outputs``.
    * ``"fail_fast"``: cancel siblings on first exception; raise to the chain.
    * ``"any_success"``: return only the successful outputs; raise
      ``ChainError`` if every child failed.

    Ignored on non-parallel node types.
    """

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "label": self.name,
            "position": self.position,
            "config": self.config,
        }
        if self.agent_name:
            d["config"]["agent_name"] = self.agent_name
        if self.tool_name:
            d["config"]["tool_name"] = self.tool_name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeConfig:
        config = data.get("config", {})
        return cls(
            id=data["id"],
            type=NodeType(data.get("type", "agent")),
            name=data.get("label", data.get("name", "")),
            agent_name=config.get("agent_name"),
            tool_name=config.get("tool_name"),
            config=config,
            position=data.get("position", {"x": 0, "y": 0}),
        )


class Edge(BaseModel):
    """An edge connecting two nodes."""

    id: str = ""
    source: str
    target: str
    label: str = ""
    condition: str | None = None
    is_cyclic: bool = False
    cycle_config: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
        }
        if self.id:
            d["id"] = self.id
        if self.label:
            d["label"] = self.label
        if self.condition:
            d["condition"] = self.condition
        if self.is_cyclic:
            d["is_cyclic"] = True
            d["cycle_config"] = self.cycle_config
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(
            id=data.get("id", ""),
            source=data["source"],
            target=data["target"],
            label=data.get("label", ""),
            condition=data.get("condition"),
            is_cyclic=data.get("is_cyclic", False),
            cycle_config=data.get("cycle_config", {}),
        )


@dataclass
class Node:
    """A code-first, typed chain node produced by :func:`node`.

    Add it with ``chain.add_node(id, node=<this>)``. It runs as a tool node:
    the function's type hints generate the ``input_schema`` (validated at the
    node boundary), ``output_key`` stores the function's return under a named
    state key, and an optional ``output_schema`` validates the return.
    """

    name: str
    fn: Callable[..., Any]
    tool: Any  # FunctionTool built from fn
    output_key: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


def node(
    name: str | None = None,
    *,
    output_key: str | None = None,
    output_schema: dict[str, Any] | None = None,
    validate_input: bool = True,
) -> Callable[[Callable[..., Any]], Node]:
    """Decorator: turn a function into a reusable, typed chain node.

    Example::

        @node(output_key="category")
        def classify(text: str) -> str:
            return "support" if "help" in text else "sales"

        chain.add_node(
            "classify", node=classify, input_mapping={"text": "{{state.input}}"}
        )

    The function's type hints generate the input schema (validated at the node
    boundary when ``validate_input``). ``output_key`` stores the function's
    return under that state key (instead of the legacy ``_<id>_output`` wrap);
    ``output_schema`` (an optional JSON schema) validates the return.

    All additive: a chain that doesn't use ``@node`` / schemas / ``output_key``
    behaves exactly as before.
    """

    def deco(fn: Callable[..., Any]) -> Node:
        from fastaiagent.tool.function import FunctionTool

        nm = name or fn.__name__
        ft = FunctionTool(name=nm, fn=fn)
        return Node(
            name=nm,
            fn=fn,
            tool=ft,
            output_key=output_key,
            input_schema=ft.parameters if validate_input else None,
            output_schema=output_schema,
        )

    return deco
