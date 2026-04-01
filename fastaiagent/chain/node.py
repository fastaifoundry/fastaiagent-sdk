"""Chain node types and configuration."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
