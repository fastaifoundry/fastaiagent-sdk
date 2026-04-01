"""Tool base class and ToolResult."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of a tool execution."""

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


class Tool:
    """Base class for all tools.

    Subclasses: FunctionTool, RESTTool, MCPTool.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool synchronously."""
        import asyncio

        return asyncio.run(self.aexecute(arguments))

    async def aexecute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool asynchronously. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement aexecute()")

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> dict:
        """Serialize to canonical format."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self._tool_type(),
            "parameters": self.parameters,
            "config": self._config_dict(),
        }

    def _tool_type(self) -> str:
        return "base"

    def _config_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, data: dict) -> Tool:
        """Deserialize from canonical format — dispatches to correct subclass."""
        from fastaiagent.tool.function import FunctionTool
        from fastaiagent.tool.mcp import MCPTool
        from fastaiagent.tool.rest import RESTTool

        tool_type = data.get("tool_type", "function")
        dispatch = {
            "function": FunctionTool,
            "rest_api": RESTTool,
            "mcp": MCPTool,
        }
        target_cls = dispatch.get(tool_type)
        if target_cls is None:
            return cls(
                name=data["name"],
                description=data.get("description", ""),
                parameters=data.get("parameters"),
            )
        return target_cls._from_dict(data)
