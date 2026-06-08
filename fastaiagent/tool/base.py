"""Tool base class and ToolResult."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync

# Replay-safety classes drive the central Replay engine's inject-vs-execute
# decision per tool call. ``side_effecting`` is the safe default: an unmarked
# tool is never re-executed during replay (its recorded output is injected).
# Marks are explicit only — a "GET" REST tool is NOT auto-classified read_only;
# auto-inferring a re-executable class would violate the replay-safety invariant.
_ALLOWED_REPLAY_CLASSES = ("read_only", "idempotent", "side_effecting")
_DEFAULT_REPLAY_CLASS = "side_effecting"


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

    # Display origin for the Local UI. Subclasses override to one of
    # "function" / "mcp" / "rest". ``LocalKB.as_tool()`` overrides the
    # instance to "kb". Anything left at "custom" is a user-defined Tool
    # subclass. Surfaces on /agents/<name> as a colored chip so users can
    # tell at a glance whether a tool came from a decorator, an MCP server,
    # a REST spec, a knowledge base, or their own code.
    origin: str = "custom"

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        replay_class: str | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        # Resolve the safe default *before* validating, so unset always passes.
        # Strict: an explicit out-of-set value is a developer error and raises
        # loudly here (the authoring boundary). The wire/replay layer stays
        # lenient and coerces unknown values to ``side_effecting`` at read time.
        resolved = _DEFAULT_REPLAY_CLASS if replay_class is None else replay_class
        if resolved not in _ALLOWED_REPLAY_CLASSES:
            raise ValueError(
                f"replay_class must be one of {_ALLOWED_REPLAY_CLASSES}, "
                f"got {replay_class!r}"
            )
        self.replay_class = resolved

    def execute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        """Execute the tool synchronously."""
        return run_sync(self.aexecute(arguments, context=context))

    async def aexecute(
        self,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> ToolResult:
        """Execute the tool asynchronously. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement aexecute()")

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_schema(self) -> dict[str, Any]:
        """Convert to MCP tool-schema shape (``name`` / ``description`` / ``inputSchema``).

        Used by :class:`fastaiagent.tool.mcp_server.FastAIAgentMCPServer` when
        an agent's tools are exposed individually (``expose_tools=True``).
        """
        params = self.parameters or {"type": "object", "properties": {}}
        if "type" not in params:
            params = {"type": "object", **params}
        return {
            "name": self.name,
            "description": self.description or self.name,
            "inputSchema": params,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self._tool_type(),
            "origin": self.origin,
            "parameters": self.parameters,
            "replay_class": self.replay_class,
            "config": self._config_dict(),
        }

    def _tool_type(self) -> str:
        return "base"

    def _config_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tool:
        """Deserialize from canonical format — dispatches to correct subclass."""
        from fastaiagent.tool.function import FunctionTool
        from fastaiagent.tool.mcp import MCPTool
        from fastaiagent.tool.rest import RESTTool

        tool_type = data.get("tool_type", "function")
        dispatch: dict[str, type[Tool]] = {
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
                replay_class=data.get("replay_class", _DEFAULT_REPLAY_CLASS),
            )
        result: Tool = target_cls._from_dict(data)  # type: ignore[attr-defined]
        return result
