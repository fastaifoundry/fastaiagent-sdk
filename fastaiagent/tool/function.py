"""FunctionTool — wraps a Python callable as a tool."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, get_type_hints

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool.base import Tool, ToolResult


def _python_type_to_json_schema(tp: type) -> dict:
    """Convert a Python type annotation to JSON Schema type."""
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        args = getattr(tp, "__args__", ())
        items = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": items}
    return mapping.get(tp, {"type": "string"})


def _generate_schema(fn: Callable) -> dict:
    """Generate JSON Schema parameters from function type hints."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        tp = hints.get(param_name, str)
        prop = _python_type_to_json_schema(tp)

        # Use docstring or param name as description
        prop["description"] = param_name

        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class FunctionTool(Tool):
    """A tool that wraps a Python callable.

    Auto-generates JSON Schema from type hints.

    Example:
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = FunctionTool(name="greet", fn=greet)
        result = tool.execute({"name": "World"})
    """

    def __init__(
        self,
        name: str,
        fn: Callable | None = None,
        description: str = "",
        parameters: dict | None = None,
    ):
        self.fn = fn
        if fn and not description:
            description = inspect.getdoc(fn) or ""
        if fn and parameters is None:
            parameters = _generate_schema(fn)
        super().__init__(name=name, description=description, parameters=parameters)

    async def aexecute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the wrapped function."""
        if self.fn is None:
            return ToolResult(error="No function attached to this tool")
        try:
            result = self.fn(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return ToolResult(output=result)
        except Exception as e:
            raise ToolExecutionError(f"Tool '{self.name}' failed: {e}") from e

    def _tool_type(self) -> str:
        return "function"

    def _config_dict(self) -> dict:
        return {}

    @classmethod
    def _from_dict(cls, data: dict) -> FunctionTool:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters"),
        )


def tool(name: str | None = None, description: str = "") -> Callable:
    """Decorator to create a FunctionTool from a function.

    Example:
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"
    """

    def decorator(fn: Callable) -> FunctionTool:
        tool_name = name or fn.__name__
        tool_desc = description or inspect.getdoc(fn) or ""
        return FunctionTool(name=tool_name, fn=fn, description=tool_desc)

    return decorator
