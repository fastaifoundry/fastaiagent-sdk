"""FunctionTool — wraps a Python callable as a tool."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from typing import Any, get_origin, get_type_hints

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.agent.context import RunContext
from fastaiagent.tool.base import Tool, ToolResult

logger = logging.getLogger(__name__)

_ARGS_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r"^\s*\w[\w\s]*:\s*$")
_PARAM_LINE_RE = re.compile(
    r"^\s{2,}(\w+)"  # indented param name
    r"(?:\s*\([^)]*\))?"  # optional (type)
    r"\s*:\s*"  # colon separator
    r"(.+)",  # description start
)


def _parse_param_descriptions(fn: Callable[..., Any]) -> dict[str, str]:
    """Extract parameter descriptions from Google-style docstring Args section."""
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    lines = doc.splitlines()
    result: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        if _ARGS_SECTION_RE.match(line):
            in_args = True
            continue

        if not in_args:
            continue

        # Check if we've hit a new section header (not indented or less indented)
        stripped = line.strip()
        if stripped and not line.startswith(" ") and not line.startswith("\t"):
            # Non-indented non-empty line means new section
            break
        if _SECTION_HEADER_RE.match(line) and not _PARAM_LINE_RE.match(line):
            break

        param_match = _PARAM_LINE_RE.match(line)
        if param_match:
            # Save previous param
            if current_param is not None:
                result[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            current_desc = [param_match.group(2).strip()]
        elif current_param is not None and stripped:
            # Continuation line for current param
            current_desc.append(stripped)

    # Save last param
    if current_param is not None:
        result[current_param] = " ".join(current_desc).strip()

    return result


def _is_context_param(annotation: Any) -> bool:
    """Check if a type annotation is RunContext or RunContext[T]."""
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    return False


def _python_type_to_json_schema(tp: type) -> dict[str, Any]:
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


def _generate_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema parameters from function type hints."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        logger.debug("Failed to get type hints for function %r", fn, exc_info=True)
        hints = {}

    param_docs = _parse_param_descriptions(fn)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        tp = hints.get(param_name, str)

        # Skip context parameters — these are injected, not from LLM
        if _is_context_param(tp):
            continue

        prop = _python_type_to_json_schema(tp)

        # Use docstring description if available, fall back to param name
        prop["description"] = param_docs.get(param_name, param_name)

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

    origin = "function"

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ):
        self.fn = fn
        self._context_param_name: str | None = None

        if fn:
            if not description:
                description = inspect.getdoc(fn) or ""
            if parameters is None:
                parameters = _generate_schema(fn)
            # Detect context parameter at init time (not per-call)
            self._context_param_name = self._detect_context_param(fn)

        super().__init__(name=name, description=description, parameters=parameters)

        # Auto-register callable-backed tools so ForkedReplay.arerun can rebind
        # them by name after reconstruction from span attributes.
        if fn is not None:
            from fastaiagent.tool.registry import ToolRegistry

            ToolRegistry.register(self)

    @staticmethod
    def _detect_context_param(fn: Callable[..., Any]) -> str | None:
        """Find the parameter name annotated as RunContext, if any."""
        try:
            hints = get_type_hints(fn)
        except Exception:
            logger.debug("Failed to get type hints for context param detection", exc_info=True)
            return None
        for param_name, annotation in hints.items():
            if _is_context_param(annotation):
                return param_name
        return None

    async def aexecute(
        self,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> ToolResult:
        """Execute the wrapped function, injecting context if declared."""
        if self.fn is None:
            return ToolResult(error="No function attached to this tool")
        try:
            call_args = dict(arguments)

            # Inject context if the function declares a context parameter
            if self._context_param_name is not None and context is not None:
                call_args[self._context_param_name] = context

            result = self.fn(**call_args)
            if inspect.isawaitable(result):
                result = await result
            return ToolResult(output=result)
        except Exception as e:
            # Control-flow signals from interrupt() / nested agent
            # suspension / claim-once-resume propagate through tool
            # boundaries unchanged so the parent executor (or the user)
            # can handle them at the right level.
            from fastaiagent.agent.executor import _AgentInterrupted
            from fastaiagent.chain.interrupt import AlreadyResumed, InterruptSignal

            if isinstance(e, InterruptSignal | _AgentInterrupted | AlreadyResumed):
                raise
            raise ToolExecutionError(f"Tool '{self.name}' failed: {e}") from e

    def _tool_type(self) -> str:
        return "function"

    def _config_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FunctionTool:
        # Prefer a live registered tool so replay reruns actually execute code.
        from fastaiagent.tool.registry import ToolRegistry

        name = data["name"]
        registered = ToolRegistry.get(name)
        if isinstance(registered, cls):
            return registered

        import logging

        logging.getLogger(__name__).warning(
            "FunctionTool '%s' not found in ToolRegistry — reconstructed without "
            "callable. Reruns that invoke this tool will surface a 'no function "
            "attached' error to the agent.",
            name,
        )
        return cls(
            name=name,
            description=data.get("description", ""),
            parameters=data.get("parameters"),
        )


def tool(name: str | None = None, description: str = "") -> Callable[..., Any]:
    """Decorator to create a FunctionTool from a function.

    Example:
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"
    """

    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        tool_name = name or fn.__name__
        tool_desc = description or inspect.getdoc(fn) or ""
        return FunctionTool(name=tool_name, fn=fn, description=tool_desc)

    return decorator
