"""FunctionTool — wraps a Python callable as a tool."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

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

# NumPy style:
#
#     Parameters
#     ----------
#     x : int
#         Description of x.
#         Continues here.
#     y : str, optional
#         Description of y.
#
_NUMPY_HEADER_RE = re.compile(r"^\s*(Parameters|Args|Arguments)\s*$", re.IGNORECASE)
_NUMPY_UNDERLINE_RE = re.compile(r"^\s*-{3,}\s*$")
_NUMPY_PARAM_RE = re.compile(
    r"^\s*(\w+)"  # param name (no leading indent required)
    r"(?:\s*:\s*[^$]+)?"  # optional " : type"
    r"\s*$"
)

# Sphinx / reStructuredText style:
#
#     :param x: Description of x.
#     :type x: int
#     :param y: Description of y, may
#         continue on the next line.
#
_SPHINX_PARAM_RE = re.compile(r"^\s*:param\s+(\w+)\s*:\s*(.*)$")
_SPHINX_OTHER_FIELD_RE = re.compile(r"^\s*:\w+(?:\s+\w+)?\s*:")


def _parse_google_style(lines: list[str]) -> dict[str, str]:
    """Google-style ``Args:`` block. Returns ``{}`` if no section found."""
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

        stripped = line.strip()
        if stripped and not line.startswith(" ") and not line.startswith("\t"):
            break
        if _SECTION_HEADER_RE.match(line) and not _PARAM_LINE_RE.match(line):
            break

        param_match = _PARAM_LINE_RE.match(line)
        if param_match:
            if current_param is not None:
                result[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            current_desc = [param_match.group(2).strip()]
        elif current_param is not None and stripped:
            current_desc.append(stripped)

    if current_param is not None:
        result[current_param] = " ".join(current_desc).strip()
    return result


def _parse_numpy_style(lines: list[str]) -> dict[str, str]:
    """NumPy-style ``Parameters\\n----------`` block. Returns ``{}`` if no
    section found."""
    result: dict[str, str] = {}
    n = len(lines)
    i = 0
    while i < n - 1:
        if _NUMPY_HEADER_RE.match(lines[i]) and _NUMPY_UNDERLINE_RE.match(lines[i + 1]):
            i += 2
            current_param: str | None = None
            current_desc: list[str] = []
            current_indent = -1
            while i < n:
                line = lines[i]
                stripped = line.strip()
                if not stripped:
                    i += 1
                    continue
                # End of section: another header underline pattern means a
                # new top-level section. Detect ``Returns\n-------`` etc.
                if (
                    i + 1 < n
                    and _NUMPY_UNDERLINE_RE.match(lines[i + 1])
                    and not lines[i].startswith(" ")
                    and stripped.lower() not in {"parameters", "args", "arguments"}
                ):
                    break
                # Param line: name (or name : type), no leading indent.
                if not line.startswith((" ", "\t")) and _NUMPY_PARAM_RE.match(line):
                    if current_param is not None:
                        result[current_param] = " ".join(current_desc).strip()
                    name_match = _NUMPY_PARAM_RE.match(line)
                    assert name_match is not None
                    current_param = name_match.group(1)
                    current_desc = []
                    current_indent = -1
                    i += 1
                    continue
                # Continuation line (indented).
                if current_param is not None and (line.startswith(" ") or line.startswith("\t")):
                    indent = len(line) - len(line.lstrip())
                    if current_indent < 0:
                        current_indent = indent
                    if indent >= current_indent:
                        current_desc.append(stripped)
                        i += 1
                        continue
                break
            if current_param is not None:
                result[current_param] = " ".join(current_desc).strip()
            return result
        i += 1
    return result


def _parse_sphinx_style(lines: list[str]) -> dict[str, str]:
    """Sphinx/reST ``:param name: description`` fields. Returns ``{}`` if no
    fields found."""
    result: dict[str, str] = {}
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        param_match = _SPHINX_PARAM_RE.match(line)
        if param_match:
            if current_param is not None:
                result[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            current_desc = [param_match.group(2).strip()]
            continue
        # Another :something: field ends the current :param: block.
        if _SPHINX_OTHER_FIELD_RE.match(line):
            if current_param is not None:
                result[current_param] = " ".join(current_desc).strip()
                current_param = None
                current_desc = []
            continue
        if current_param is not None:
            stripped = line.strip()
            if stripped:
                current_desc.append(stripped)

    if current_param is not None:
        result[current_param] = " ".join(current_desc).strip()
    return result


def _parse_param_descriptions(fn: Callable[..., Any]) -> dict[str, str]:
    """Extract parameter descriptions from a function's docstring.

    Tries Google → NumPy → Sphinx in order; returns the first non-empty
    result. Behaviour for Google-style docstrings is unchanged from
    earlier versions; NumPy and Sphinx are added in v1.9.0.
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    lines = doc.splitlines()
    for parser in (_parse_google_style, _parse_numpy_style, _parse_sphinx_style):
        result = parser(lines)
        if result:
            return result
    return {}


def _is_context_param(annotation: Any) -> bool:
    """Check if a type annotation is RunContext or RunContext[T]."""
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    return False


_SIMPLE_TYPES = (str, int, float, bool, dict)


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


def _is_simple_annotation(tp: Any) -> bool:
    """True for the primitive annotations the hand-rolled schema path handles
    exactly — ``str/int/float/bool/dict``, bare ``list``, and ``list[<simple>]``.

    Everything else (Pydantic models, ``Enum``, ``Literal``, ``Union``/
    ``Optional``, typed ``dict[str, ...]``, nested generics, ``datetime``, …) is
    "rich" and routes through :func:`_build_pydantic_schema`. Keeping the simple
    set narrow guarantees existing primitive-only tools emit byte-identical
    schemas (no drift), while rich types gain a proper self-contained schema.
    """
    if tp in _SIMPLE_TYPES:
        return True
    if tp is list:
        return True
    if get_origin(tp) is list:
        args = get_args(tp)
        return not args or _is_simple_annotation(args[0])
    return False


def _strip_schema_titles(node: Any) -> None:
    """Recursively drop Pydantic's auto-generated ``title`` keys in place.

    Function-calling schemas don't need titles; removing them keeps the emitted
    JSON Schema compact and closer to the hand-rolled shape. ``$ref`` targets in
    ``$defs`` are keyed by model name, not ``title``, so this is ref-safe.
    """
    if isinstance(node, dict):
        # Only drop the schema ``title`` *annotation* (always a string). A
        # parameter literally named "title" appears as a properties key whose
        # value is a sub-schema (a dict), so it must be preserved.
        if isinstance(node.get("title"), str):
            node.pop("title")
        for value in node.values():
            _strip_schema_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_schema_titles(item)


def _build_pydantic_schema(
    llm_params: list[tuple[str, inspect.Parameter, Any]],
    param_docs: dict[str, str],
) -> dict[str, Any]:
    """Build a self-contained JSON Schema for a signature via Pydantic.

    Constructs one model from the LLM-facing parameters and emits
    ``{type, properties, required, $defs}`` — with ``$ref`` for nested models,
    ``enum`` for ``Enum``/``Literal``, and ``anyOf`` for ``Optional``/``Union``.
    Both OpenAI (strict tools) and Anthropic (``input_schema``) accept this shape.
    """
    fields: dict[str, Any] = {}
    for param_name, param, tp in llm_params:
        desc = param_docs.get(param_name, param_name)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (tp, Field(description=desc))
        else:
            fields[param_name] = (tp, Field(default=param.default, description=desc))

    model = create_model("ToolArgs", **fields)
    schema: dict[str, Any] = model.model_json_schema(ref_template="#/$defs/{model}")
    _strip_schema_titles(schema)
    return schema


def _build_arg_validator(
    fn: Callable[..., Any], context_param: str | None, allowed_names: set[str]
) -> type[BaseModel] | None:
    """Build a Pydantic model that validates/coerces this tool's LLM arguments.

    One field per typed parameter that the model can actually send — i.e. a key
    of the tool's JSON-Schema ``properties`` (``allowed_names``). This is the
    authoritative LLM contract: injected dependencies (``RunContext`` and the
    common ``_dep=<obj>`` default-argument DI pattern) are *not* in the schema
    and must be excluded, both because the model never provides them and because
    their values can be un-copyable (e.g. a checkpointer holding a thread-local).

    Optional params store a plain ``None`` default rather than the function's
    real default: coercion only reads back keys the model actually provided, so
    the validator's default value is never used — and keeping it simple avoids
    Pydantic deep-copying a complex default at model-construction time. Untyped
    params are skipped (passed through raw). Returns ``None`` when there's
    nothing to validate or the model can't be built.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        logger.debug("Failed to get type hints for arg validator on %r", fn, exc_info=True)
        return None

    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls") or param_name == context_param:
            continue
        if param_name not in allowed_names:
            continue  # not an LLM-facing argument (injected / DI default)
        tp = hints.get(param_name)
        if tp is None or _is_context_param(tp):
            continue  # untyped → no coercion, pass through as-is
        default = ... if param.default is inspect.Parameter.empty else None
        fields[param_name] = (tp, default)

    if not fields:
        return None
    try:
        model: type[BaseModel] = create_model(f"{fn.__name__}_ArgValidator", **fields)
        return model
    except Exception:
        logger.debug("Failed to build arg validator for %r", fn, exc_info=True)
        return None


def _generate_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema parameters from function type hints.

    Hybrid strategy: signatures whose LLM-facing parameters are all *simple*
    (see :func:`_is_simple_annotation`) use the original hand-rolled path and
    emit byte-identical schemas. If any parameter is a rich type (Pydantic
    model, ``Enum``, ``Literal``, ``Optional``/``Union``, nested generics, …)
    the whole signature is generated via Pydantic for a proper schema. Pydantic
    generation failures fall back to the simple path so this never raises.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        logger.debug("Failed to get type hints for function %r", fn, exc_info=True)
        hints = {}

    param_docs = _parse_param_descriptions(fn)

    # LLM-facing params only: drop self/cls and injected RunContext params.
    llm_params: list[tuple[str, inspect.Parameter, Any]] = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        tp = hints.get(param_name, str)
        if _is_context_param(tp):
            continue
        llm_params.append((param_name, param, tp))

    # Any rich type → Pydantic-quality schema for the whole signature.
    if any(not _is_simple_annotation(tp) for _, _, tp in llm_params):
        try:
            return _build_pydantic_schema(llm_params, param_docs)
        except Exception:
            logger.debug(
                "Pydantic schema generation failed for %r; falling back to simple path",
                fn,
                exc_info=True,
            )

    # Simple path — unchanged output for primitive-only signatures.
    properties = {}
    required = []
    for param_name, param, tp in llm_params:
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
        replay_class: str | None = None,
        *,
        validate_args: bool = True,
        timeout: float | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.5,
        output_type: Any | None = None,
    ):
        self.fn = fn
        self._context_param_name: str | None = None
        self.validate_args = validate_args
        self._arg_validator: type[BaseModel] | None = None

        if fn:
            if not description:
                description = inspect.getdoc(fn) or ""
            if parameters is None:
                parameters = _generate_schema(fn)
            # Detect context parameter at init time (not per-call)
            self._context_param_name = self._detect_context_param(fn)
            if validate_args:
                # Validate only the tool's declared LLM-facing arguments — the
                # keys of the (possibly caller-supplied) parameter schema.
                allowed_names = set((parameters or {}).get("properties", {}).keys())
                self._arg_validator = _build_arg_validator(
                    fn, self._context_param_name, allowed_names
                )

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            replay_class=replay_class,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            output_type=output_type,
        )

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

            # Validate + coerce LLM arguments against the function's type hints.
            # Only keys the model provided are coerced (so omitted optionals keep
            # the function's own defaults); a mismatch is fed back to the model
            # as an error instead of blowing up inside the function.
            if self._arg_validator is not None:
                validate_keys = [
                    k for k in call_args if k in self._arg_validator.model_fields
                ]
                try:
                    # Validate the provided keys; Pydantic also flags any
                    # required (no-default) field the model left out.
                    validated = self._arg_validator(
                        **{k: call_args[k] for k in validate_keys}
                    )
                except ValidationError as e:
                    return ToolResult(
                        error=f"Invalid arguments for tool '{self.name}': {e}"
                    )
                for k in validate_keys:
                    call_args[k] = getattr(validated, k)

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
            replay_class=data.get("replay_class", "side_effecting"),
        )


def tool(
    name: str | None = None,
    description: str = "",
    replay_class: str | None = None,
    *,
    validate_args: bool = True,
    timeout: float | None = None,
    max_retries: int = 0,
    retry_delay: float = 0.5,
    output_type: Any | None = None,
) -> Callable[..., Any]:
    """Decorator to create a FunctionTool from a function.

    ``replay_class`` marks the tool's replay-safety class
    (``read_only`` / ``idempotent`` / ``side_effecting``); unset resolves to the
    safe ``side_effecting`` default. It is never auto-inferred — only an explicit
    mark makes a tool re-executable in replay.

    Execution policy (all optional):
      - ``validate_args`` — validate/coerce the LLM's arguments against the
        function's type hints before calling it (default ``True``).
      - ``timeout`` — per-call wall-clock timeout in seconds.
      - ``max_retries`` / ``retry_delay`` — retry the call on failure, with
        exponential backoff (``retry_delay * 2**attempt``).
      - ``output_type`` — validate/coerce the return value against this type.

    Example:
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"
    """

    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        tool_name = name or fn.__name__
        tool_desc = description or inspect.getdoc(fn) or ""
        return FunctionTool(
            name=tool_name,
            fn=fn,
            description=tool_desc,
            replay_class=replay_class,
            validate_args=validate_args,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            output_type=output_type,
        )

    return decorator
