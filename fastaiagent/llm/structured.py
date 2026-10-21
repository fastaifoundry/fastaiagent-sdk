"""Structured-output resolution: schema building, strict transform, parsing.

``OutputSpec`` turns an Agent's ``output_type`` into a ``response_format`` and a
parser. It supports any Pydantic-compatible type via ``TypeAdapter`` — a
``BaseModel``, a primitive (``int``/``str``/...), ``list[Model]``, unions, etc.

The ``json_schema`` response_format only permits an *object* at the top level,
so non-object types are wrapped in a one-field object (``{"value": ...}``) and
transparently unwrapped on parse. A plain ``BaseModel`` (already an object) takes
the original path unchanged, so its emitted schema is byte-identical to before.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

# The single field a non-object output is wrapped in for json_schema mode.
_WRAP_KEY = "value"

# Anthropic (and other prompt-fallback providers) sometimes wrap JSON in a
# markdown fence; strip it before parsing.
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from an LLM response."""
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _is_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _wrap_schema(inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap a non-object schema in a single ``value`` property, hoisting ``$defs``."""
    inner = dict(inner)
    defs = inner.pop("$defs", None)
    wrapper: dict[str, Any] = {
        "type": "object",
        "properties": {_WRAP_KEY: inner},
        "required": [_WRAP_KEY],
    }
    if defs is not None:
        wrapper["$defs"] = defs
    return wrapper


def _make_strict(node: Any) -> Any:
    """Transform a JSON schema in place into OpenAI strict-mode shape.

    Every object gets ``additionalProperties: false`` and *all* its properties
    listed in ``required`` (OpenAI strict requires this; optional fields are
    already nullable in the Pydantic-generated schema). ``default`` keywords —
    which strict mode rejects — are removed. ``$ref``/``$defs`` are preserved.
    """
    if isinstance(node, dict):
        node.pop("default", None)
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _make_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_strict(item)
    return node


#: JSON Schema keywords Anthropic's ``output_config.format`` refuses (structured
#: outputs page, read 2026-10-19). They move into the description, and the reply
#: is still validated against the full type locally.
_ANTHROPIC_UNSUPPORTED = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "maxItems",
    "uniqueItems",
)


def anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` in the shape Anthropic's native structured output takes
    (1.91.0): every object closed (``additionalProperties: false``), and the
    constraints it does not support written into the field's description."""

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            moved = [f"{key}={node.pop(key)}" for key in _ANTHROPIC_UNSUPPORTED if key in node]
            min_items = node.get("minItems")
            if isinstance(min_items, int) and min_items > 1:
                moved.append(f"minItems={node.pop('minItems')}")
            if moved:
                note = "Constraints: " + ", ".join(moved) + "."
                existing = node.get("description")
                node["description"] = f"{existing} {note}" if existing else note
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)
        return node

    return visit(copy.deepcopy(schema))  # type: ignore[no-any-return]


def partial_json(text: str) -> Any | None:
    """The JSON in ``text`` so far, parsed as incomplete JSON (1.91.0), or
    ``None`` when ``text`` is not the start of a JSON object or array — prose
    before a tool call, an empty reply. A leading code fence is skipped."""
    from pydantic_core import from_json

    body = text.lstrip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0] if body.rstrip().endswith("```") else body
        body = body.lstrip()
    if not body.startswith(("{", "[")):
        return None
    try:
        return from_json(body, allow_partial="trailing-strings")
    except ValueError:
        return None


class OutputSpec:
    """Resolved structured-output plan for one ``output_type`` (built once)."""

    def __init__(self, output_type: Any) -> None:
        self.output_type = output_type
        self.adapter: TypeAdapter[Any] = TypeAdapter(output_type)

        if _is_model(output_type):
            inner_schema = output_type.model_json_schema()
        else:
            inner_schema = self.adapter.json_schema()

        # response_format json_schema needs an object at the top level.
        self.wrapped = inner_schema.get("type") != "object"
        self.schema = _wrap_schema(inner_schema) if self.wrapped else inner_schema

        raw_name = getattr(output_type, "__name__", None) or "Response"
        self.name = _NAME_RE.sub("_", raw_name) or "Response"

    def response_format(self, *, strict: bool) -> dict[str, Any]:
        """Build the ``response_format`` dict; apply strict transform if asked."""
        if strict:
            schema = _make_strict(copy.deepcopy(self.schema))
            return {
                "type": "json_schema",
                "json_schema": {"name": self.name, "schema": schema, "strict": True},
            }
        return {
            "type": "json_schema",
            "json_schema": {"name": self.name, "schema": self.schema},
        }

    def parse(self, text: str) -> tuple[Any | None, str | None]:
        """Parse ``text`` into the output type. Returns ``(value, error)``.

        ``error`` is a short human-readable reason on failure (used to re-ask the
        model), or ``None`` on success.
        """
        clean = _strip_code_fences(text)
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            return None, f"the response was not valid JSON ({e})"
        if self.wrapped and isinstance(data, dict) and _WRAP_KEY in data:
            data = data[_WRAP_KEY]
        try:
            return self.adapter.validate_python(data), None
        except ValidationError as e:
            return None, f"the response did not match the required schema ({e})"
