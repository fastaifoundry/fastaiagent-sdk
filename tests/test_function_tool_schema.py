"""Schema generation for FunctionTool — hybrid simple/Pydantic path.

These run without API keys. They lock in two guarantees:
  1. Primitive-only signatures emit byte-identical schemas (no drift for
     existing tools).
  2. Rich types (Pydantic models, Enum, Literal, Optional, nested) produce a
     proper self-contained JSON Schema with ``$defs``/``$ref``.

Models/enums are defined at module scope on purpose: with PEP 563 stringized
annotations, ``get_type_hints`` resolves names against the function's module
globals, which is exactly how users declare tool argument types.
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel

from fastaiagent.agent.context import RunContext
from fastaiagent.tool.function import FunctionTool, _generate_schema


class Priority(str, enum.Enum):
    low = "low"
    high = "high"


class Ticket(BaseModel):
    title: str
    body: str


class Addr(BaseModel):
    street: str


class Payload(BaseModel):
    x: int


def test_primitive_only_schema_is_unchanged() -> None:
    """The simple path must match the original hand-rolled output exactly."""

    def weather(city: str, days: int = 3):
        """Get weather.

        Args:
            city: the city name
            days: number of days
        """
        return city

    schema = _generate_schema(weather)
    assert schema == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "the city name"},
            "days": {"type": "integer", "description": "number of days"},
        },
        "required": ["city"],
    }


def test_list_of_primitives_stays_simple() -> None:
    def tagger(tags: list[str]):
        return tags

    schema = _generate_schema(tagger)
    assert schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "tags",
    }
    assert "$defs" not in schema


def test_pydantic_model_param_emits_defs_and_ref() -> None:
    def create_ticket(ticket: Ticket):
        """Create a ticket.

        Args:
            ticket: the ticket to create
        """
        return ticket

    schema = _generate_schema(create_ticket)
    assert schema["properties"]["ticket"]["$ref"] == "#/$defs/Ticket"
    assert schema["properties"]["ticket"]["description"] == "the ticket to create"
    # A field literally named "title" must survive title-stripping.
    assert set(schema["$defs"]["Ticket"]["properties"]) == {"title", "body"}
    assert schema["required"] == ["ticket"]


def test_enum_literal_and_optional() -> None:
    def act(priority: Priority, kind: Literal["bug", "feat"], note: str | None = None):
        return priority

    schema = _generate_schema(act)
    # Enum -> $ref into a $defs enum entry.
    assert schema["properties"]["priority"]["$ref"] == "#/$defs/Priority"
    assert schema["$defs"]["Priority"]["enum"] == ["low", "high"]
    # Literal -> enum inline.
    assert schema["properties"]["kind"]["enum"] == ["bug", "feat"]
    # Optional -> nullable via anyOf and not required.
    assert {"type": "null"} in schema["properties"]["note"]["anyOf"]
    assert "note" not in schema.get("required", [])
    assert set(schema["required"]) == {"priority", "kind"}


def test_titles_are_stripped() -> None:
    def f(addr: Addr):
        return addr

    schema = _generate_schema(f)
    assert "title" not in schema
    assert "title" not in schema["$defs"]["Addr"]


def test_runcontext_param_excluded_in_rich_path() -> None:
    def f(payload: Payload, ctx: RunContext):
        return payload

    schema = _generate_schema(f)
    assert "payload" in schema["properties"]
    assert "ctx" not in schema["properties"]


def test_function_tool_uses_generated_schema() -> None:
    def f(payload: Payload):
        return payload

    tool = FunctionTool(name="schema_test_f", fn=f)
    assert tool.parameters["properties"]["payload"]["$ref"] == "#/$defs/Payload"
    # OpenAI wire format carries the schema verbatim.
    assert tool.to_openai_format()["function"]["parameters"] == tool.parameters
