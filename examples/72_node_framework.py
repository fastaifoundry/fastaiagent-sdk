"""Example 72: Code-first chain nodes — @node, typed I/O, output_key.

Write a node as a plain function: its type hints become a validated input
schema, ``output_key`` names where its result lands in state, and an optional
``output_schema`` validates the return. All additive — chains that don't use
these behave exactly as before.

Runnable as pytest (no API keys, no network):
    pytest examples/72_node_framework.py -v
"""

from __future__ import annotations

import pytest

from fastaiagent import Chain, node
from fastaiagent._internal.errors import ChainError


@node(output_key="category")
def classify(text: str) -> str:
    """Route a message to a queue."""
    return "support" if "help" in text.lower() else "sales"


@node(
    output_key="ticket",
    output_schema={
        "type": "object",
        "properties": {"id": {"type": "string"}, "queue": {"type": "string"}},
        "required": ["id", "queue"],
    },
)
def open_ticket(category: str) -> dict:
    return {"id": "T-100", "queue": category}


def test_typed_two_node_chain() -> None:
    chain = Chain("router", checkpoint_enabled=False)
    chain.add_node("classify", node=classify, input_mapping={"text": "{{state.input}}"})
    chain.add_node("open", node=open_ticket, input_mapping={"category": "{{state.category}}"})
    chain.connect("classify", "open")

    res = chain.execute({"input": "I need help with my order"})
    assert res.final_state["category"] == "support"
    assert res.final_state["ticket"] == {"id": "T-100", "queue": "support"}


def test_output_schema_is_enforced() -> None:
    @node(output_schema={"type": "object", "required": ["id"]})
    def bad(x: str) -> str:
        return "oops-not-a-dict"

    chain = Chain("bad", checkpoint_enabled=False)
    chain.add_node("bad", node=bad, input_mapping={"x": "{{state.input}}"})
    with pytest.raises(ChainError):
        chain.execute({"input": "hi"})
