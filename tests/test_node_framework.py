"""2.4b — code-first node framework: @node, typed I/O, explicit output_key.

No mocks, no LLM: deterministic function/tool nodes over a real Chain. Every
feature is additive — a chain that uses none of them behaves exactly as before.
"""

from __future__ import annotations

import pytest

from fastaiagent import Chain, FunctionTool, node
from fastaiagent._internal.errors import ChainError
from fastaiagent.chain.node import Node, NodeType


def _chain(name: str = "c") -> Chain:
    return Chain(name, checkpoint_enabled=False)


def _double(x: str) -> int:
    return int(x) * 2


def _echo(text: str = "") -> str:
    return text


class TestNodeDecorator:
    def test_decorator_builds_typed_node(self):
        @node(output_key="category")
        def classify(text: str) -> str:
            return "support" if "help" in text else "sales"

        assert isinstance(classify, Node)
        assert classify.name == "classify"
        assert classify.output_key == "category"
        # Input schema is auto-derived from the function's type hints.
        assert classify.input_schema["properties"]["text"]["type"] == "string"

    def test_output_key_stores_unwrapped_output(self):
        @node(output_key="category")
        def classify(text: str) -> str:
            return "support" if "help" in text else "sales"

        ch = _chain()
        ch.add_node("classify", node=classify, input_mapping={"text": "{{state.input}}"})
        res = ch.execute({"input": "I need help"})
        assert res.final_state["category"] == "support"

    def test_output_key_works_on_a_plain_tool_node(self):
        # output_key is available on any node, not just @node ones.
        ch = _chain()
        ch.add_node(
            "dbl",
            tool=FunctionTool(name="dbl", fn=_double),
            type=NodeType.tool,
            input_mapping={"x": "{{state.n}}"},
            output_key="doubled",
        )
        res = ch.execute({"n": 5})
        assert res.final_state["doubled"] == 10


class TestSchemas:
    _USER_SCHEMA = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }

    def test_output_schema_pass(self):
        @node(output_key="user", output_schema=self._USER_SCHEMA)
        def make_user(name: str) -> dict:
            return {"id": f"u-{name}"}

        ch = _chain()
        ch.add_node("mk", node=make_user, input_mapping={"name": "{{state.input}}"})
        res = ch.execute({"input": "alice"})
        assert res.final_state["user"] == {"id": "u-alice"}

    def test_output_schema_violation_raises(self):
        @node(output_schema=self._USER_SCHEMA)
        def bad(name: str) -> str:
            return "not-a-dict"

        ch = _chain()
        ch.add_node("bad", node=bad, input_mapping={"name": "{{state.input}}"})
        with pytest.raises(ChainError, match="output_schema"):
            ch.execute({"input": "x"})

    def test_input_schema_violation_raises(self):
        ch = _chain()
        ch.add_node(
            "need_two",
            tool=FunctionTool(name="need_two", fn=_echo),
            type=NodeType.tool,
            input_mapping={"text": "{{state.input}}"},  # provides 'text' but not 'n'
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["text", "n"],
            },
        )
        with pytest.raises(ChainError, match="input_schema"):
            ch.execute({"input": "x"})


class TestBackwardCompat:
    def test_plain_chain_is_unchanged(self):
        # No @node / schema / output_key -> the legacy dict-merge behavior, so a
        # tool node's {"output": ...} still lands at state["output"].
        ch = _chain()
        ch.add_node(
            "n1",
            tool=FunctionTool(name="d", fn=_double),
            type=NodeType.tool,
            input_mapping={"x": "{{state.start}}"},
        )
        res = ch.execute({"start": 3})
        assert res.final_state["output"] == 6
