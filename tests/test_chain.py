"""Tests for fastaiagent.chain module."""

from __future__ import annotations

import pytest

from fastaiagent import SQLiteCheckpointer
from fastaiagent._internal.errors import (
    ChainStateValidationError,
)
from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, ChainResult, ChainState, NodeType
from fastaiagent.chain.node import Edge, NodeConfig
from fastaiagent.chain.validator import detect_cycles, validate_chain
from fastaiagent.llm.client import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(self, response_text: str = "mock output"):
        super().__init__(provider="mock", model="mock")
        self._response_text = response_text
        self._call_count = 0

    async def acomplete(self, messages, tools=None, **kwargs):
        self._call_count += 1
        return LLMResponse(content=self._response_text, finish_reason="stop")


def _make_agent(name: str, response: str = "result") -> Agent:
    return Agent(name=name, llm=MockLLMClient(response), system_prompt="test")


# --- ChainState tests ---


class TestChainState:
    def test_basic_operations(self):
        state = ChainState({"x": 1, "y": 2})
        assert state.get("x") == 1
        state.set("z", 3)
        assert state["z"] == 3
        state.update({"x": 10})
        assert state["x"] == 10

    def test_snapshot_and_restore(self):
        state = ChainState({"a": 1, "b": [1, 2, 3]})
        snap = state.snapshot()
        restored = ChainState.from_snapshot(snap)
        assert restored.data == state.data
        # Verify deep copy
        snap["b"].append(4)
        assert len(state.get("b")) == 3

    def test_validate_passes(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        state = ChainState({"name": "Alice"})
        state.validate(schema)  # should not raise

    def test_validate_fails(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        state = ChainState({"age": 30})
        with pytest.raises(ChainStateValidationError, match="validation failed"):
            state.validate(schema)

    def test_contains(self):
        state = ChainState({"x": 1})
        assert "x" in state
        assert "y" not in state


# --- Validator tests ---


class TestValidator:
    def test_detect_no_cycles(self):
        nodes = [NodeConfig(id="a"), NodeConfig(id="b"), NodeConfig(id="c")]
        edges = [
            Edge(source="a", target="b"),
            Edge(source="b", target="c"),
        ]
        cycles = detect_cycles(nodes, edges)
        assert len(cycles) == 0

    def test_detect_cycle(self):
        nodes = [NodeConfig(id="a"), NodeConfig(id="b")]
        edges = [
            Edge(source="a", target="b"),
            Edge(source="b", target="a"),
        ]
        cycles = detect_cycles(nodes, edges)
        assert len(cycles) > 0

    def test_validate_valid_chain(self):
        nodes = [NodeConfig(id="a"), NodeConfig(id="b")]
        edges = [Edge(source="a", target="b")]
        errors = validate_chain(nodes, edges)
        assert len(errors) == 0

    def test_validate_missing_target(self):
        nodes = [NodeConfig(id="a")]
        edges = [Edge(source="a", target="missing")]
        errors = validate_chain(nodes, edges)
        assert any("missing" in e for e in errors)

    def test_validate_cyclic_without_max_iterations(self):
        nodes = [NodeConfig(id="a"), NodeConfig(id="b")]
        edges = [Edge(source="a", target="b", is_cyclic=True, cycle_config={})]
        errors = validate_chain(nodes, edges)
        assert any("max_iterations" in e for e in errors)


# --- Chain execution tests ---


class TestChainExecution:
    @pytest.mark.asyncio
    async def test_linear_chain(self):
        """A → B → C linear chain."""
        chain = Chain("linear", checkpoint_enabled=False)
        chain.add_node("a", agent=_make_agent("a", "result_a"))
        chain.add_node("b", agent=_make_agent("b", "result_b"))
        chain.add_node("c", agent=_make_agent("c", "result_c"))
        chain.connect("a", "b")
        chain.connect("b", "c")

        result = await chain.aexecute({"input": "start"})
        assert isinstance(result, ChainResult)
        assert result.execution_id
        assert "a" in result.node_results
        assert "b" in result.node_results
        assert "c" in result.node_results

    @pytest.mark.asyncio
    async def test_chain_with_typed_state(self):
        """Chain validates state at each step."""
        schema = {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "quality": {"type": "number"},
            },
        }
        chain = Chain("typed", state_schema=schema, checkpoint_enabled=False)
        chain.add_node("a", agent=_make_agent("a"))
        result = await chain.aexecute({"input": "hello", "quality": 0.5})
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_chain_state_validation_failure(self):
        """Chain raises on invalid initial state."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        chain = Chain("strict", state_schema=schema, checkpoint_enabled=False)
        chain.add_node("a", agent=_make_agent("a"))

        with pytest.raises(ChainStateValidationError):
            await chain.aexecute({"wrong_field": "hello"})

    @pytest.mark.asyncio
    async def test_transformer_node(self):
        """Transformer node renders templates."""
        chain = Chain("transform", checkpoint_enabled=False)
        chain.add_node(
            "t",
            type=NodeType.transformer,
            template="Hello {{input}}!",
        )

        result = await chain.aexecute({"input": "World"})
        assert "World" in str(result.node_results.get("t", {}))

    @pytest.mark.asyncio
    async def test_hitl_node_auto_approved(self):
        """HITL node auto-approves when no handler is set."""
        chain = Chain("hitl", checkpoint_enabled=False)
        chain.add_node("approval", type=NodeType.hitl)
        result = await chain.aexecute({})
        assert result.node_results["approval"]["approved"] is True

    @pytest.mark.asyncio
    async def test_hitl_with_handler(self):
        """HITL node calls the handler."""

        def handler(node, context, state):
            return True

        chain = Chain("hitl", checkpoint_enabled=False)
        chain.add_node("approval", type=NodeType.hitl)
        result = await chain.aexecute({}, hitl_handler=handler)
        assert result.node_results["approval"]["approved"] is True


# --- Checkpoint tests ---


class TestCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoints_saved(self, temp_dir):
        """Chain saves checkpoints after each node."""
        store = SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
        chain = Chain("cp-test", checkpointer=store)
        chain.add_node("a", agent=_make_agent("a"))
        chain.add_node("b", agent=_make_agent("b"))
        chain.connect("a", "b")

        result = await chain.aexecute({"input": "test"})

        checkpoints = store.list(result.execution_id)
        assert len(checkpoints) == 2
        assert checkpoints[0].node_id == "a"
        assert checkpoints[1].node_id == "b"
        store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_get_latest(self, temp_dir):
        store = SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
        chain = Chain("cp-test", checkpointer=store)
        chain.add_node("a", agent=_make_agent("a"))
        chain.add_node("b", agent=_make_agent("b"))
        chain.connect("a", "b")

        result = await chain.aexecute({"input": "test"})
        latest = store.get_last(result.execution_id)
        assert latest is not None
        assert latest.node_id == "b"
        store.close()


# --- Chain serialization tests ---


class TestChainSerialization:
    def test_to_dict(self):
        chain = Chain("test-chain")
        chain.add_node("a", type=NodeType.agent, name="Agent A")
        chain.add_node("b", type=NodeType.agent, name="Agent B")
        chain.connect("a", "b")
        chain.connect("b", "a", max_iterations=3, exit_condition="done == true")

        d = chain.to_dict()
        assert d["name"] == "test-chain"
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 2

        cyclic_edges = [e for e in d["edges"] if e.get("is_cyclic")]
        assert len(cyclic_edges) == 1
        assert cyclic_edges[0]["cycle_config"]["max_iterations"] == 3

    def test_from_dict(self):
        data = {
            "name": "restored",
            "nodes": [
                {"id": "a", "type": "agent", "label": "A", "config": {}},
                {"id": "b", "type": "agent", "label": "B", "config": {}},
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {
                    "source": "b",
                    "target": "a",
                    "is_cyclic": True,
                    "cycle_config": {"max_iterations": 5},
                },
            ],
        }
        chain = Chain.from_dict(data)
        assert chain.name == "restored"
        assert len(chain.nodes) == 2
        assert len(chain.edges) == 2

    def test_roundtrip(self):
        chain = Chain("roundtrip", state_schema={"type": "object"})
        chain.add_node("x", name="Node X")
        chain.add_node("y", name="Node Y")
        chain.connect("x", "y")

        d = chain.to_dict()
        restored = Chain.from_dict(d)
        d2 = restored.to_dict()
        assert d["name"] == d2["name"]
        assert len(d["nodes"]) == len(d2["nodes"])
        assert len(d["edges"]) == len(d2["edges"])
        assert d.get("state_schema") == d2.get("state_schema")

    def test_validate_method(self):
        chain = Chain("valid")
        chain.add_node("a")
        chain.add_node("b")
        chain.connect("a", "b")
        errors = chain.validate()
        assert len(errors) == 0
