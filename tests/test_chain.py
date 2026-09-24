"""Tests for fastaiagent.chain module."""

from __future__ import annotations

import pytest

from fastaiagent import SQLiteCheckpointer
from fastaiagent._internal.errors import (
    ChainStateValidationError,
)
from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, ChainResult, ChainState, NodeType
from fastaiagent.chain.checkpoint import latest_resumable
from fastaiagent.chain.interrupt import AlreadyResumed
from fastaiagent.chain.node import Edge, NodeConfig
from fastaiagent.chain.validator import detect_cycles, validate_chain
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.tool.base import Tool, ToolResult
from fastaiagent.tool.function import FunctionTool


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
        # Agents attached: since 1.67.0 ``validate()`` also asks whether each
        # node has anything to run, and an agent node with no agent is exactly
        # the configuration that used to complete a run having done nothing.
        nodes = [
            NodeConfig(id="a", agent=_make_agent("a")),
            NodeConfig(id="b", agent=_make_agent("b")),
        ]
        edges = [Edge(source="a", target="b")]
        errors = validate_chain(nodes, edges)
        assert errors == []

    def test_validate_flags_an_agent_node_with_no_agent(self):
        """The design-time half of the 1.67.0 rule.

        Structural validation used to be all there was, so a chain whose every
        node was empty validated clean and then ran to ``completed``.
        """
        nodes = [NodeConfig(id="a", agent=_make_agent("a")), NodeConfig(id="b")]
        edges = [Edge(source="a", target="b")]
        errors = validate_chain(nodes, edges)
        assert any("'b'" in e and "no agent attached" in e for e in errors)
        assert not any("'a'" in e for e in errors)

    def test_validate_does_not_confuse_a_payload_error_for_a_routing_one(self):
        """Callers filter these strings; the two families must stay separable."""
        nodes = [NodeConfig(id="a")]
        errors = validate_chain(nodes, [])
        assert errors and not any("handle" in e or "default" in e for e in errors)

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


# --- add_node: what gets attached, and what silently did not ---


class TestAddNodeAttachment:
    """``add_node`` used to absorb every unrecognised keyword into ``**config``
    and default the node's type to ``agent``.

    So ``chain.add_node("fetch", tool=my_tool)`` — the form the docs' own Node
    Types table taught — built an **agent** node with ``agent=None``, and the run
    reported ``completed`` with ``{"error": "No agent attached…"}`` as that
    node's output. These pin the three ways that is now closed.
    """

    def test_tool_without_an_explicit_type_builds_a_tool_node(self):
        from fastaiagent.tool.function import FunctionTool

        chain = Chain("infer")
        chain.add_node("fetch", tool=FunctionTool(name="fetch", fn=lambda: "x"))

        node = chain.nodes[0]
        assert node.type is NodeType.tool
        assert node.tool is not None
        assert node.tool_name == "fetch"
        assert chain.validate() == []

    def test_an_explicit_type_still_wins_over_inference(self):
        from fastaiagent.tool.function import FunctionTool

        chain = Chain("explicit")
        chain.add_node(
            "n",
            tool=FunctionTool(name="t", fn=lambda: "x"),
            type=NodeType.transformer,
            template="hi",
        )
        assert chain.nodes[0].type is NodeType.transformer

    def test_agent_and_tool_together_still_mean_an_agent_node(self):
        """Backwards compatibility, deliberately: this shape already worked, and
        inference must not quietly re-route it to the tool."""
        from fastaiagent.tool.function import FunctionTool

        chain = Chain("both")
        chain.add_node("n", agent=_make_agent("a"), tool=FunctionTool(name="t", fn=lambda: "x"))
        assert chain.nodes[0].type is NodeType.agent

    def test_a_bare_callable_as_tool_is_refused_at_add_node(self):
        """It used to reach the executor and die with ``AttributeError:
        'function' object has no attribute 'aexecute'`` — loud, but eight frames
        away from the line that caused it."""
        chain = Chain("bare")
        with pytest.raises(TypeError) as excinfo:
            chain.add_node("n", tool=lambda x: x, type=NodeType.tool)
        assert "FunctionTool" in str(excinfo.value)
        assert chain.nodes == []

    @pytest.mark.parametrize("kwarg", ["fn", "function", "func", "callable"])
    def test_callable_lookalike_kwargs_are_refused(self, kwarg):
        chain = Chain("lookalike")
        with pytest.raises(TypeError) as excinfo:
            chain.add_node("n", **{kwarg: lambda x: x})
        assert kwarg in str(excinfo.value)
        assert "tool=" in str(excinfo.value)
        assert chain.nodes == []

    def test_a_node_decorated_function_passed_as_tool_points_at_node(self):
        from fastaiagent.chain.node import node as node_deco

        @node_deco()
        def classify(text: str) -> str:
            return text

        chain = Chain("nodearg")
        with pytest.raises(TypeError, match="node="):
            chain.add_node("n", tool=classify)

    def test_an_unknown_keyword_warns_but_is_still_carried(self, caplog):
        """``**config`` is load-bearing for input_mapping/template/conditions/
        agents and for a chain's own metadata, so this is a warning, not a
        refusal. A typo like ``agnet=`` is the case it is for."""
        import logging

        chain = Chain("unknown")
        with caplog.at_level(logging.WARNING, logger="fastaiagent.chain.chain"):
            chain.add_node("n", agent=_make_agent("a"), retries=3)
        assert "retries" in caplog.text
        assert chain.nodes[0].config["retries"] == 3


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
    async def test_hitl_node_with_no_handler_refuses(self):
        """An approval gate nobody can answer must not answer itself.

        Until 1.67.0 an unconfigured gate returned
        ``{"approved": True, "message": "Auto-approved (no HITL handler)"}`` and
        the run reported ``completed`` — a control that could not run reporting a
        clean pass, in the one node type whose entire job is to stop things.
        """
        from fastaiagent._internal.errors import ChainError

        chain = Chain("hitl", checkpoint_enabled=False)
        chain.add_node("approval", type=NodeType.hitl)
        with pytest.raises(ChainError, match="no handler"):
            await chain.aexecute({})

    @pytest.mark.asyncio
    async def test_hitl_node_auto_approves_when_asked_to(self):
        """Local-dev convenience survives — it just has to be asked for."""
        chain = Chain("hitl", checkpoint_enabled=False)
        chain.add_node("approval", type=NodeType.hitl, auto_approve=True)
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


# --- Tool node failure (backlog #5, 1.78.0) ---


class _RefusingTool(Tool):
    """A tool that reports failure by *returning* an error, as MCPTool does for
    an ``isError`` reply — no exception crosses the tool boundary."""

    async def aexecute(self, arguments, context=None):
        return ToolResult(error="upstream refused the request")


def _charge_chain(ran: list[str], **chain_kwargs) -> Chain:
    """prepare → charge(amount: int) → receipt, with the amount taken from state.

    ``input_mapping`` renders templates as strings, so a non-numeric
    ``state.amount`` fails ``charge``'s argument validation before it runs.
    """

    def prepare() -> str:
        ran.append("prepare")
        return "ready"

    def charge(amount: int) -> str:
        ran.append("charge")
        return f"charged {amount}"

    def receipt() -> str:
        ran.append("receipt")
        return "receipt sent"

    chain = Chain("charge-flow", **chain_kwargs)
    chain.add_node("prepare", tool=FunctionTool(name="prepare", fn=prepare))
    chain.add_node(
        "charge",
        tool=FunctionTool(name="charge", fn=charge),
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node("receipt", tool=FunctionTool(name="receipt", fn=receipt))
    chain.connect("prepare", "charge")
    chain.connect("charge", "receipt")
    return chain


class TestToolNodeFailure:
    """A tool node whose tool could not run fails the run.

    Until 1.78.0 a tool that *returned* an error — its arguments failed
    validation, say — was stored as the node's result: the downstream nodes ran
    on ``output=None`` and the run reported ``completed``. A tool that *raised*
    already failed the run; the two were the same failure reported two ways.
    """

    @pytest.mark.asyncio
    async def test_invalid_arguments_fail_the_run_and_stop_downstream(self):
        from fastaiagent._internal.errors import ChainError

        ran: list[str] = []
        chain = _charge_chain(ran, checkpoint_enabled=False)
        with pytest.raises(ChainError, match=r"Tool node 'charge' \(charge\) failed: Invalid"):
            await chain.aexecute({"amount": "abc"})
        # The tool never ran, and nothing after it did either — no receipt for
        # a charge that did not happen.
        assert ran == ["prepare"]

    @pytest.mark.asyncio
    async def test_failed_run_is_marked_failed_and_stays_resumable(self, temp_dir):
        from fastaiagent._internal.errors import ChainError

        store = SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
        ran: list[str] = []
        chain = _charge_chain(ran, checkpointer=store)
        with pytest.raises(ChainError):
            await chain.aexecute({"amount": "abc"}, execution_id="charge-run")

        rows = store.list("charge-run")
        assert [c.step_type for c in rows] == ["node", "run_end"]
        assert rows[-1].status == "failed"
        # ``completed`` would refuse a resume (AlreadyResumed); ``failed`` hands
        # back the last node that finished, so the run can be fixed and re-entered.
        resumable = latest_resumable(store, "charge-run")
        assert resumable is not None and resumable.node_id == "prepare"
        store.close()

    @pytest.mark.asyncio
    async def test_a_returned_tool_error_fails_the_run(self):
        from fastaiagent._internal.errors import ChainError

        chain = Chain("refused", checkpoint_enabled=False)
        chain.add_node("call", tool=_RefusingTool(name="remote"))
        with pytest.raises(ChainError, match="upstream refused the request"):
            await chain.aexecute({})

    @pytest.mark.asyncio
    async def test_valid_arguments_complete_with_the_same_result_shape(self):
        ran: list[str] = []
        chain = _charge_chain(ran, checkpoint_enabled=False)
        result = await chain.aexecute({"amount": "42"})
        assert result.status == "completed"
        assert ran == ["prepare", "charge", "receipt"]
        assert result.node_results["charge"] == {"output": "charged 42", "error": None}


# --- Tool policy inside a chain (backlog #16, 1.78.0) ---


class TestToolNodePolicy:
    """A tool node applies the tool's own timeout / max_retries / output_type.

    The agent loop calls ``tool.ainvoke`` (the policy-aware entry point); until
    1.78.0 a chain tool node called ``tool.aexecute``, which deliberately skips
    it — so the same tool timed out, retried and validated inside an agent and
    did none of that inside a chain.
    """

    @pytest.mark.asyncio
    async def test_timeout_applies(self):
        import asyncio

        from fastaiagent._internal.errors import ToolExecutionError

        async def slow() -> str:
            await asyncio.sleep(0.5)
            return "late"

        chain = Chain("slow", checkpoint_enabled=False)
        chain.add_node("call", tool=FunctionTool(name="slow", fn=slow, timeout=0.05))
        with pytest.raises(ToolExecutionError, match="timed out"):
            await chain.aexecute({})

    @pytest.mark.asyncio
    async def test_max_retries_applies(self):
        attempts: list[int] = []

        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("upstream blip")
            return "ok"

        chain = Chain("flaky", checkpoint_enabled=False)
        chain.add_node(
            "call", tool=FunctionTool(name="flaky", fn=flaky, max_retries=2, retry_delay=0)
        )
        result = await chain.aexecute({})
        assert result.status == "completed"
        assert len(attempts) == 3
        assert result.node_results["call"]["output"] == "ok"

    @pytest.mark.asyncio
    async def test_output_type_mismatch_fails_the_run(self):
        from fastaiagent._internal.errors import ChainError

        def count() -> str:
            return "not a number"

        chain = Chain("typed", checkpoint_enabled=False)
        chain.add_node("call", tool=FunctionTool(name="count", fn=count, output_type=int))
        with pytest.raises(ChainError, match="output failed schema validation"):
            await chain.aexecute({})

    @pytest.mark.asyncio
    async def test_output_type_stores_the_validated_json_form(self, temp_dir):
        from pydantic import BaseModel

        class Ticket(BaseModel):
            id: int
            title: str

        def open_ticket() -> dict:
            return {"id": "7", "title": "refund"}

        store = SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
        chain = Chain("tickets", checkpointer=store)
        chain.add_node(
            "open",
            tool=FunctionTool(name="open_ticket", fn=open_ticket, output_type=Ticket),
            output_key="ticket",
        )
        result = await chain.aexecute({}, execution_id="t-1")

        # Coerced by output_type ("7" -> 7), stored as JSON-safe data rather than
        # a Ticket instance, so the checkpoint (plain json.dumps) still writes.
        assert result.status == "completed"
        assert result.final_state["ticket"] == {"id": 7, "title": "refund"}
        rows = store.list("t-1")
        assert rows[-1].step_type == "run_end" and rows[-1].status == "completed"
        store.close()


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
        # Two node checkpoints, then the run-end marker (audit D5) — the row
        # that lets anyone downstream tell a finished run from one that died
        # right after its last node.
        assert len(checkpoints) == 3
        assert checkpoints[0].node_id == "a"
        assert checkpoints[1].node_id == "b"
        assert [c.step_type for c in checkpoints] == ["node", "node", "run_end"]
        assert checkpoints[2].status == "completed"
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
        # The newest row is now the run-end marker, not the last node. That is
        # the point of D5: "the last node completed" and "the run finished" were
        # the same row, so neither the plane nor resume could tell them apart.
        assert latest.step_type == "run_end"
        assert latest.status == "completed"
        # The last re-entry point is still node "b" — the marker is a tombstone,
        # never something a resume restarts at.
        with pytest.raises(AlreadyResumed):
            latest_resumable(store, result.execution_id)
        resumable = [c for c in store.list(result.execution_id) if c.step_type != "run_end"]
        assert resumable[-1].node_id == "b"
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
        chain.add_node("a", agent=_make_agent("a"))
        chain.add_node("b", agent=_make_agent("b"))
        chain.connect("a", "b")
        errors = chain.validate()
        assert errors == []
