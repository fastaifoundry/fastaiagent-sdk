"""Conditional routing tests for the chain executor.

Covers the contract documented in ``docs/chains/index.md``:

* Edge ``condition=`` expressions select a single branch using
  ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``, ``contains``,
  ``startswith``.
* When all outgoing edges from a source are unconditional, every
  edge still fires (fan-out preserved).
* ``NodeType.condition`` nodes return ``{"matched": handle}``; the
  matching ``label`` edge wins, otherwise the unlabeled or
  ``"default"`` edge fires.
* The validator rejects condition nodes whose outgoing edges miss
  declared handles and sources with multiple defaults alongside
  conditional siblings.
"""

from __future__ import annotations

import pytest

from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.node import Edge, NodeConfig
from fastaiagent.chain.validator import validate_chain
from fastaiagent.llm.client import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(self, response_text: str = "mock output") -> None:
        super().__init__(provider="mock", model="mock")
        self._response_text = response_text

    async def acomplete(self, messages, tools=None, **kwargs):
        return LLMResponse(content=self._response_text, finish_reason="stop")


def _agent(name: str, response: str) -> Agent:
    """Agent that always returns ``response`` so ``node_results[name].output``
    is the value our routing conditions key off of."""
    return Agent(name=name, llm=MockLLMClient(response), system_prompt="test")


# -- Edge `condition=` routing ----------------------------------------------


class TestEdgeConditionRouting:
    @pytest.mark.asyncio
    async def test_equality_routes_first_match(self):
        chain = Chain("eq", checkpoint_enabled=False)
        chain.add_node("classify", agent=_agent("classify", "billing"))
        chain.add_node("billing", agent=_agent("billing", "served_billing"))
        chain.add_node("tech", agent=_agent("tech", "served_tech"))
        chain.add_node("general", agent=_agent("general", "served_general"))
        chain.connect(
            "classify",
            "billing",
            condition="{{node_results.classify.output}} == billing",
        )
        chain.connect(
            "classify",
            "tech",
            condition="{{node_results.classify.output}} == technical",
        )
        chain.connect("classify", "general")  # default

        result = await chain.aexecute({"input": "hi"})
        assert "billing" in result.node_results
        assert "tech" not in result.node_results
        assert "general" not in result.node_results

    @pytest.mark.asyncio
    async def test_default_fallback_when_no_condition_matches(self):
        chain = Chain("default", checkpoint_enabled=False)
        chain.add_node("classify", agent=_agent("classify", "weather"))
        chain.add_node("billing", agent=_agent("billing", "served_billing"))
        chain.add_node("general", agent=_agent("general", "served_general"))
        chain.connect(
            "classify",
            "billing",
            condition="{{node_results.classify.output}} == billing",
        )
        chain.connect("classify", "general")

        result = await chain.aexecute({"input": "hi"})
        assert "general" in result.node_results
        assert "billing" not in result.node_results

    @pytest.mark.asyncio
    async def test_numeric_ge_routing(self):
        chain = Chain("ge", checkpoint_enabled=False)
        chain.add_node("scorer", agent=_agent("scorer", "0.9"))
        chain.add_node("ship", agent=_agent("ship", "shipped"))
        chain.add_node("review", agent=_agent("review", "reviewed"))
        chain.connect(
            "scorer",
            "ship",
            condition="{{node_results.scorer.output}} >= 0.8",
        )
        chain.connect("scorer", "review")

        result = await chain.aexecute({"input": "hi"})
        assert "ship" in result.node_results
        assert "review" not in result.node_results

    @pytest.mark.asyncio
    async def test_numeric_lt_fallback(self):
        chain = Chain("lt", checkpoint_enabled=False)
        chain.add_node("scorer", agent=_agent("scorer", "0.3"))
        chain.add_node("ship", agent=_agent("ship", "shipped"))
        chain.add_node("review", agent=_agent("review", "reviewed"))
        chain.connect(
            "scorer",
            "ship",
            condition="{{node_results.scorer.output}} >= 0.8",
        )
        chain.connect("scorer", "review")

        result = await chain.aexecute({"input": "hi"})
        assert "review" in result.node_results
        assert "ship" not in result.node_results

    @pytest.mark.asyncio
    async def test_not_equal_routing(self):
        chain = Chain("neq", checkpoint_enabled=False)
        chain.add_node("tag", agent=_agent("tag", "open"))
        chain.add_node("close_path", agent=_agent("close_path", "closed"))
        chain.add_node("keep_open", agent=_agent("keep_open", "open"))
        chain.connect(
            "tag",
            "close_path",
            condition="{{node_results.tag.output}} != open",
        )
        chain.connect("tag", "keep_open")

        result = await chain.aexecute({"input": "hi"})
        assert "keep_open" in result.node_results
        assert "close_path" not in result.node_results

    @pytest.mark.asyncio
    async def test_contains_routing(self):
        chain = Chain("contains", checkpoint_enabled=False)
        chain.add_node("tag", agent=_agent("tag", "my order is late"))
        chain.add_node("urgent", agent=_agent("urgent", "u"))
        chain.add_node("normal", agent=_agent("normal", "n"))
        chain.connect(
            "tag",
            "urgent",
            condition="{{node_results.tag.output}} contains late",
        )
        chain.connect("tag", "normal")

        result = await chain.aexecute({"input": "hi"})
        assert "urgent" in result.node_results
        assert "normal" not in result.node_results

    @pytest.mark.asyncio
    async def test_startswith_routing(self):
        chain = Chain("starts", checkpoint_enabled=False)
        chain.add_node("tag", agent=_agent("tag", "/api/orders"))
        chain.add_node("api", agent=_agent("api", "a"))
        chain.add_node("web", agent=_agent("web", "w"))
        chain.connect(
            "tag",
            "api",
            condition='{{node_results.tag.output}} startswith "/api"',
        )
        chain.connect("tag", "web")

        result = await chain.aexecute({"input": "hi"})
        assert "api" in result.node_results
        assert "web" not in result.node_results

    @pytest.mark.asyncio
    async def test_pure_fanout_runs_every_branch(self):
        """Backwards compat: no condition= anywhere → all targets execute."""
        chain = Chain("fanout", checkpoint_enabled=False)
        chain.add_node("src", agent=_agent("src", "s"))
        chain.add_node("left", agent=_agent("left", "l"))
        chain.add_node("right", agent=_agent("right", "r"))
        chain.connect("src", "left")
        chain.connect("src", "right")

        result = await chain.aexecute({"input": "hi"})
        assert "left" in result.node_results
        assert "right" in result.node_results

    @pytest.mark.asyncio
    async def test_pruned_branch_can_be_reactivated_by_other_source(self):
        """A → C is conditional and fails; B → C still activates C."""
        chain = Chain("multi-source", checkpoint_enabled=False)
        chain.add_node("a", agent=_agent("a", "skip"))
        chain.add_node("b", agent=_agent("b", "ok"))
        chain.add_node("c", agent=_agent("c", "final"))
        # A routes to C only when category == billing — it doesn't.
        chain.connect("a", "c", condition="{{node_results.a.output}} == billing")
        # A also routes to B unconditionally; B then routes to C.
        chain.connect("a", "b")
        chain.connect("b", "c")

        result = await chain.aexecute({"input": "hi"})
        assert "c" in result.node_results

    @pytest.mark.asyncio
    async def test_chain_state_input_lookup(self):
        """Conditions can also reference ``{{input.<key>}}`` directly."""
        chain = Chain("input-lookup", checkpoint_enabled=False)
        chain.add_node("root", agent=_agent("root", "ok"))
        chain.add_node("billing", agent=_agent("billing", "b"))
        chain.add_node("general", agent=_agent("general", "g"))
        chain.connect("root", "billing", condition="{{input.category}} == billing")
        chain.connect("root", "general")

        result = await chain.aexecute({"input": "hi", "category": "billing"})
        assert "billing" in result.node_results
        assert "general" not in result.node_results

    @pytest.mark.asyncio
    async def test_condition_reads_state_written_by_just_executed_node(self):
        """A node writes to state, the very next edge routes on that state.

        Regression for a Codex finding: ``context["state"]`` was a copy
        taken before the node ran, so ``_select_outgoing_edges`` saw
        stale state and missed the just-written key. Reproducer mirrors
        the docs example: a score tool returning ``{"score": 0.9}`` and
        an edge with ``condition="{{state.output.score}} >= 0.7"`` —
        the high-score branch must activate.
        """
        from fastaiagent.tool.base import Tool, ToolResult

        class _ScoreTool(Tool):
            def __init__(self) -> None:
                super().__init__(name="score", description="Returns a static score dict.")

            async def aexecute(self, args, context=None):  # type: ignore[override]
                return ToolResult(output={"score": 0.9}, error=None)

        chain = Chain("state-after-write", checkpoint_enabled=False)
        chain.add_node("scorer", tool=_ScoreTool(), type=NodeType.tool)
        chain.add_node("ship", agent=_agent("ship", "shipped"))
        chain.add_node("review", agent=_agent("review", "needs review"))
        chain.connect("scorer", "ship", condition="{{state.output.score}} >= 0.7")
        chain.connect("scorer", "review")

        result = await chain.aexecute({"input": "evaluate me"})
        assert "ship" in result.node_results
        assert "review" not in result.node_results
        # The scorer wrote ``{"output": {"score": 0.9}}`` into state, and
        # the high-score branch ran because the edge condition read it
        # back — verified above. (``state["output"]`` is then overwritten
        # by ``ship``'s own output, so don't assert on final_state.)
        assert result.node_results["scorer"]["output"] == {"score": 0.9}


# -- NodeType.condition routing ---------------------------------------------


class TestConditionNodeRouting:
    @pytest.mark.asyncio
    async def test_matched_handle_selects_labeled_edge(self):
        chain = Chain("cond-node", checkpoint_enabled=False)
        chain.add_node(
            "router",
            type=NodeType.condition,
            conditions=[
                {"expression": "{{input.category}} == billing", "handle": "billing"},
                {"expression": "{{input.category}} == technical", "handle": "tech"},
            ],
        )
        chain.add_node("billing_agent", agent=_agent("billing_agent", "b"))
        chain.add_node("tech_agent", agent=_agent("tech_agent", "t"))
        chain.add_node("general_agent", agent=_agent("general_agent", "g"))
        chain.connect("router", "billing_agent", label="billing")
        chain.connect("router", "tech_agent", label="tech")
        chain.connect("router", "general_agent")  # default

        result = await chain.aexecute({"input": "hi", "category": "billing"})
        assert "billing_agent" in result.node_results
        assert "tech_agent" not in result.node_results
        assert "general_agent" not in result.node_results

    @pytest.mark.asyncio
    async def test_default_handle_falls_back_to_unlabeled_edge(self):
        chain = Chain("cond-default", checkpoint_enabled=False)
        chain.add_node(
            "router",
            type=NodeType.condition,
            conditions=[
                {"expression": "{{input.category}} == billing", "handle": "billing"},
            ],
        )
        chain.add_node("billing_agent", agent=_agent("billing_agent", "b"))
        chain.add_node("general_agent", agent=_agent("general_agent", "g"))
        chain.connect("router", "billing_agent", label="billing")
        chain.connect("router", "general_agent")

        result = await chain.aexecute({"input": "hi", "category": "weather"})
        assert "general_agent" in result.node_results
        assert "billing_agent" not in result.node_results


# -- Validator rules --------------------------------------------------------


class TestRoutingValidator:
    def test_multiple_defaults_alongside_condition_is_rejected(self):
        nodes = [
            NodeConfig(id="src"),
            NodeConfig(id="a"),
            NodeConfig(id="b"),
            NodeConfig(id="c"),
        ]
        edges = [
            Edge(source="src", target="a", condition="{{x}} == 1"),
            Edge(source="src", target="b"),
            Edge(source="src", target="c"),
        ]
        errors = validate_chain(nodes, edges)
        assert any("default" in e for e in errors)

    def test_condition_node_missing_handle_edge_is_rejected(self):
        nodes = [
            NodeConfig(
                id="router",
                type=NodeType.condition,
                config={
                    "conditions": [
                        {"expression": "x == 1", "handle": "left"},
                        {"expression": "x == 2", "handle": "right"},
                    ]
                },
            ),
            NodeConfig(id="left"),
            NodeConfig(id="right"),
        ]
        edges = [
            Edge(source="router", target="left", label="left"),
            # right handle not covered, no default
        ]
        errors = validate_chain(nodes, edges)
        assert any("right" in e for e in errors)
        assert any("default" in e for e in errors)

    def test_condition_node_with_full_coverage_passes(self):
        nodes = [
            NodeConfig(
                id="router",
                type=NodeType.condition,
                config={"conditions": [{"expression": "x == 1", "handle": "left"}]},
            ),
            NodeConfig(id="left"),
            NodeConfig(id="default_target"),
        ]
        edges = [
            Edge(source="router", target="left", label="left"),
            Edge(source="router", target="default_target"),
        ]
        errors = validate_chain(nodes, edges)
        assert not any("handle" in e.lower() for e in errors)
        assert not any("default" in e.lower() for e in errors)

    def test_fanout_chain_without_conditions_still_validates(self):
        """No conditions anywhere → no routing-related errors."""
        nodes = [NodeConfig(id="a"), NodeConfig(id="b"), NodeConfig(id="c")]
        edges = [
            Edge(source="a", target="b"),
            Edge(source="a", target="c"),
        ]
        errors = validate_chain(nodes, edges)
        assert not any("default" in e for e in errors)


# -- Strict routing (v1.14) -------------------------------------------------


class TestStrictRouting:
    """``Chain(..., strict_routing=True)`` raises ``ChainRoutingError`` when
    no outgoing edge matches, instead of silently terminating the branch.
    The default (``False``) preserves the pre-v1.14 silent-termination
    behavior — see ``docs/chains/spec.md`` §Routing.
    """

    @pytest.mark.asyncio
    async def test_all_conditions_fail_default_terminates_silently(self):
        chain = Chain("loose", checkpoint_enabled=False)
        chain.add_node("src", agent=_agent("src", "x"))
        chain.add_node("a", agent=_agent("a", "a_out"))
        chain.add_node("b", agent=_agent("b", "b_out"))
        chain.connect("src", "a", condition="{{node_results.src.output}} == foo")
        chain.connect("src", "b", condition="{{node_results.src.output}} == bar")

        # Legacy behavior: no condition matches, no default → silent prune.
        result = await chain.aexecute({"input": "hi"})
        assert "src" in result.node_results
        assert "a" not in result.node_results
        assert "b" not in result.node_results

    @pytest.mark.asyncio
    async def test_all_conditions_fail_strict_raises(self):
        from fastaiagent._internal.errors import ChainRoutingError

        chain = Chain("strict", checkpoint_enabled=False, strict_routing=True)
        chain.add_node("src", agent=_agent("src", "x"))
        chain.add_node("a", agent=_agent("a", "a_out"))
        chain.add_node("b", agent=_agent("b", "b_out"))
        chain.connect("src", "a", condition="{{node_results.src.output}} == foo")
        chain.connect("src", "b", condition="{{node_results.src.output}} == bar")

        with pytest.raises(ChainRoutingError, match="no default"):
            await chain.aexecute({"input": "hi"})

    @pytest.mark.asyncio
    async def test_strict_passes_when_default_edge_present(self):
        chain = Chain("strict-ok", checkpoint_enabled=False, strict_routing=True)
        chain.add_node("src", agent=_agent("src", "x"))
        chain.add_node("a", agent=_agent("a", "a_out"))
        chain.add_node("fallback", agent=_agent("fallback", "f_out"))
        chain.connect("src", "a", condition="{{node_results.src.output}} == foo")
        chain.connect("src", "fallback")  # unconditional fallback

        result = await chain.aexecute({"input": "hi"})
        assert "fallback" in result.node_results
        assert "a" not in result.node_results

    @pytest.mark.asyncio
    async def test_strict_condition_node_with_no_matching_handle_raises(self):
        from fastaiagent._internal.errors import ChainRoutingError

        chain = Chain("strict-cond", checkpoint_enabled=False, strict_routing=True)
        chain.add_node(
            "router",
            type=NodeType.condition,
            conditions=[{"expression": "{{input.category}} == billing", "handle": "billing"}],
        )
        chain.add_node("billing", agent=_agent("billing", "b"))
        chain.connect("router", "billing", label="billing")
        # No default edge — strict should raise when input doesn't match.

        with pytest.raises(ChainRoutingError):
            await chain.aexecute({"input": "hi", "category": "weather"})


# -- Parallel failure modes (v1.14) -----------------------------------------


class _BoomAgent:
    """Minimal Agent stand-in that raises on ``arun`` — used to test
    parallel failure modes without needing real LLM calls."""

    def __init__(self, name: str, message: str = "boom"):
        self.name = name
        self.message = message

    async def arun(self, _input, **_kwargs):
        raise RuntimeError(self.message)


class _OkAgent:
    """Minimal Agent stand-in that returns a deterministic output."""

    def __init__(self, name: str, output: str = "ok"):
        self.name = name
        self.output = output

    async def arun(self, _input, **_kwargs):
        class _R:
            pass

        r = _R()
        r.output = self.output
        r.tool_calls = []
        return r


class TestParallelFailureModes:
    """``NodeConfig.parallel_failure_mode`` controls how a ``parallel``
    node handles child exceptions. See ``docs/chains/spec.md`` §Parallel.
    """

    @pytest.mark.asyncio
    async def test_continue_default_collects_errors_as_outputs(self):
        # Default mode: every result lands in ``outputs``; exceptions
        # become ``{"error": ...}`` entries. Backwards-compatible.
        chain = Chain("par-continue", checkpoint_enabled=False)
        chain.add_node(
            "fan",
            type=NodeType.parallel,
            agents=[_OkAgent("a", "good"), _BoomAgent("b", "kaboom")],
        )

        result = await chain.aexecute({"input": "x"})
        outputs = result.node_results["fan"]["outputs"]
        assert len(outputs) == 2
        assert any("good" in str(o.get("output", "")) for o in outputs)
        assert any("kaboom" in str(o.get("error", "")) for o in outputs)

    @pytest.mark.asyncio
    async def test_fail_fast_raises_on_first_child_error(self):
        from fastaiagent._internal.errors import ChainError

        chain = Chain("par-failfast", checkpoint_enabled=False)
        node = NodeConfig(
            id="fan",
            type=NodeType.parallel,
            config={"agents": [_BoomAgent("a", "first"), _OkAgent("b", "good")]},
            parallel_failure_mode="fail_fast",
        )
        chain.nodes.append(node)

        with pytest.raises(ChainError, match="fail_fast"):
            await chain.aexecute({"input": "x"})

    @pytest.mark.asyncio
    async def test_any_success_returns_only_successful_children(self):
        chain = Chain("par-any", checkpoint_enabled=False)
        node = NodeConfig(
            id="fan",
            type=NodeType.parallel,
            config={"agents": [_BoomAgent("a"), _OkAgent("b", "winner"), _BoomAgent("c")]},
            parallel_failure_mode="any_success",
        )
        chain.nodes.append(node)

        result = await chain.aexecute({"input": "x"})
        outputs = result.node_results["fan"]["outputs"]
        assert len(outputs) == 1
        assert "winner" in str(outputs[0].get("output", ""))

    @pytest.mark.asyncio
    async def test_any_success_raises_when_all_children_fail(self):
        from fastaiagent._internal.errors import ChainError

        chain = Chain("par-any-fail", checkpoint_enabled=False)
        node = NodeConfig(
            id="fan",
            type=NodeType.parallel,
            config={"agents": [_BoomAgent("a"), _BoomAgent("b")]},
            parallel_failure_mode="any_success",
        )
        chain.nodes.append(node)

        with pytest.raises(ChainError, match="any_success"):
            await chain.aexecute({"input": "x"})


# -- Isolated nodes (v1.14) -------------------------------------------------


class TestIsolatedNodes:
    """``NodeConfig.config["reachable"]=False`` opts a node out of the
    validator's "orphan (no edges)" error. Use for diagnostic-only nodes
    that are intentionally disconnected. See ``docs/chains/spec.md`` §Validation.
    """

    def test_orphan_node_is_rejected_by_default(self):
        nodes = [
            NodeConfig(id="a"),
            NodeConfig(id="b"),
            NodeConfig(id="orphan"),
        ]
        edges = [Edge(source="a", target="b")]
        errors = validate_chain(nodes, edges)
        assert any("orphan" in e.lower() for e in errors)

    def test_orphan_node_with_reachable_false_passes(self):
        nodes = [
            NodeConfig(id="a"),
            NodeConfig(id="b"),
            NodeConfig(id="diag", config={"reachable": False}),
        ]
        edges = [Edge(source="a", target="b")]
        errors = validate_chain(nodes, edges)
        assert not any("orphan" in e.lower() and "diag" in e for e in errors)


# -- ChainState copy semantics (v1.14) --------------------------------------


class TestChainStateSemantics:
    """``ChainState.data`` returns a copy — mutating the returned dict
    never affects chain state. Documented stable contract.
    """

    def test_data_returns_copy(self):
        from fastaiagent.chain.state import ChainState

        st = ChainState({"k": 1})
        d = st.data
        d["k"] = 999
        d["new"] = "x"
        assert st.data == {"k": 1}  # original unchanged
        assert "new" not in st.data

    def test_snapshot_is_deep_and_json_safe(self):
        from fastaiagent.chain.state import ChainState

        st = ChainState({"nested": {"list": [1, 2, 3]}})
        snap = st.snapshot()
        snap["nested"]["list"].append(999)
        assert st.data["nested"]["list"] == [1, 2, 3]  # original unchanged
