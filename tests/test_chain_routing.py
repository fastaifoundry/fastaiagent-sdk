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
