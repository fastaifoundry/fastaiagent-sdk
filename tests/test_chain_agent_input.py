"""``Chain.add_node(..., input=...)`` — an agent node that works on another node's output (1.105.0).

An agent node is given the chain's input. That is right for many chains (two agents that each
answer the user's question), and it is the default still. But "A drafts, B reviews the draft" was
not possible: B was handed the original question and never saw the draft, and the README, the
loops page and an example all showed that shape as if it worked. ``input=`` is an opt-in template
per agent node.

No mocks: real chains, scripted models that record what they were given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, Chain
from fastaiagent._internal.errors import ChainError
from fastaiagent.chain import NodeType
from fastaiagent.testing import FunctionModel


def _recording(name: str, seen: dict[str, list[str]], reply: str) -> Agent:
    """An agent that records the user message it was given."""

    def fn(messages: Any) -> str:
        seen.setdefault(name, []).append(str(messages[-1].content))
        return reply

    return Agent(name=name, llm=FunctionModel(fn))


def test_without_input_an_agent_node_gets_the_chain_input_as_before(
    isolated_local_db: Path,
) -> None:
    seen: dict[str, list[str]] = {}
    chain = Chain("default", checkpoint_enabled=False)
    chain.add_node("a", agent=_recording("a", seen, "from a"))
    chain.add_node("b", agent=_recording("b", seen, "from b"))
    chain.connect("a", "b")
    chain.execute({"message": "the question"}, trace=False)
    assert "the question" in seen["a"][0]
    assert "the question" in seen["b"][0]
    assert "from a" not in seen["b"][0]


def test_input_hands_an_agent_another_nodes_output(isolated_local_db: Path) -> None:
    seen: dict[str, list[str]] = {}
    chain = Chain("draft-review", checkpoint_enabled=False)
    chain.add_node("draft", agent=_recording("draft", seen, "THE DRAFT TEXT"))
    chain.add_node(
        "review",
        agent=_recording("review", seen, "PASS"),
        input="Review this draft: {{node_results.draft.output}}",
    )
    chain.connect("draft", "review")
    result = chain.execute({"message": "write a reply"}, trace=False)
    assert seen["review"] == ["Review this draft: THE DRAFT TEXT"]
    assert result.node_results["review"]["output"] == "PASS"


def test_input_can_read_state_and_the_chain_input(isolated_local_db: Path) -> None:
    seen: dict[str, list[str]] = {}
    chain = Chain("mixed", checkpoint_enabled=False)
    chain.add_node("a", agent=_recording("a", seen, "x"), output_key="topic_note")
    chain.add_node(
        "b",
        agent=_recording("b", seen, "y"),
        input="Topic: {{input.topic}}; note: {{state.topic_note}}",
    )
    chain.connect("a", "b")
    chain.execute({"topic": "refunds"}, trace=False)
    assert seen["b"] == ["Topic: refunds; note: x"]


def test_only_the_nodes_that_ask_for_it_change(isolated_local_db: Path) -> None:
    """Mixed chain: c reads b's output, b still answers the user's question."""
    seen: dict[str, list[str]] = {}
    chain = Chain("mixed-usage", checkpoint_enabled=False)
    chain.add_node("a", agent=_recording("a", seen, "out-a"))
    chain.add_node("b", agent=_recording("b", seen, "out-b"))
    chain.add_node("c", agent=_recording("c", seen, "out-c"), input="{{node_results.b.output}}")
    chain.connect("a", "b").connect("b", "c")
    chain.execute({"message": "q"}, trace=False)
    assert "q" in seen["b"][0] and "out-a" not in seen["b"][0]
    assert seen["c"] == ["out-b"]


def test_in_a_loop_each_pass_reads_the_latest_output(isolated_local_db: Path) -> None:
    """The reviewer judges the latest draft on every pass, and the drafter reads the review."""
    seen: dict[str, list[str]] = {}
    drafts = iter(["draft one", "draft two"])
    reviews = iter(["FAIL: too long", "PASS"])

    def draft_fn(messages: Any) -> str:
        seen.setdefault("draft", []).append(str(messages[-1].content))
        return next(drafts)

    def review_fn(messages: Any) -> str:
        seen.setdefault("review", []).append(str(messages[-1].content))
        return next(reviews)

    chain = Chain("loop", checkpoint_enabled=False)
    chain.add_node(
        "draft",
        agent=Agent(name="d", llm=FunctionModel(draft_fn)),
        input="Topic: {{input.message}}. Last review: {{node_results.review.output}}",
    )
    chain.add_node(
        "review",
        agent=Agent(name="r", llm=FunctionModel(review_fn)),
        input="{{node_results.draft.output}}",
    )
    chain.connect("draft", "review")
    chain.connect(
        "review",
        "draft",
        max_iterations=3,
        exit_condition="{{node_results.review.output}} startswith PASS",
    )
    result = chain.execute({"message": "refunds"}, trace=False)
    assert result.status == "completed"
    assert seen["review"] == ["draft one", "draft two"]
    assert seen["draft"] == [
        "Topic: refunds. Last review: ",
        "Topic: refunds. Last review: FAIL: too long",
    ]


def test_an_input_that_renders_empty_stops_before_calling_the_agent(
    isolated_local_db: Path,
) -> None:
    """``draft`` comes after ``review`` here, so ``review`` has nothing to read."""
    seen: dict[str, list[str]] = {}
    chain = Chain("wrong-order", checkpoint_enabled=False)
    chain.add_node(
        "review", agent=_recording("review", seen, "x"), input="{{node_results.draft.output}}"
    )
    chain.add_node("draft", agent=_recording("draft", seen, "y"))
    chain.connect("review", "draft")
    with pytest.raises(ChainError, match="rendered an empty input"):
        chain.execute({"message": "q"}, trace=False)
    assert "review" not in seen


def test_validate_reports_a_misspelled_node(isolated_local_db: Path) -> None:
    chain = Chain("typo", checkpoint_enabled=False)
    chain.add_node("draft", agent=_recording("draft", {}, "x"))
    chain.add_node(
        "review", agent=_recording("review", {}, "y"), input="{{node_results.drafft.output}}"
    )
    chain.connect("draft", "review")
    (error,) = chain.validate()
    assert "node_results.drafft" in error and "no node 'drafft'" in error


def test_validate_reports_input_on_a_node_that_is_not_an_agent() -> None:
    chain = Chain("not-agent", checkpoint_enabled=False)
    chain.add_node("t", type=NodeType.transformer, template="hi", input="{{input.message}}")
    errors = chain.validate()
    assert any("input= is read by agent nodes only" in e for e in errors)


def test_input_must_be_a_string() -> None:
    chain = Chain("bad", checkpoint_enabled=False)
    with pytest.raises(TypeError, match="template string"):
        chain.add_node("a", agent=_recording("a", {}, "x"), input={"x": 1})  # type: ignore[arg-type]


def test_input_survives_to_dict_and_from_dict(isolated_local_db: Path) -> None:
    chain = Chain("roundtrip", checkpoint_enabled=False)
    chain.add_node("a", agent=_recording("a", {}, "x"))
    chain.add_node("b", agent=_recording("b", {}, "y"), input="{{node_results.a.output}}")
    chain.connect("a", "b")
    data = json.loads(json.dumps(chain.to_dict(), default=str))
    (b,) = [n for n in data["nodes"] if n["id"] == "b"]
    assert b["config"]["agent_input"] == "{{node_results.a.output}}"
    restored = Chain.from_dict(data)
    (rb,) = [n for n in restored.nodes if n.id == "b"]
    assert rb.config["agent_input"] == "{{node_results.a.output}}"
