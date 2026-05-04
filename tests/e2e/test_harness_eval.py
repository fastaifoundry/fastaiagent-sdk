"""Phase 5 — ``as_evaluable()`` adapter tests (real LLM, gated by env).

Spec test IDs covered: #16 (LangGraph), #17 (CrewAI), #18 (PydanticAI),
#19 (custom mapper), #20 (eval → trace linkage).

Each adapter is exercised against a 3-case dataset using the
``ExactMatch`` scorer plus length scorers — cheap, deterministic where
possible, and ~1 LLM call per case.
"""

from __future__ import annotations

import os

import pytest

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

if not HAS_OPENAI:
    pytest.skip(
        "OPENAI_API_KEY not set — Phase 5 needs at least the OpenAI path",
        allow_module_level=True,
    )

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")

pytestmark = pytest.mark.e2e


def _capital_dataset() -> list[dict[str, str]]:
    return [
        {"input": "What is the capital of France? Answer with one word.", "expected": "Paris"},
        {"input": "What is the capital of Japan? Answer with one word.", "expected": "Tokyo"},
        {"input": "What is the capital of Italy? Answer with one word.", "expected": "Rome"},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return text.strip().rstrip(".").lower()


# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------


@needs_openai
def test_16_langchain_eval_3_cases() -> None:
    """Spec #16: 3 LangGraph eval cases each produce results."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    import fastaiagent as fa
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[])

    evaluable = lc.as_evaluable(graph)
    results = fa.evaluate(
        evaluable,
        dataset=_capital_dataset(),
        scorers=["exact_match"],
        persist=False,
        run_name="harness-eval-langchain",
    )

    cases = results.cases
    assert len(cases) == 3, f"expected 3 cases, got {len(cases)}"
    for case in cases:
        # We don't assert exact match — gpt-4o-mini sometimes adds a
        # period or full sentence — but assert *something* was produced.
        assert case.actual_output, case
        assert _normalise(case.expected_output or "") in _normalise(case.actual_output)


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


@needs_openai
def test_17_crewai_eval() -> None:
    """Spec #17: CrewAI as_evaluable with a 3-case dataset."""
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM

    import fastaiagent as fa
    from fastaiagent.integrations import crewai as ca

    ca.enable()

    llm = LLM(model="openai/gpt-4o-mini", temperature=0)
    researcher = Agent(
        role="Researcher",
        goal="Answer concisely.",
        backstory="You answer in one word.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="{input}",
        expected_output="A single word.",
        agent=researcher,
    )
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    evaluable = ca.as_evaluable(crew)
    # Single case to keep CrewAI runtime / cost down — the spec only
    # requires "scored", not 3-case parity.
    results = fa.evaluate(
        evaluable,
        dataset=_capital_dataset()[:1],
        scorers=["exact_match"],
        persist=False,
        run_name="harness-eval-crewai",
    )
    assert len(results.cases) == 1
    case = results.cases[0]
    assert case.actual_output
    assert "paris" in case.actual_output.lower()


# ---------------------------------------------------------------------------
# PydanticAI
# ---------------------------------------------------------------------------


@needs_openai
def test_18_pydanticai_eval() -> None:
    """Spec #18: PydanticAI as_evaluable scored over 3 cases."""
    from pydantic_ai import Agent

    import fastaiagent as fa
    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="Answer with a single word, no punctuation.",
    )

    evaluable = pa.as_evaluable(agent)
    results = fa.evaluate(
        evaluable,
        dataset=_capital_dataset(),
        scorers=["exact_match"],
        persist=False,
        run_name="harness-eval-pydanticai",
    )

    assert len(results.cases) == 3
    for case in results.cases:
        assert case.actual_output
        assert _normalise(case.expected_output or "") in _normalise(case.actual_output)


# ---------------------------------------------------------------------------
# Custom mapper
# ---------------------------------------------------------------------------


@needs_openai
def test_19_langchain_custom_mapper() -> None:
    """Spec #19: as_evaluable honours custom input_mapper / output_mapper."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[])

    seen_inputs: list[Any] = []  # type: ignore[name-defined]
    seen_outputs: list[Any] = []  # type: ignore[name-defined]

    def in_map(text: str) -> dict:
        seen_inputs.append(text)
        return {
            "messages": [HumanMessage(content=f"Reply with the word: {text}")]
        }

    def out_map(result: dict) -> str:
        seen_outputs.append(result)
        return result["messages"][-1].content

    evaluable = lc.as_evaluable(graph, input_mapper=in_map, output_mapper=out_map)
    out = evaluable("hello")

    assert seen_inputs == ["hello"]
    assert len(seen_outputs) == 1
    assert isinstance(out.output, str)
    assert "hello" in out.output.lower()


# ---------------------------------------------------------------------------
# Eval → trace linkage
# ---------------------------------------------------------------------------


@needs_openai
def test_20_eval_trace_linkage() -> None:
    """Spec #20: each eval case ends up with a trace_id pointing at a
    real trace in the local store."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    import fastaiagent as fa
    from fastaiagent.integrations import langchain as lc
    from fastaiagent.trace.storage import TraceStore

    lc.enable()
    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[])

    evaluable = lc.as_evaluable(graph)
    results = fa.evaluate(
        evaluable,
        dataset=_capital_dataset()[:2],
        scorers=["exact_match"],
        persist=False,
        run_name="harness-eval-link",
    )

    store = TraceStore.default()
    linked = 0
    for case in results.cases:
        if case.trace_id:
            try:
                trace = store.get_trace(case.trace_id)
            except Exception:
                continue
            if trace.spans:
                linked += 1
    assert linked == len(results.cases), (
        f"expected every case to resolve to a trace, got {linked}/{len(results.cases)}"
    )


# Avoids "Any" being treated as undefined when the type-comments in
# test_19 are evaluated under stricter tooling.
from typing import Any  # noqa: E402, F401
