"""AutoLLM keeps a stateless agent stateless (1.87.0).

The few-shot and memory levers carry their block in a ``ComposableMemory``. For
an agent with no memory, that wrapper's primary window used to be a real
``AgentMemory``: the candidate remembered every earlier run, so eval cases bled
into each other, and an applied winner put one user's request into the next
user's prompt.

Swept across lever x how the agent was built (a candidate, the applied winner,
a re-optimized winner) x run path. No mocks: real agents, offline models.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.optimize import OptimizeConfig, optimize
from fastaiagent.optimize.candidate import Candidate, apply_candidate
from fastaiagent.testing.models import FunctionModel

DEMOS = [{"input": "demo question", "output": "demo answer"}]
LEVERS = {
    "fewshot": Candidate(fewshot_demos=DEMOS),
    "memory": Candidate(fact_ids=[]),
    "both": Candidate(fewshot_demos=DEMOS, fact_ids=[]),
}


class _Recorder:
    """A model that answers "ok" and keeps the user turns of every call."""

    def __init__(self) -> None:
        self.user_turns: list[list[str]] = []

    def __call__(self, messages: list[Any]) -> str:
        self.user_turns.append([str(m.content) for m in messages if m.role.value == "user"])
        return "ok"


def _build(how: str, lever: str, rec: _Recorder) -> fa.Agent:
    base = fa.Agent(name=f"stateless-{lever}", system_prompt="You triage.", llm=FunctionModel(rec))
    if how == "candidate":
        return apply_candidate(base, LEVERS[lever])
    if how == "applied":  # what report.apply_to() does
        return apply_candidate(base, LEVERS[lever], allow_writable_memory=True)
    # re-optimized: a winner tuned again clones the memory it was given
    return apply_candidate(apply_candidate(base, LEVERS[lever]), Candidate(fewshot_demos=DEMOS))


def _run(agent: fa.Agent, text: str) -> None:
    agent.run(text)


def _astream(agent: fa.Agent, text: str) -> None:
    async def _consume() -> None:
        async for _ in agent.astream(text):
            pass

    asyncio.run(_consume())


@pytest.mark.parametrize("path", ["run", "astream"])
@pytest.mark.parametrize("how", ["candidate", "applied", "reoptimized"])
@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_a_tuned_stateless_agent_never_sees_an_earlier_run(
    lever: str, how: str, path: str, isolated_local_db: Any
) -> None:
    rec = _Recorder()
    agent = _build(how, lever, rec)
    call = _run if path == "run" else _astream

    call(agent, "customer one: card 4111 1111 1111 1111")
    call(agent, "customer two: hello")

    assert rec.user_turns[-1] == ["customer two: hello"]


def test_the_lever_block_still_reaches_the_prompt(isolated_local_db: Any) -> None:
    """Keeping no conversation must not drop what the lever injects."""
    seen: list[str] = []

    def model(messages: list[Any]) -> str:
        seen.extend(str(m.content) for m in messages if m.role.value == "system")
        return "ok"

    base = fa.Agent(name="fewshot-kept", system_prompt="You triage.", llm=FunctionModel(model))
    apply_candidate(base, LEVERS["fewshot"]).run("hello")

    assert any("demo answer" in s for s in seen)


def test_optimize_scores_each_case_on_its_own(isolated_local_db: Any) -> None:
    """A real few-shot run: no eval case's prompt carries another case's input, and
    the agent report.apply_to() returns is as stateless as the one passed in."""
    rec = _Recorder()
    agent = fa.Agent(name="stateless-optimize", system_prompt="Answer.", llm=FunctionModel(rec))
    cases = [{"input": f"question {i}", "expected_output": "ok"} for i in range(16)]

    report = optimize(
        agent,
        cases,
        ["exact_match"],
        config=OptimizeConfig(levers=("fewshot",), max_iterations=1, patience=1),
        persist=False,
    )

    assert rec.user_turns, "the run evaluated candidates"
    assert all(len(turns) == 1 for turns in rec.user_turns), [
        t for t in rec.user_turns if len(t) > 1
    ][:3]

    rec.user_turns.clear()
    shipped = report.apply_to(agent)
    shipped.run("customer one: my card is 4111")
    shipped.run("customer two: hello")
    assert rec.user_turns[-1] == ["customer two: hello"]
