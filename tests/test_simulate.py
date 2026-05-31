"""Deterministic tests for agent simulation (no network, no mocks).

Agent, simulated user, and judge are all driven by ``TestModel`` /
``FunctionModel`` — real ``LLMClient`` subclasses returning canned data — so
the full multi-turn loop, judging, and local persistence run offline.
"""

from __future__ import annotations

import json

from fastaiagent.agent.agent import Agent
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.simulate import (
    Scenario,
    SimulatedUser,
    SimulationResults,
    asimulate,
    simulate,
)
from fastaiagent.testing.models import FunctionModel, TestModel
from fastaiagent.ui.db import init_local_db


def _passing_judge() -> LLMJudge:
    """A judge whose LLM always returns score 1.0 (criterion satisfied)."""
    return LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))


def _failing_judge() -> LLMJudge:
    return LLMJudge(llm=TestModel(response=json.dumps({"score": 0.0, "reasoning": "no"})))


def test_scripted_user_transcript_order() -> None:
    """A scripted user alternates with the agent; transcript order is correct."""
    agent = Agent(name="bot", llm=TestModel(response="agent reply"))
    scenario = Scenario(
        name="scripted",
        user=SimulatedUser(script=["hello", "tell me more", "thanks"]),
        success_criteria=["The agent responded."],
        max_turns=10,
    )
    results = simulate(scenario, agent, judge=_passing_judge(), persist=False)

    assert isinstance(results, SimulationResults)
    r = results.results[0]
    roles = [t.role for t in r.transcript]
    # user, assistant, user, assistant, user, assistant
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert r.transcript[0].content == "hello"
    assert r.transcript[1].content == "agent reply"
    assert r.transcript[2].content == "tell me more"


def test_max_turns_cap() -> None:
    """``max_turns`` hard-caps the total number of turns."""
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenario = Scenario(
        name="capped",
        user=SimulatedUser(script=["a", "b", "c", "d", "e"]),
        success_criteria=["ok"],
        max_turns=4,
    )
    results = simulate(scenario, agent, judge=_passing_judge(), persist=False)
    assert len(results.results[0].transcript) == 4


def test_early_stop_when_script_exhausted() -> None:
    """When the scripted user runs out of lines, the conversation ends early."""
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenario = Scenario(
        name="early-stop",
        user=SimulatedUser(script=["only one"]),
        success_criteria=["ok"],
        max_turns=10,
    )
    results = simulate(scenario, agent, judge=_passing_judge(), persist=False)
    # user "only one" + agent reply, then user returns None → stop at 2 turns.
    assert len(results.results[0].transcript) == 2


def test_pass_verdict_all_success() -> None:
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenario = Scenario(
        name="pass",
        user=SimulatedUser(script=["hi"]),
        success_criteria=["crit A", "crit B"],
        max_turns=4,
    )
    results = simulate(scenario, agent, judge=_passing_judge(), persist=False)
    r = results.results[0]
    assert r.passed is True
    assert all(v.passed for v in r.verdicts)
    assert {v.kind for v in r.verdicts} == {"success"}


def test_fail_when_success_criterion_unmet() -> None:
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenario = Scenario(
        name="fail",
        user=SimulatedUser(script=["hi"]),
        success_criteria=["crit"],
        max_turns=4,
    )
    results = simulate(scenario, agent, judge=_failing_judge(), persist=False)
    assert results.results[0].passed is False


def test_failure_criterion_inverts() -> None:
    """A failure criterion that the judge says occurred (score 1.0) → scenario fails,
    and the verdict's ``passed`` is False (desired state = failure absent)."""
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenario = Scenario(
        name="fail-crit",
        user=SimulatedUser(script=["hi"]),
        success_criteria=[],
        failure_criteria=["The agent leaked secrets."],
        max_turns=4,
    )
    # Judge says score 1.0 → the failure DID occur.
    results = simulate(scenario, agent, judge=_passing_judge(), persist=False)
    r = results.results[0]
    assert r.passed is False
    fail_verdict = next(v for v in r.verdicts if v.kind == "failure")
    assert fail_verdict.passed is False


def test_callable_adapter() -> None:
    """A non-Agent callable adapter is supported."""
    seen: dict[str, int] = {"calls": 0}

    def adapter(messages):
        seen["calls"] += 1
        return f"echo: {messages[-1].content}"

    scenario = Scenario(
        name="adapter",
        user=SimulatedUser(script=["ping", "ping2"]),
        success_criteria=["ok"],
        max_turns=4,
    )
    results = simulate(scenario, adapter, judge=_passing_judge(), persist=False)
    r = results.results[0]
    assert r.transcript[1].content == "echo: ping"
    assert seen["calls"] == 2


async def test_persona_user_via_function_model() -> None:
    """A persona-driven user is fully deterministic with a FunctionModel."""
    user_lines = iter(["I need help", "END"])

    def user_responder(messages):
        return next(user_lines)

    agent = Agent(name="bot", llm=TestModel(response="how can I help?"))
    scenario = Scenario(
        name="persona",
        user=SimulatedUser(
            persona="A confused user.", llm=FunctionModel(user_responder)
        ),
        success_criteria=["ok"],
        max_turns=10,
    )
    results = await asimulate(scenario, agent, judge=_passing_judge(), persist=False)
    r = results.results[0]
    # opening "I need help" → agent reply → user "END" stops.
    assert r.transcript[0].content == "I need help"
    assert r.transcript[1].content == "how can I help?"
    assert len(r.transcript) == 2


def test_export_writes_json(tmp_path) -> None:
    """SimulationResults.export() writes a JSON file with transcript + verdicts."""
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenario = Scenario(
        name="exp",
        user=SimulatedUser(script=["hi"]),
        success_criteria=["ok"],
        max_turns=4,
    )
    results = simulate(scenario, agent, judge=_passing_judge(), persist=False)
    out = tmp_path / "sim.json"
    results.export(out)

    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["scenario_name"] == "exp"
    assert data[0]["passed"] is True
    assert data[0]["transcript"][0]["role"] == "user"
    assert data[0]["verdicts"][0]["kind"] == "success"


def test_persist_local_writes_rows(tmp_path) -> None:
    """``persist_local`` writes real sim_runs / sim_cases rows (real SQLite)."""
    db_file = tmp_path / "local.db"
    agent = Agent(name="bot", llm=TestModel(response="reply"))
    scenarios = [
        Scenario(name="s1", user=SimulatedUser(script=["hi"]), success_criteria=["ok"]),
        Scenario(name="s2", user=SimulatedUser(script=["yo"]), success_criteria=["ok"]),
    ]
    results = simulate(scenarios, agent, judge=_passing_judge(), persist=False)
    run_id = results.persist_local(db_path=db_file, run_name="my-run")

    assert results.run_id == run_id
    db = init_local_db(db_file)
    try:
        run = db.fetchone("SELECT * FROM sim_runs WHERE run_id = ?", (run_id,))
        assert run is not None
        assert run["run_name"] == "my-run"
        assert run["scenario_count"] == 2
        assert run["agent_name"] == "bot"

        cases = db.fetchall(
            "SELECT * FROM sim_cases WHERE run_id = ? ORDER BY ordinal", (run_id,)
        )
        assert len(cases) == 2
        transcript = json.loads(cases[0]["transcript"])
        assert transcript[0]["role"] == "user"
        criteria = json.loads(cases[0]["criteria"])
        assert criteria["success"] == ["ok"]
        per_criterion = json.loads(cases[0]["per_criterion"])
        assert per_criterion[0]["kind"] == "success"
    finally:
        db.close()
