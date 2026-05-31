"""Seed a real local.db with deterministic simulation runs for the UI / Playwright.

No hand-written SQL and no mocks — this runs the real ``simulate()`` with
``TestModel`` / ``FunctionModel`` (real ``LLMClient`` subclasses) and persists
via ``SimulationResults.persist_local``.

Usage:
    python tests/seed_simulations.py [db_path]

Defaults to ``./.fastaiagent/local.db`` (the configured local DB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def seed(db_path: str | Path | None = None) -> str:
    from fastaiagent import Agent, Scenario, SimulatedUser, simulate
    from fastaiagent.eval import LLMJudge
    from fastaiagent.testing import FunctionModel, TestModel
    from fastaiagent.ui.db import init_local_db

    resolved = Path(db_path) if db_path else Path(".fastaiagent/local.db")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(resolved).close()

    def judge_responder(messages):
        # Success criteria pass; failure criteria are judged ABSENT — except
        # for the one we deliberately want to trip (see "rude" below).
        prompt = str(messages[-1].content)
        if "undesirable condition occurred" in prompt:
            occurred = "rude" in prompt  # only the angry scenario trips this
            return json.dumps(
                {"score": 1.0 if occurred else 0.0, "reasoning": "judged"}
            )
        return json.dumps({"score": 1.0, "reasoning": "criterion met"})

    judge = LLMJudge(llm=FunctionModel(judge_responder))

    agent = Agent(
        name="support-bot",
        system_prompt="You are a friendly support agent.",
        llm=TestModel(response="Our policy allows returns within 30 days. Happy to help!"),
    )

    scenarios = [
        Scenario(
            name="refund-policy",
            user=SimulatedUser(script=["Can I get a refund?", "Great, thank you!"]),
            success_criteria=["The agent explains the refund policy clearly."],
            failure_criteria=["The agent is dismissive."],
            max_turns=6,
        ),
        Scenario(
            name="password-reset",
            user=SimulatedUser(script=["I'm locked out", "ok thanks"]),
            success_criteria=["The agent offers a concrete next step."],
            max_turns=6,
        ),
        Scenario(
            name="angry-escalation",
            user=SimulatedUser(script=["This is unacceptable!"]),
            success_criteria=["The agent stays calm."],
            failure_criteria=["The agent is rude."],  # judged as occurred → FAIL
            max_turns=4,
        ),
    ]

    results = simulate(scenarios, agent, judge=judge, persist=False)
    run_id = results.persist_local(db_path=resolved, run_name="support-suite-v1")
    return run_id


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rid = seed(path)
    print(f"Seeded simulation run: {rid}")
