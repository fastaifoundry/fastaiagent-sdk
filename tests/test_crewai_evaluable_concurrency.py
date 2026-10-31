"""``as_evaluable(crew)`` under ``evaluate()``'s concurrency (1.95.0, S12).

One Crew was shared by every case. ``evaluate()`` runs four cases at a time and
a Crew's executors refuse to run twice at once, so most cases errored with
"Executor is already running", and the run still exited 0. A real CrewAI 1.x
crew; its model is a ``BaseLLM`` subclass, CrewAI's own extension point, slow
enough that the cases overlap.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

crewai = pytest.importorskip("crewai")

from crewai import Agent as CrewAgent  # noqa: E402
from crewai import Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402

from fastaiagent.eval import Dataset, evaluate  # noqa: E402
from fastaiagent.integrations import crewai as fa_crewai  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")


class _Slow(BaseLLM):
    """Answers after a pause, so concurrent cases overlap inside the crew."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):  # noqa: D102
        time.sleep(0.3)
        return "Thought: done\nFinal Answer: Paris"

    async def acall(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        return self.call(*args, **kwargs)


def test_concurrent_cases_each_run_their_own_crew(isolated_local_db: Any) -> None:
    agent = CrewAgent(role="analyst", goal="answer", backstory="terse", llm=_Slow(model="slow"))
    task = Task(description="Answer {input}", expected_output="one word", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], name="desk")
    fa_crewai.enable()

    results = evaluate(
        agent_fn=fa_crewai.as_evaluable(crew),
        dataset=Dataset.from_list([{"input": f"q{i}", "expected": "Paris"} for i in range(4)]),
        scorers=["exact_match"],
        concurrency=4,
        persist=False,
    )

    assert [c.error for c in results.cases if c.error] == []
    assert results.case_counts["case_pass_count"] == 4
    trace_ids = [c.trace_id for c in results.cases]
    assert all(trace_ids) and len(set(trace_ids)) == 4, "each case keeps its own trace"
    assert crew.name == "desk"
