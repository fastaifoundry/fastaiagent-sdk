"""A Supervisor's result reports the whole team's spend (1.99.0).

``Supervisor.run()`` reported the lead's own turns only — live, 7,528 of the
team's 29,339 tokens — and the documented remedy, "sum it with each worker's",
was impossible: the delegate tool dropped every worker's ``AgentResult``. The
plane's trace total was right all along. A routed run with ``validate_outputs``
also dropped each rejected attempt's spend when it retried.

No mocks: the offline ``TestModel`` from ``fastaiagent.testing``, with fixed usage
so the expected totals are exact.
"""

from __future__ import annotations

import asyncio

import pytest

import fastaiagent as fa
from fastaiagent.testing import TestModel

LEAD = (100, 10)  # per call; the lead makes two (delegate, then answer)
WORKER = (7, 3)  # per call; each worker run makes one


def _team() -> fa.Supervisor:
    lead = TestModel(
        response=["", "the team's answer"],
        tool_calls=[{"name": "delegate_to_w", "arguments": {"task": "do it"}}],
        usage=LEAD,
    )
    worker = fa.Agent(name="w", llm=TestModel("worker reply", usage=WORKER))
    return fa.Supervisor(name="spend-lead", llm=lead, workers=[fa.Worker(worker, role="w")])


@pytest.mark.parametrize("path", ["run", "arun", "stream"])
def test_the_result_counts_the_workers(path: str) -> None:
    team = _team()
    if path == "run":
        result = team.run("go")
    elif path == "arun":
        result = asyncio.run(team.arun("go"))
    else:
        result = team.stream("go")

    assert result.output == "the team's answer"
    assert result.tokens_used == 2 * sum(LEAD) + sum(WORKER)
    assert result.cost_known is True


def _verdict(probability: float) -> dict:
    return {
        "answers": [{"type": "predicate", "name": "approved", "probability": probability}],
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    }


def _routed() -> fa.Supervisor:
    route = {
        "answers": [
            {
                "type": "choice",
                "name": "worker",
                "choice": "w",
                "probabilities": [
                    {"value": "w", "probability": 0.9},
                    {"value": "other", "probability": 0.1},
                ],
                "confidence": 0.9,
            }
        ],
        "usage": {"input_tokens": 40, "output_tokens": 0, "total_tokens": 40},
    }
    workers = [
        fa.Worker(fa.Agent(name="w", llm=TestModel("draft", usage=WORKER)), role="w"),
        fa.Worker(fa.Agent(name="o", llm=TestModel("other", usage=WORKER)), role="other"),
    ]
    return fa.Supervisor(
        name="spend-desk",
        llm=TestModel("unused"),
        workers=workers,
        routing="decisions",
        router_llm=TestModel(decisions=route),
        fallback_worker="other",
        validate_outputs=True,
        validation_mode="decisions",
        # Rejects the first attempt, approves the retry.
        validation_llm=TestModel(decisions=[_verdict(0.1), _verdict(0.9)]),
    )


@pytest.mark.parametrize("path", ["run", "stream"])
def test_a_rejected_attempt_is_still_spent(path: str) -> None:
    team = _routed()
    result = team.run("go") if path == "run" else team.stream("go")

    # Both worker attempts plus the routing decision; the same on run and stream.
    assert result.tokens_used == 2 * sum(WORKER) + 40
