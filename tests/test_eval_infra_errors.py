"""Part 2: an infrastructure failure during scoring is not an agent-quality miss.

When the agent run itself infra-fails while being scored (provider 500, timeout,
network/auth error), the case must be recorded as *errored* — not scored as a
spurious ``passed=False`` that would feed the optimizer's proposer as a fault to
"fix". No mocking: ``evaluate`` runs a real Python ``agent_fn`` (one that raises,
one that returns a wrong answer, one that returns ``None``).
"""

from __future__ import annotations

from fastaiagent.eval.evaluate import evaluate
from fastaiagent.eval.harden import _failures_text


def test_infra_error_during_scoring_marks_case_errored_not_failed() -> None:
    def boom(_x: str) -> str:
        raise RuntimeError("provider 500")

    res = evaluate(
        agent_fn=boom,
        dataset=[{"input": "q", "expected_output": "a"}],
        scorers=["exact_match"],
        persist=False,
    )
    # Not scored → no spurious passed=False data point.
    assert res.scores.get("exact_match", []) == []
    assert len(res.cases) == 1
    case = res.cases[0]
    assert case.error and "provider 500" in case.error
    assert case.per_scorer == {}
    # The proposer's failure view never sees it.
    _text, count = _failures_text(res)
    assert count == 0


def test_real_wrong_answer_still_counts_as_failure() -> None:
    """Contrast: a clean run that produced the WRONG answer is a real quality
    failure and must still surface (we didn't suppress genuine signal)."""
    res = evaluate(
        agent_fn=lambda _x: "wrong",
        dataset=[{"input": "q", "expected_output": "right"}],
        scorers=["exact_match"],
        persist=False,
    )
    case = res.cases[0]
    assert case.error is None
    assert case.per_scorer["exact_match"]["passed"] is False
    _text, count = _failures_text(res)
    assert count == 1


def test_none_output_does_not_crash_scorers() -> None:
    """A result whose ``.output`` is None is coerced to "" instead of crashing
    ExactMatch/.strip()."""

    class _Result:
        output = None
        trace_id = None

    res = evaluate(
        agent_fn=lambda _x: _Result(),
        dataset=[{"input": "q", "expected_output": ""}],
        scorers=["exact_match"],
        persist=False,
    )
    case = res.cases[0]
    assert case.error is None  # a None output is a (valid, empty) answer, not an infra error
    assert res.scores["exact_match"][0].passed is True  # "" == "" → no crash
