"""AgentEval slice — deterministic, mock-free unit tests.

Covers the no-LLM surfaces: scorer registration, Scorecard aggregation, the
hardening failure-extraction + report shape, and the Hallucination no-context
path. LLM-backed behaviour (generate_scenarios, task_completion /
reflection_quality scoring, the harden LLM analysis) is covered by the real-LLM
tests in ``tests/e2e/test_eval_agenteval_e2e.py`` — no mocking anywhere.
"""

from __future__ import annotations

from fastaiagent.eval import Hallucination, MetricSummary, Recommendation, Scorecard
from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.harden import HardeningReport, _failures_text, harden
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import ScorerResult
from fastaiagent.eval.simulate import (
    CriterionVerdict,
    SimulationResult,
    SimulationResults,
    TranscriptTurn,
)

# --------------------------------------------------------------------------- #
# Scorer registration + Hallucination no-context
# --------------------------------------------------------------------------- #


def test_new_named_metrics_registered() -> None:
    for name in ("task_completion", "hallucination", "reflection_quality"):
        assert name in BUILTIN_SCORERS, name


def test_hallucination_no_context_path() -> None:
    res = Hallucination().score(input="q", output="a")  # no context kwarg
    assert res.score == 0.0
    assert res.passed is False
    assert res.reason == "No context provided"


# --------------------------------------------------------------------------- #
# Scorecard aggregation (no LLM)
# --------------------------------------------------------------------------- #


def test_scorecard_from_eval_results() -> None:
    er = EvalResults(
        scores={
            "task_completion": [
                ScorerResult(score=1.0, passed=True),
                ScorerResult(score=0.0, passed=False),
            ],
            "faithfulness": [
                ScorerResult(score=0.8, passed=True),
                ScorerResult(score=0.9, passed=True),
            ],
        }
    )
    sc = Scorecard.from_eval_results(er, label="demo")
    assert sc.label == "demo"
    assert sc.overall_pass_rate == 0.75  # 3 of 4 passed
    by_name = {m.name: m for m in sc.metrics}
    assert isinstance(by_name["task_completion"], MetricSummary)
    assert by_name["task_completion"].avg_score == 0.5
    assert by_name["task_completion"].pass_rate == 0.5
    assert by_name["faithfulness"].pass_rate == 1.0
    d = sc.to_dict()
    assert d["overall_pass_rate"] == 0.75
    assert {m["name"] for m in d["metrics"]} == {"task_completion", "faithfulness"}


def test_scorecard_from_simulation_ducktyped() -> None:
    sr = SimulationResults(
        [
            SimulationResult(scenario_name="a", passed=True, transcript=[], verdicts=[]),
            SimulationResult(scenario_name="b", passed=False, transcript=[], verdicts=[]),
            SimulationResult(scenario_name="c", passed=True, transcript=[], verdicts=[]),
        ],
        agent_name="bot",
    )
    sc = Scorecard.from_simulation(sr)
    assert sc.label == "bot"
    assert sc.metrics[0].n == 3
    assert round(sc.overall_pass_rate, 2) == 0.67


def test_scorecard_empty_is_zero() -> None:
    sc = Scorecard.from_eval_results(EvalResults(scores={}))
    assert sc.overall_pass_rate == 0.0
    assert sc.metrics == []


# --------------------------------------------------------------------------- #
# Hardening — failure extraction + report shape (no LLM)
# --------------------------------------------------------------------------- #


def test_failures_text_from_simulation_results() -> None:
    sr = SimulationResults(
        [
            SimulationResult(
                scenario_name="rude-bot",
                passed=False,
                transcript=[TranscriptTurn(turn_index=0, role="user", content="help")],
                verdicts=[CriterionVerdict("be polite", "success", False, "was rude")],
            ),
            SimulationResult(scenario_name="ok", passed=True, transcript=[], verdicts=[]),
        ]
    )
    text, count = _failures_text(sr)
    assert count == 1
    assert "rude-bot" in text
    assert "be polite" in text


def test_failures_text_from_eval_results() -> None:
    er = EvalResults()
    from fastaiagent.eval.results import EvalCaseRecord

    er.add_case(
        EvalCaseRecord(
            input="q1",
            actual_output="bad",
            per_scorer={"task_completion": {"score": 0.1, "passed": False, "reason": "missed"}},
        )
    )
    er.add_case(
        EvalCaseRecord(
            input="q2",
            actual_output="good",
            per_scorer={"task_completion": {"score": 1.0, "passed": True}},
        )
    )
    text, count = _failures_text(er)
    assert count == 1
    assert "task_completion" in text


def test_harden_no_failures_is_key_free() -> None:
    # All-passing results → failure_count 0, no LLM client constructed.
    er = EvalResults(scores={"task_completion": [ScorerResult(score=1.0, passed=True)]})
    report = harden(object(), er)
    assert isinstance(report, HardeningReport)
    assert report.failure_count == 0
    assert report.recommendations == []


def test_hardening_report_to_dict() -> None:
    report = HardeningReport(
        agent_name="bot",
        failure_count=2,
        recommendations=[
            Recommendation(target="tools", recommendation="add lookup_order", rationale="needed"),
            Recommendation(target="instructions", recommendation="cite policy", rationale=""),
        ],
    )
    d = report.to_dict()
    assert d["agent_name"] == "bot"
    assert d["failure_count"] == 2
    assert d["recommendations"][0]["target"] == "tools"
    assert "add lookup_order" in report.summary()


# --------------------------------------------------------------------------- #
# 1.86.0: harden() shows a bounded failure block and reads any reply shape
# --------------------------------------------------------------------------- #


def _failing_eval(n: int) -> EvalResults:
    from fastaiagent.eval.results import EvalCaseRecord

    er = EvalResults()
    for i in range(n):
        er.add_case(
            EvalCaseRecord(
                input=f"q{i}",
                expected_output="yes",
                actual_output="no",
                per_scorer={"exact_match": {"score": 0.0, "passed": False}},
            )
        )
    return er


def _recording(reply: str, seen: list[str]):
    from fastaiagent.testing import FunctionModel

    def respond(messages):
        seen.append(messages[-1].content)
        return reply

    return FunctionModel(respond)


def test_harden_shows_a_bounded_number_of_failures() -> None:
    """Every failing case went into one request; 2,000 made a ~335k-token prompt."""
    seen: list[str] = []
    reply = '{"recommendations": [{"target": "instructions", "recommendation": "be exact"}]}'
    report = harden(object(), _failing_eval(500), llm=_recording(reply, seen))
    assert report.failure_count == 500
    assert "(40 of 500)" in seen[0] and seen[0].count("Case input=") == 40
    assert [r.recommendation for r in report.recommendations] == ["be exact"]


def test_failures_text_caps_simulation_scenarios() -> None:
    sr = SimulationResults(
        [
            SimulationResult(
                scenario_name=f"s{i}",
                passed=False,
                transcript=[TranscriptTurn(turn_index=0, role="user", content="help")],
                verdicts=[CriterionVerdict("be polite", "success", False, "was rude")],
            )
            for i in range(5)
        ]
    )
    text, count = _failures_text(sr, max_cases=2)
    assert count == 5
    assert text.count("FAILED") == 2


def test_harden_reads_a_bare_list_reply() -> None:
    """A reply that is a JSON list raised AttributeError out of harden()."""
    reply = '[{"target": "tools", "recommendation": "add lookup"}, "not an object"]'
    report = harden(object(), _failing_eval(2), llm=_recording(reply, []))
    assert [(r.target, r.recommendation) for r in report.recommendations] == [
        ("tools", "add lookup")
    ]


def test_harden_reports_a_reply_with_no_recommendations_list() -> None:
    """Valid JSON with no list read as "nothing to recommend" — a clean verdict
    from an analysis that never happened."""
    report = harden(object(), _failing_eval(2), llm=_recording('{"advice": "none"}', []))
    assert [r.recommendation for r in report.recommendations] == ["(analysis failed)"]
