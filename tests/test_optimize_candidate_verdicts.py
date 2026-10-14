"""AutoLLM selection reads 1.89.0's verdict states correctly.

* A **skipped** verdict (the check does not apply to that case) is not a 0: it
  used to drag a candidate's metric down for rows that never carried an
  expectation. A metric with no applicable verdict is dropped.
* An **errored** verdict still scores 0: selection must never reward a
  candidate whose judge could not run on it (the 1.85.0 rule for agent errors).

WS4 §1.6: candidate-C's ``CandidateScore.pass_rate`` goes from 20/30 to 16/25.
Real ``ScorerResult`` objects; no mocks.
"""

from __future__ import annotations

from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import ScorerResult
from fastaiagent.optimize.candidate import CandidateScore

P = ScorerResult(score=1.0, passed=True)
F = ScorerResult(score=0.0, passed=False)
NA = ScorerResult.not_applicable("No expected trajectory")
ERR = ScorerResult.error("judge down")


def _results(scores: dict[str, list[ScorerResult]], *, agent_errors: int = 0) -> EvalResults:
    er = EvalResults(scores=scores)
    for _ in range(agent_errors):
        er.add_case(EvalCaseRecord(input="q", error="BOOM"))
    return er


def test_skipped_verdicts_do_not_count_as_zero() -> None:
    er = _results({"exact_match": [P, P], "tool_usage_accuracy": [NA, NA]})
    cs = CandidateScore.from_eval("c", "dev", er, primary_metric=None)
    assert "tool_usage_accuracy" not in cs.per_metric, "an inapplicable metric was a 0"
    assert cs.per_metric["exact_match"] == 1.0
    assert cs.pass_rate == 1.0


def test_errored_verdicts_still_score_zero() -> None:
    er = _results({"judge": [P, ERR]})
    cs = CandidateScore.from_eval("c", "dev", er, primary_metric="judge")
    assert cs.per_metric["judge"] == 0.5
    assert cs.pass_rate == 0.5


def test_candidate_c_pass_rate() -> None:
    """5 scorers × 4 scored cases (16 pass, 4 fail) + 4 n/a + 1 agent error."""
    p16_f4 = [P] * 16 + [F] * 4
    scores = {f"s{i}": p16_f4[i * 4 : (i + 1) * 4] for i in range(5)}
    scores["tool_usage_accuracy"] = [NA] * 4
    cs = CandidateScore.from_eval("c", "dev", _results(scores, agent_errors=1), primary_metric=None)
    assert cs.pass_rate == 0.64, "16 / (20 applicable + 5 metrics × 1 agent-errored case)"
