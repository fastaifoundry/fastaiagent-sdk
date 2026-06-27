"""Eval-driven optimize loop — real-LLM end-to-end test (no mocking).

Runs the full P1 prompt-only loop against a live provider: baseline → propose →
score → keep/revert → holdout guard. Gated on ``OPENAI_API_KEY`` so it skips
cleanly when absent; run with the key from your shell profile::

    zsh -lc 'pytest tests/e2e/test_optimize_e2e.py -q'
"""

from __future__ import annotations

import os

import pytest

from fastaiagent import Agent, LLMClient
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.optimize import OptimizationReport, OptimizeConfig, optimize

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

MODEL = "gpt-4o-mini"


class ContainsCI(Scorer):
    """Deterministic scorer: passes if the expected token appears (case-insensitive)."""

    name = "contains_ci"

    def score(self, input: str = "", output: str = "", expected=None, **kwargs) -> ScorerResult:
        ok = bool(expected) and str(expected).lower() in str(output).lower()
        return ScorerResult(score=1.0 if ok else 0.0, passed=ok)


# A small capitals dataset — a vague baseline prompt tends to over-explain, which
# the contains scorer tolerates, so the loop should at worst hold at baseline.
_CASES = [
    {"input": "Capital of France?", "expected_output": "Paris"},
    {"input": "Capital of Japan?", "expected_output": "Tokyo"},
    {"input": "Capital of Italy?", "expected_output": "Rome"},
    {"input": "Capital of Spain?", "expected_output": "Madrid"},
    {"input": "Capital of Germany?", "expected_output": "Berlin"},
    {"input": "Capital of Canada?", "expected_output": "Ottawa"},
    {"input": "Capital of Egypt?", "expected_output": "Cairo"},
    {"input": "Capital of Brazil?", "expected_output": "Brasilia"},
    {"input": "Capital of Norway?", "expected_output": "Oslo"},
    {"input": "Capital of Kenya?", "expected_output": "Nairobi"},
]


def _agent() -> Agent:
    return Agent(
        name="capitals",
        system_prompt="You answer questions.",
        llm=LLMClient(provider="openai", model=MODEL),
    )


def test_optimize_runs_end_to_end_and_never_regresses():
    report = optimize(
        _agent(),
        _CASES,
        [ContainsCI()],
        config=OptimizeConfig(
            # Pin both levers so this e2e covers prompt + few-shot even though the
            # default is now prompt-only (cheapest default; few-shot is opt-in).
            levers=("instructions", "fewshot"),
            max_iterations=2,
            patience=2,
            candidates_per_iteration=2,
            seed=0,
        ),
        persist=False,
    )

    assert isinstance(report, OptimizationReport)
    # Trajectory always opens with the baseline point.
    assert report.trajectory[0].iteration == 0
    assert report.trajectory[0].lever == "baseline"
    # both levers are exercised (instructions round 1, fewshot round 2)
    assert any(p.lever == "fewshot" for p in report.trajectory)
    # By construction the winner is never worse than baseline on dev (accept-only,
    # plus the holdout guard reverts regressions).
    assert report.best.score >= report.baseline.score - 1e-9
    # Holdout guard ran.
    assert report.holdout_best is not None and report.holdout_baseline is not None
    # A real stopping reason fired.
    assert any(
        report.stopped_reason.startswith(r)
        for r in ("patience", "max_iterations", "target_score", "budget")
    )
    # The winner is applyable and the original is untouched.
    base = _agent()
    tuned = report.apply_to(base)
    assert isinstance(tuned, Agent)
    assert base.system_prompt == "You answer questions."
    # Summary renders without error.
    assert "Optimization" in report.summary()


def test_optimize_memory_lever_preserves_fact_store(tmp_path, monkeypatch):
    """Memory lever runs end-to-end and never mutates the learned-fact store."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    from fastaiagent.learn.store import Fact, MemoryStore

    store = MemoryStore()
    store.add_many(
        [
            Fact(
                scope="agent",
                scope_id="capitals",
                fact="answer with only the city name",
                confidence=0.9,
            ),
            Fact(
                scope="agent", scope_id="capitals", fact="do not add explanations", confidence=0.6
            ),
        ]
    )
    before = [(f.id, f.fact, f.confidence, f.superseded_by) for f in store.list_all()]

    report = optimize(
        _agent(),
        _CASES,
        [ContainsCI()],
        config=OptimizeConfig(
            levers=("memory",),
            max_iterations=2,
            patience=2,
            candidates_per_iteration=2,
            seed=0,
        ),
        persist=False,
    )
    after = [(f.id, f.fact, f.confidence, f.superseded_by) for f in store.list_all()]
    assert before == after  # optimize never mutates the learned-fact store (audit chain intact)
    assert any(p.lever == "memory" and not p.skipped for p in report.trajectory)  # lever ran
    assert report.best.score >= report.baseline.score - 1e-9
