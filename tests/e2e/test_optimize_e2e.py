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


def test_optimize_persists_run_to_local_db(tmp_path, monkeypatch):
    """Real-LLM loop with persist=True lands in local.db and links each iteration
    to the eval_runs row it produced — the AutoLLM persistence + UI data path.

    Guards the full aoptimize → persist_local → eval_run_id chain on a *live*
    run (no mocking), which the hand-built round-trip unit test can't reach.
    """
    from fastaiagent._internal.config import reset_config
    from fastaiagent.ui.db import init_local_db

    db_path = tmp_path / ".fastaiagent" / "local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    try:
        init_local_db(db_path).close()
        report = optimize(
            _agent(),
            _CASES,
            [ContainsCI()],
            config=OptimizeConfig(
                levers=("instructions",),
                max_iterations=2,
                patience=2,
                candidates_per_iteration=2,
                seed=0,
            ),
            run_name="autollm-e2e",
            persist=True,
        )
        assert report.run_id, "a persisted run must stash its run_id"

        db = init_local_db(db_path)
        try:
            run = db.fetchone(
                "SELECT * FROM optimize_runs WHERE run_id = ?", (report.run_id,)
            )
            assert run is not None
            assert run["agent_name"] == "capitals"
            assert run["run_name"] == "autollm-e2e"

            iters = db.fetchall(
                "SELECT * FROM optimize_iterations WHERE run_id = ? ORDER BY ordinal",
                (report.run_id,),
            )
            assert len(iters) == len(report.trajectory) >= 1
            # Every scored (non-skipped) iteration links to a REAL eval_runs row —
            # the eval the live loop actually ran for that candidate.
            linked = [it for it in iters if it["eval_run_id"]]
            assert linked, "scored iterations must carry an eval_run_id"
            for it in linked:
                ev = db.fetchone(
                    "SELECT run_id FROM eval_runs WHERE run_id = ?", (it["eval_run_id"],)
                )
                assert ev is not None, "iteration eval_run_id must resolve to a real eval_runs row"
        finally:
            db.close()
    finally:
        monkeypatch.delenv("FASTAIAGENT_LOCAL_DB", raising=False)
        reset_config()


# ── 1.85.0: the AutoLLM audit, on a live model ──────────────────────────────


def test_a_live_case_that_raises_counts_against_the_candidate():
    """A guardrail-blocked case left the score's denominator, so a candidate
    scored only on the cases it managed to answer."""
    import asyncio

    from fastaiagent.eval.evaluate import aevaluate
    from fastaiagent.guardrail import Guardrail, GuardrailResult
    from fastaiagent.optimize import CandidateScore

    blocked = {"Tokyo", "Cairo"}
    guard = Guardrail(
        name="no_tokyo_cairo",
        fn=lambda text: GuardrailResult(passed=not any(c in text for c in blocked)),
    )
    agent = Agent(
        name="capitals",
        system_prompt="Answer with only the city name.",
        llm=LLMClient(provider="openai", model=MODEL),
        guardrails=[guard],
    )
    res = asyncio.run(aevaluate(agent.arun, _CASES, [ContainsCI()], persist=False))
    cs = CandidateScore.from_eval("c", "dev", res, primary_metric=None)
    correct = sum(
        1 for c in res.cases if not c.error and all(d["passed"] for d in c.per_scorer.values())
    )
    assert cs.errored == res.errored_count >= 2  # Japan and Egypt were blocked
    assert cs.n == len(_CASES)
    assert cs.score == pytest.approx(correct / len(_CASES), abs=1e-3)


def test_a_live_proposer_that_cannot_run_is_reported():
    """A proposer on a model that doesn't exist ended "stopped: patience" with
    nothing logged — indistinguishable from "no better prompt exists"."""
    report = optimize(
        _agent(),
        _CASES,
        ["exact_match"],
        config=OptimizeConfig(max_iterations=2, patience=2),
        proposer_llm=LLMClient(provider="openai", model="gpt-model-that-does-not-exist"),
        persist=False,
    )
    assert report.stopped_reason == "proposer_failed"
    assert report.proposer_errors and not report.improved
    assert "proposer failed" in report.summary()


def test_live_judge_in_scorers_stays_under_max_judge_calls():
    """max_judge_calls counted only a selection_judge: a judge in scorers made 33
    real calls under a cap of 4."""
    from fastaiagent.eval.llm_judge import LLMJudge

    class CountingJudge(LLMJudge):
        calls = 0

        def score(self, *a, **kw):  # type: ignore[override]
            CountingJudge.calls += 1
            return super().score(*a, **kw)

    judge = CountingJudge(
        criteria="Is the output exactly the expected city name, nothing else?",
        llm=LLMClient(provider="openai", model=MODEL),
    )
    report = optimize(
        _agent(),
        _CASES,
        [judge],
        config=OptimizeConfig(max_iterations=4, patience=4, max_judge_calls=14),
        persist=False,
    )
    assert CountingJudge.calls <= 14
    assert report.holdout_baseline is not None


def test_live_fewshot_does_not_leak_favorites_into_the_holdout(tmp_path, monkeypatch):
    """The documented trace→eval path builds the eval set from favorite traces;
    the few-shot lever then added those same favorites as demos and the holdout
    guard reported the leaked answer as a win (audit: +0.333 holdout)."""
    from datetime import datetime, timezone

    from fastaiagent._internal.config import reset_config
    from fastaiagent.eval.curate import curate_from_traces
    from fastaiagent.trace import otel
    from fastaiagent.ui.db import init_local_db

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    otel.reset()
    codes = "A7 C2 F9 B4 D1 E6 G3 H8 J5 K2 L7 M4".split()
    bins = {f"SKU-{1000 + i}": code for i, code in enumerate(codes)}
    table = "; ".join(f"{k} -> {v}" for k, v in bins.items())
    try:
        prod = Agent(
            name="bins",
            system_prompt=f"Answer with only the bin code. Bin table: {table}",
            llm=LLMClient(provider="openai", model=MODEL),
        )
        trace_ids = [prod.run(f"Which bin holds {sku}?").trace_id for sku in bins]
        otel.get_tracer_provider().force_flush()
        db = init_local_db(db_path)
        for tid in trace_ids:
            db.execute(
                "INSERT INTO trace_favorites (trace_id, created_at) VALUES (?, ?)",
                (tid, datetime.now(tz=timezone.utc).isoformat()),
            )
        db.close()
        cases = [dict(c) for c in curate_from_traces(filter="favorites", agent="bins")]
        assert len(cases) == 12

        new = Agent(
            name="bins",
            system_prompt="Answer with only the bin code.",  # no table: it can't know any bin
            llm=LLMClient(provider="openai", model=MODEL),
        )
        cfg = OptimizeConfig(levers=("fewshot",), max_iterations=1, patience=1, seed=0)
        report = optimize(new, cases, ["exact_match"], config=cfg, persist=False)
    finally:
        otel.reset()
        reset_config()

    from fastaiagent.optimize.loop import _split

    _train, dev, holdout = _split(cases, cfg.splits, cfg.seed)
    scored = {c["input"] for c in dev + holdout}
    demos = {d["input"] for d in (report.best_candidate.fewshot_demos or [])}
    assert not demos & scored
    assert report.holdout_best.score == 0.0  # no way to know a held-out bin
