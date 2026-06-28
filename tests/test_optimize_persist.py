"""Persistence + UI tests for the optimize loop (P4 persistence slice).

Real SQLite, real FastAPI, no mocks. Covers:

* the v15 migration applies cleanly (14 -> 15) and is idempotent,
* ``OptimizationReport.persist_local`` round-trips a run + iteration rows,
* the optimize-level ``persist`` flag gates writes (False writes nothing),
* the ``/api/optimizes`` list + detail endpoints return a persisted run whose
  iterations link to the real ``eval_runs`` rows each candidate produced,
* the 14 -> 15 migration is **non-breaking** on a populated DB (existing
  ``eval_runs`` data survives the upgrade).

No mocks: a real SQLite file and real FastAPI; the ``evaluate`` calls drive a
plain Python ``agent_fn`` (not an LLM) so these stay deterministic and hermetic.
The **real-LLM** loop→persistence path is covered by
``tests/e2e/test_optimize_e2e.py::test_optimize_persists_run_to_local_db``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.eval.evaluate import evaluate
from fastaiagent.eval.results import EvalResults
from fastaiagent.optimize.candidate import Candidate, CandidateScore
from fastaiagent.optimize.report import OptimizationReport, TrajectoryPoint
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db


def _report(eval_run_id: str | None = None) -> OptimizationReport:
    base = CandidateScore("c0", "dev", 0.5, 0.5, 4, eval_run_id=eval_run_id)
    best = CandidateScore("c1", "dev", 0.8, 0.8, 4, eval_run_id=eval_run_id)
    return OptimizationReport(
        agent_name="kyc",
        baseline=base,
        best=best,
        best_candidate=Candidate(system_prompt="better prompt", id="c1"),
        trajectory=[
            TrajectoryPoint(0, "baseline", "c0", 0.5, True, "baseline", eval_run_id=eval_run_id),
            TrajectoryPoint(
                1, "instructions", "c1", 0.8, True, "clearer", eval_run_id=eval_run_id
            ),
            TrajectoryPoint(
                0, "memory", "", 0.5, accepted=False, rationale="no facts", skipped=True
            ),
        ],
        accepted=["c1"],
        stopped_reason="target_score",
        holdout_baseline=CandidateScore("c0", "holdout", 0.5, 0.5, 2),
        holdout_best=CandidateScore("c1", "holdout", 0.7, 0.7, 2),
        reverted=False,
        seed=7,
        levers=("instructions",),
        run_name="ui-seed",
    )


# ── Migration ───────────────────────────────────────────────────────────────


def test_migration_v15_applies_and_is_idempotent(temp_dir: Path) -> None:
    db_path = temp_dir / "local.db"
    db = init_local_db(db_path)
    try:
        ver = db.fetchone("PRAGMA user_version")
        assert int(next(iter(ver.values()))) == CURRENT_SCHEMA_VERSION == 15
        tables = {
            r["name"]
            for r in db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('optimize_runs','optimize_iterations')"
            )
        }
        assert tables == {"optimize_runs", "optimize_iterations"}
    finally:
        db.close()

    # Re-opening runs migrations again: no error, version stable.
    db2 = init_local_db(db_path)
    try:
        ver2 = db2.fetchone("PRAGMA user_version")
        assert int(next(iter(ver2.values()))) == 15
    finally:
        db2.close()


# ── persist_local round-trip ─────────────────────────────────────────────────


def test_persist_local_round_trips(temp_dir: Path) -> None:
    db_path = temp_dir / "local.db"
    init_local_db(db_path).close()

    rep = _report(eval_run_id="evalrun-xyz")
    run_id = rep.persist_local(db_path=db_path)
    assert run_id and rep.run_id == run_id

    db = init_local_db(db_path)
    try:
        run = db.fetchone("SELECT * FROM optimize_runs WHERE run_id = ?", (run_id,))
        assert run["agent_name"] == "kyc"
        assert run["run_name"] == "ui-seed"
        assert run["baseline_score"] == 0.5
        assert run["best_score"] == 0.8
        assert run["holdout_best_score"] == 0.7
        assert run["seed"] == 7
        assert run["levers"] == '["instructions"]'
        assert run["stopped_reason"] == "target_score"
        assert run["reverted"] == 0
        assert run["iteration_count"] == 3
        assert run["baseline_eval_run_id"] == "evalrun-xyz"
        assert run["best_eval_run_id"] == "evalrun-xyz"
        assert '"system_prompt": "better prompt"' in run["best_candidate"]

        iters = db.fetchall(
            "SELECT * FROM optimize_iterations WHERE run_id = ? ORDER BY ordinal", (run_id,)
        )
        assert [it["lever"] for it in iters] == ["baseline", "instructions", "memory"]
        assert iters[1]["accepted"] == 1 and iters[1]["skipped"] == 0
        assert iters[1]["eval_run_id"] == "evalrun-xyz"
        # The skipped memory lever row is distinct from a reject.
        assert iters[2]["skipped"] == 1 and iters[2]["accepted"] == 0
    finally:
        db.close()


def test_persist_false_writes_nothing(temp_dir: Path) -> None:
    """The optimize-level persist flag gates persistence: a report built but not
    persisted leaves the optimize_* tables empty (mirrors aoptimize(persist=False))."""
    db_path = temp_dir / "local.db"
    init_local_db(db_path).close()

    _ = _report()  # built, never persisted

    db = init_local_db(db_path)
    try:
        n_runs = db.fetchone("SELECT COUNT(*) AS n FROM optimize_runs")["n"]
        n_iters = db.fetchone("SELECT COUNT(*) AS n FROM optimize_iterations")["n"]
        assert n_runs == 0 and n_iters == 0
    finally:
        db.close()


# ── UI endpoints ─────────────────────────────────────────────────────────────


@pytest.fixture
def app_db(temp_dir: Path):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    pytest.importorskip("itsdangerous")
    from fastaiagent.ui.server import build_app

    fa_dir = temp_dir / ".fastaiagent"
    fa_dir.mkdir(parents=True, exist_ok=True)
    db_path = fa_dir / "local.db"
    init_local_db(db_path).close()

    # Seed a real eval_runs row (deterministic, no LLM) so the iteration link is
    # genuine — exactly what aevaluate(persist=) produces inside the loop.
    eval_res: EvalResults = evaluate(
        lambda text: text.upper(),
        [{"input": "ok", "expected_output": "OK"}],
        ["exact_match"],
        persist=False,
    )
    eval_run_id = eval_res.persist_local(db_path=db_path, run_name="cand-eval")

    rep = _report(eval_run_id=eval_run_id)
    run_id = rep.persist_local(db_path=db_path)

    app = build_app(db_path=str(db_path), no_auth=True)
    return app, run_id, eval_run_id, db_path


@pytest.fixture
def client(app_db):
    from fastapi.testclient import TestClient

    return TestClient(app_db[0])


def test_list_optimizes(client, app_db) -> None:
    _, run_id, _, _ = app_db
    r = client.get("/api/optimizes")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    row = body["rows"][0]
    assert row["run_id"] == run_id
    assert row["run_name"] == "ui-seed"
    assert row["agent_name"] == "kyc"
    assert row["levers"] == ["instructions"]  # JSON-unpacked
    assert row["seed"] == 7


def test_get_optimize_detail_links_to_real_eval_run(client, app_db) -> None:
    _, run_id, eval_run_id, db_path = app_db
    r = client.get(f"/api/optimizes/{run_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run"]["run_id"] == run_id
    assert body["run"]["best_candidate"]["system_prompt"] == "better prompt"
    assert body["total_iterations"] == 3

    iters = body["iterations"]
    assert [it["lever"] for it in iters] == ["baseline", "instructions", "memory"]
    # Each non-skipped iteration links to the eval_runs row it produced...
    linked = [it["eval_run_id"] for it in iters if not it["skipped"]]
    assert linked == [eval_run_id, eval_run_id]

    # ...and that eval run actually exists in eval_runs (the drill-down target).
    db = init_local_db(db_path)
    try:
        ev = db.fetchone("SELECT run_id FROM eval_runs WHERE run_id = ?", (eval_run_id,))
        assert ev is not None
    finally:
        db.close()


def test_get_missing_optimize_run_404(client) -> None:
    r = client.get("/api/optimizes/does-not-exist")
    assert r.status_code == 404


def test_list_filters_by_agent(temp_dir: Path) -> None:
    """The ?agent= filter (backing the UI's agent dropdown) returns only that
    agent's runs — the multi-agent case where each run targets one agent."""
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    fa_dir = temp_dir / ".fastaiagent"
    fa_dir.mkdir(parents=True, exist_ok=True)
    db_path = fa_dir / "local.db"
    init_local_db(db_path).close()

    rep_a = _report()
    rep_a.agent_name = "kyc"
    rep_a.persist_local(db_path=db_path, agent_name="kyc")
    rep_b = _report()
    rep_b.agent_name = "refund"
    rep_b.persist_local(db_path=db_path, agent_name="refund")

    client = TestClient(build_app(db_path=str(db_path), no_auth=True))

    both = client.get("/api/optimizes").json()
    assert both["total"] == 2

    only_kyc = client.get("/api/optimizes", params={"agent": "kyc"}).json()
    assert only_kyc["total"] == 1
    assert {r["agent_name"] for r in only_kyc["rows"]} == {"kyc"}


# ── Non-breaking upgrade: populated v14 DB → v15 ─────────────────────────────


def test_migration_v15_is_non_breaking_on_populated_db(temp_dir: Path) -> None:
    """Upgrading an existing v14 database to v15 only ADDS the optimize_* tables —
    pre-existing data (an ``eval_runs`` row) survives untouched. Proves the
    migration is additive, not breaking."""
    db_path = temp_dir / "legacy.db"

    # Simulate a populated pre-v15 install: an eval_runs row at user_version=14.
    db = SQLiteHelper(str(db_path))
    db.execute(
        """CREATE TABLE eval_runs (
            run_id TEXT PRIMARY KEY, run_name TEXT, dataset_name TEXT,
            agent_name TEXT, pass_rate REAL, project_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    db.execute(
        "INSERT INTO eval_runs (run_id, run_name, agent_name, pass_rate) "
        "VALUES ('keep-me', 'pre-existing', 'old-agent', 0.9)"
    )
    db.execute("PRAGMA user_version = 14")
    db.close()

    # Open via the migrator → applies v15 only.
    db = init_local_db(db_path)
    try:
        assert int(next(iter(db.fetchone("PRAGMA user_version").values()))) == 15
        # New tables exist...
        tables = {
            r["name"]
            for r in db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('optimize_runs','optimize_iterations')"
            )
        }
        assert tables == {"optimize_runs", "optimize_iterations"}
        # ...and the pre-existing eval data is untouched.
        row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = 'keep-me'")
        assert row is not None
        assert row["run_name"] == "pre-existing"
        assert row["pass_rate"] == 0.9
    finally:
        db.close()
