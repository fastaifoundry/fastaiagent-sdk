"""What an eval run sends the plane about verdicts (1.89.0).

The user's rule: no secret reaches the plane or the logs. A judge's ``reason``
quotes the case and can quote an error carrying a key, so on every path that
sends it (the Agent-CI ingest, ``EvalResults.publish()``) it is scrubbed of
credentials, and withheld entirely under ``FASTAIAGENT_TRACE_PAYLOADS=0``, the
same switch that keeps span payloads local. A case's error text likewise.

Also: no new top-level keys (D11). ``pass_*`` count checks (applicable
verdicts), ``errored_count`` counts cases whose outcome is errored, including a
scorer that could not run. Real SQLite, a real localhost capture server; no
mocks.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.client import _connection
from fastaiagent.eval.platform_export import build_payloads
from fastaiagent.eval.results import EvalCaseRecord, EvalResults, record_gate_result
from fastaiagent.eval.scorer import ScorerResult

KEY = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@pytest.fixture(autouse=True)
def _reset_connection():  # noqa: ANN202
    saved = (
        _connection.api_key,
        _connection.target,
        _connection.project,
        _connection.project_id,
        _connection.export_evals,
    )
    yield
    (
        _connection.api_key,
        _connection.target,
        _connection.project,
        _connection.project_id,
        _connection.export_evals,
    ) = saved


def _results() -> EvalResults:
    er = EvalResults()
    verdicts = [
        ("q1", ScorerResult(score=1.0, passed=True, reason="The answer '$10' matches.")),
        ("q2", ScorerResult.error(f"Judge error: 401 bad key {KEY}")),
        ("q3", ScorerResult.not_applicable("No expected output")),
    ]
    for q, r in verdicts:
        er.add("judge", r)
        er.add_case(EvalCaseRecord(input=q, per_scorer={"judge": r.stored()}))
    er.add_case(EvalCaseRecord(input="q4", error=f"RuntimeError: upstream said {KEY}"))
    return er


def _seed(db: Path) -> str:
    saved = _connection.export_evals
    _connection.export_evals = False
    try:
        run_id = _results().persist_local(db_path=db, run_name="reasons")
        record_gate_result(run_id, gate_outcome="invalid", thresholds={}, db_path=db)
    finally:
        _connection.export_evals = saved
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE eval_runs SET synced = 0 WHERE run_id = ?", (run_id,))
    return run_id


def _case_by_ordinal(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {c["ordinal"]: c for c in run["cases"]}


def test_the_key_never_reaches_local_db(isolated_local_db: Path) -> None:
    _seed(isolated_local_db)
    with sqlite3.connect(isolated_local_db) as conn:
        dump = "\n".join(str(r) for r in conn.execute("SELECT per_scorer, error FROM eval_cases"))
    assert KEY not in dump
    assert "[REDACTED]" in dump


def test_payload_counts_and_scrubbed_reasons(isolated_local_db: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("FASTAIAGENT_TRACE_PAYLOADS", raising=False)
    run_id = _seed(isolated_local_db)
    run = build_payloads(run_id=run_id, db_path=isolated_local_db)[0]
    assert (run["pass_count"], run["fail_count"]) == (1, 0), "checks: applicable verdicts only"
    assert run["errored_count"] == 3, "agent error + scorer error + nothing applied"
    cases = _case_by_ordinal(run)
    assert cases[0]["per_scorer"]["judge"]["reason"] == "The answer '$10' matches."
    assert KEY not in str(run)
    assert "[REDACTED]" in cases[1]["per_scorer"]["judge"]["reason"]
    assert cases[1]["per_scorer"]["judge"]["errored"] is True
    assert cases[3]["error"] and KEY not in cases[3]["error"]


def test_payloads_off_withholds_reason_and_error(isolated_local_db: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    run_id = _seed(isolated_local_db)
    run = build_payloads(run_id=run_id, db_path=isolated_local_db)[0]
    cases = _case_by_ordinal(run)
    assert all(
        entry.get("reason") is None
        for c in cases.values()
        for entry in (c["per_scorer"] or {}).values()
    )
    assert cases[3]["error"] == "errored (detail withheld: FASTAIAGENT_TRACE_PAYLOADS=0)"
    assert "$10" not in str(run)


def test_publish_sends_three_keys_scrubbed_and_no_skipped(
    isolated_local_db: Path,
    capture_server: Any,
    monkeypatch,  # noqa: ANN001
) -> None:
    monkeypatch.delenv("FASTAIAGENT_TRACE_PAYLOADS", raising=False)
    _connection.api_key = "test-key"
    _connection.target = capture_server.url
    _connection.project = "test-proj"
    _connection.project_id = "test-proj"
    _results().publish(run_name="legacy")
    (req,) = [r for r in capture_server.requests if r["path"].endswith("/public/v1/eval/runs")]
    sent = req["body"]["scores"]["judge"]
    assert len(sent) == 2, "the skipped verdict is not a case to the legacy endpoint"
    assert all(set(e) == {"score", "passed", "reason"} for e in sent)
    assert KEY not in str(req["body"])

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    _results().publish(run_name="legacy-off")
    last = [r for r in capture_server.requests if r["path"].endswith("/public/v1/eval/runs")][-1]
    assert all(e["reason"] is None for e in last["body"]["scores"]["judge"])
