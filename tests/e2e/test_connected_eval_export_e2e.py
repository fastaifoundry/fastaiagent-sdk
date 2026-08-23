"""Live e2e: Agent-CI verdicts reach a connected plane (Part D, 1.49.0).

No mocks anywhere — a real key against a real plane. ``synced=1`` on the local
row is the proof: the exporter flips it only after a confirmed 2xx from
``POST /public/v1/eval/runs/ingest``.

Run against the local plane:

    zsh -lc 'FASTAIAGENT_TARGET=http://localhost:20001 \
             FASTAIAGENT_API_KEY=<key with eval:execute> \
             python -m pytest tests/e2e/test_connected_eval_export_e2e.py -m e2e'

The key must carry the **eval:execute** scope (not a default) and the domain must
be entitled to ``connected_state_plane``; both surface as a clean skip below
rather than an opaque failure.
"""

from __future__ import annotations

import os
import sqlite3
import time

import httpx
import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e

_BANNED = {"input", "expected_output", "actual_output"}

# Deliberately trivial instructions: this gate proves the export path, not model
# skill, so the assertions must not be hostage to phrasing drift.
DATASET = [
    {"input": "Reply with exactly the word: ping", "expected_output": "ping"},
    {"input": "Reply with exactly the word: pong", "expected_output": "pong"},
]


def _live_agent():
    """A REAL LLM agent — no TestModel, no stub. Keys come from the environment."""
    from fastaiagent.agent import Agent
    from fastaiagent.llm import LLMClient

    return Agent(
        name="partd-e2e-agent",
        llm=LLMClient(provider="openai", model="gpt-4o-mini", temperature=0),
        system_prompt="Follow the instruction literally. Output only the requested word.",
    )


def _synced(db_path: str, run_id: str) -> int | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT synced FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()
    return row[0] if row else None


def _drain_until_synced(db_path: str, run_id: str, timeout: float = 25.0) -> int | None:
    """Poll the exporter until the local row is acked (or we give up)."""
    from fastaiagent.eval.platform_export import get_eval_exporter

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _synced(db_path, run_id) == 1:
            return 1
        get_eval_exporter().export([])
        time.sleep(0.5)
    return _synced(db_path, run_id)


def test_gated_eval_run_reaches_the_plane(tmp_path, monkeypatch) -> None:
    require_env()
    require_platform()

    db_path = str(tmp_path / "local.db")
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", db_path)

    import fastaiagent as fa
    from fastaiagent._internal.config import reset_config
    from fastaiagent.client import _connection
    from fastaiagent.eval import evaluate, gate
    from fastaiagent.eval.platform_export import build_payloads, eval_export_enabled
    from fastaiagent.eval.results import record_gate_result

    reset_config()
    fa.connect(
        api_key=os.environ["FASTAIAGENT_API_KEY"],
        target=os.environ["FASTAIAGENT_TARGET"],
    )
    try:
        assert _connection.is_connected
        assert eval_export_enabled() is True, "export should default on when connected"

        # Entitlement / scope pre-check — clean skip rather than a red gate.
        probe = httpx.post(
            f"{_connection.target}/public/v1/eval/runs/ingest",
            headers=_connection.headers,
            json={"runs": []},
            timeout=10,
        )
        if probe.status_code == 403:
            pytest.skip(
                "eval ingest refused (403): the API key needs the 'eval:execute' "
                "scope and the domain needs connected_state_plane. "
                f"Probe body: {probe.text[:160]}"
            )
        if probe.status_code == 404:
            pytest.skip(
                "plane predates the eval ingest endpoint (needs wire v1.6) — "
                f"probe HTTP {probe.status_code}"
            )
        assert probe.status_code < 400, f"probe failed: {probe.status_code} {probe.text[:200]}"

        # A real gated run: real LLM calls, real scorers, real persistence.
        agent = _live_agent()
        results = evaluate(
            agent.run,
            DATASET,
            scorers=["contains"],
            run_name="e2e-partd",
            concurrency=2,
        )
        assert results.run_id
        assert results.errored_count == 0, f"live agent errored: {results.summary()}"
        report = gate(results, fail_under=["contains.pass_rate=0.5"], max_error_rate=0.5)
        assert report.outcome == "passed", report.describe()
        # A real trace id per case is what lets the plane corroborate the verdict
        # against traces it ingested independently.
        assert any(c.trace_id for c in results.cases), "expected trace ids from a live run"

        # The privacy contract, checked on the exact bytes that will be sent.
        payload = build_payloads(run_id=results.run_id, db_path=db_path)[0]
        assert not (_BANNED & set(payload)), "run payload leaks case content"
        for case in payload["cases"]:
            assert not (_BANNED & set(case)), "case payload leaks content"
        assert payload["gate_outcome"] == "passed"

        record_gate_result(
            results.run_id,
            gate_outcome=report.outcome,
            thresholds={"overall.pass_rate": 0.9},
        )

        assert _drain_until_synced(db_path, results.run_id) == 1, (
            "local run never acked — the plane did not confirm the push"
        )

        # At-least-once tolerance: a re-send must be safe (plane dedupes on run_id).
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE eval_runs SET synced = 0 WHERE run_id = ?", (results.run_id,))
        assert _drain_until_synced(db_path, results.run_id) == 1
    finally:
        try:
            from fastaiagent.eval.platform_export import get_eval_exporter

            get_eval_exporter().shutdown()
        finally:
            fa.disconnect()
