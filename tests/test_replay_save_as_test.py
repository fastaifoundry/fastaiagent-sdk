"""Tests for the failure-trace -> regression-test workflow.

Covers both surfaces of "Save as regression test":

* ``ReplayResult.save_as_test()`` — programmatic Python API
* ``POST /api/replay/forks/{fork_id}/save-as-test`` — UI button backend

Both must write the same JSONL schema (``input``, ``expected_output``,
``trace_id``, ``created_at``) so a regression case captured from either
path is immediately consumable by ``evaluate()``.

No mocking — these tests exercise real file I/O, a real FastAPI app,
and the real ``evaluate()`` loop with a deterministic in-process
``agent_fn`` (no LLM calls).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from fastaiagent.trace.replay import ReplayResult

# ─── Python API: ReplayResult.save_as_test() ──────────────────────────────


class TestSaveAsTestMethod:
    def test_writes_jsonl_with_eval_compatible_fields(self, tmp_path: Path) -> None:
        result = ReplayResult(
            original_output="wrong",
            new_output="right",
            steps_executed=2,
            trace_id="trace_abc123",
        )
        path = tmp_path / "regressions.jsonl"

        returned = result.save_as_test(
            path,
            input="What is our refund policy?",
            expected_output="30-day full refund.",
        )

        assert returned == path
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["input"] == "What is our refund policy?"
        assert record["expected_output"] == "30-day full refund."
        assert record["trace_id"] == "trace_abc123"
        # ISO-8601 timestamp parses cleanly
        datetime.fromisoformat(record["created_at"])

    def test_appends_does_not_overwrite(self, tmp_path: Path) -> None:
        result = ReplayResult(trace_id="t1")
        path = tmp_path / "regressions.jsonl"
        result.save_as_test(path, input="a", expected_output="A")
        result.save_as_test(path, input="b", expected_output="B")
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["input"] == "a"
        assert json.loads(lines[1])["input"] == "b"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = ReplayResult(trace_id="t1")
        path = tmp_path / "nested" / "subdir" / "regressions.jsonl"
        result.save_as_test(path, input="x", expected_output="X")
        assert path.exists()

    def test_source_trace_id_lands_in_its_own_field_v1_14_1(self, tmp_path: Path) -> None:
        """v1.14.1: ``source_trace_id`` no longer overwrites ``trace_id``.

        The original failure id and the rerun id live in distinct fields
        (``source_trace_id`` and ``fixed_trace_id`` respectively), and
        ``trace_id`` consistently means "the rerun's id" so any code
        that grew to read it during v1.13/v1.14.0 keeps seeing the same
        value when no ``source_trace_id`` is passed.
        """
        result = ReplayResult(trace_id="rerun_trace")
        path = tmp_path / "r.jsonl"
        result.save_as_test(
            path,
            input="x",
            expected_output="X",
            source_trace_id="original_failure_trace",
        )
        record = json.loads(path.read_text().splitlines()[0])
        # Pre-v1.14.1 behavior was record["trace_id"] == "original_failure_trace"
        # which conflated the two ids — fixed in v1.14.1.
        assert record["trace_id"] == "rerun_trace"
        assert record["fixed_trace_id"] == "rerun_trace"
        assert record["source_trace_id"] == "original_failure_trace"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        result = ReplayResult(trace_id="t1")
        path = tmp_path / "r.jsonl"
        result.save_as_test(str(path), input="x", expected_output="X")
        assert path.exists()


# ─── End-to-end: save_as_test() -> evaluate() ─────────────────────────────


class TestRoundTripWithEvaluate:
    """The JSONL written by save_as_test must be directly consumable by
    evaluate() with no schema massaging. This is the contract the article's
    'every failure becomes a test' workflow depends on."""

    def test_saved_case_runs_through_evaluate_and_passes_on_fix(self, tmp_path: Path) -> None:
        from fastaiagent.eval import evaluate

        # 1. Simulate the fix-and-save step: a rerun result captured as
        #    a regression case (the agent now produces the right answer).
        rerun = ReplayResult(
            original_output="WRONG",
            new_output="REFUND_OK",
            steps_executed=1,
            trace_id="trace_fix_1",
        )
        dataset_path = tmp_path / "regression_tests.jsonl"
        rerun.save_as_test(
            dataset_path,
            input="refund?",
            expected_output="REFUND_OK",
        )

        # 2. Re-running eval with a deterministic in-process agent_fn that
        #    mirrors the fixed behavior. No LLM, no network — pure logic.
        def fixed_agent(text: str) -> str:
            return "REFUND_OK" if "refund" in text.lower() else "?"

        results = evaluate(
            agent_fn=fixed_agent,
            dataset=str(dataset_path),
            scorers=["exact_match"],
            persist=False,
        )
        scored = results.scores["exact_match"]
        assert len(scored) == 1
        assert scored[0].passed is True

    def test_saved_case_fails_eval_when_regression_returns(self, tmp_path: Path) -> None:
        """The point of the regression suite: if the fix regresses, eval
        catches it. Same dataset, broken agent_fn -> failure."""
        from fastaiagent.eval import evaluate

        rerun = ReplayResult(new_output="REFUND_OK", trace_id="trace_fix_1")
        dataset_path = tmp_path / "regression_tests.jsonl"
        rerun.save_as_test(dataset_path, input="refund?", expected_output="REFUND_OK")

        def regressed_agent(_text: str) -> str:
            return "WRONG"  # the bug is back

        results = evaluate(
            agent_fn=regressed_agent,
            dataset=str(dataset_path),
            scorers=["exact_match"],
            persist=False,
        )
        scored = results.scores["exact_match"]
        assert len(scored) == 1
        assert scored[0].passed is False


# ─── UI endpoint: POST /api/replay/forks/{fork_id}/save-as-test ───────────


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.routes.replay import _clear_forks_for_tests  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def ui_env(tmp_path: Path):
    """Build a real no-auth UI app backed by a real local.db with one
    minimal agent trace inserted, so /fork can succeed."""
    fa_dir = tmp_path / ".fastaiagent"
    fa_dir.mkdir(parents=True, exist_ok=True)
    db_path = fa_dir / "local.db"
    init_local_db(db_path).close()

    trace_id = "trace_ui_save_as_test"
    with SQLiteHelper(db_path) as db:
        attrs = {
            "agent.name": "support-bot",
            "agent.input": "refund?",
            "agent.output": "WRONG",
            "agent.system_prompt": "You are a support bot.",
            "agent.config": json.dumps({}),
            "agent.tools": json.dumps([]),
            "agent.guardrails": json.dumps([]),
            "agent.llm.config": json.dumps({"provider": "openai", "model": "gpt-4o-mini"}),
        }
        db.execute(
            """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                                   start_time, end_time, status, attributes, events)
               VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]')""",
            (
                "span_root",
                trace_id,
                "agent.support-bot",
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:01Z",
                json.dumps(attrs),
            ),
        )

    _clear_forks_for_tests()
    app = build_app(db_path=str(db_path), no_auth=True)
    yield TestClient(app), trace_id
    _clear_forks_for_tests()


class TestSaveAsTestEndpoint:
    def test_endpoint_writes_jsonl_with_provenance(self, ui_env) -> None:
        client, trace_id = ui_env

        r = client.post(f"/api/replay/{trace_id}/fork", json={"step": 0})
        assert r.status_code == 200, r.text
        fork_id = r.json()["fork_id"]

        r = client.post(
            f"/api/replay/forks/{fork_id}/save-as-test",
            json={
                "input": "refund?",
                "expected_output": "REFUND_OK",
            },
        )
        assert r.status_code == 200, r.text
        out_path = Path(r.json()["path"])
        assert out_path.exists()
        assert out_path.name == "regression_tests.jsonl"

        record = json.loads(out_path.read_text().splitlines()[-1])
        assert record["input"] == "refund?"
        assert record["expected_output"] == "REFUND_OK"
        # v1.14.1: ``source_trace_id`` carries the original failure trace
        # (auto-derived from the fork). ``trace_id`` is the rerun id —
        # None here because the UI didn't pass one (it hadn't run yet).
        assert record["source_trace_id"] == trace_id
        assert record["trace_id"] is None
        assert record["fixed_trace_id"] is None
        datetime.fromisoformat(record["created_at"])

    def test_endpoint_honors_explicit_trace_id_and_dataset_path(
        self, ui_env, tmp_path: Path
    ) -> None:
        client, trace_id = ui_env
        r = client.post(f"/api/replay/{trace_id}/fork", json={"step": 0})
        fork_id = r.json()["fork_id"]

        explicit_path = tmp_path / "custom" / "regressions.jsonl"
        r = client.post(
            f"/api/replay/forks/{fork_id}/save-as-test",
            json={
                "input": "refund?",
                "expected_output": "REFUND_OK",
                "trace_id": "rerun_trace_id",
                "fork_step": 0,
                "modifications": {"prompt": "Be specific."},
                "dataset_path": str(explicit_path),
            },
        )
        assert r.status_code == 200, r.text
        record = json.loads(explicit_path.read_text().splitlines()[-1])
        # v1.14.1: the body's ``trace_id`` is the rerun id; source comes
        # from the fork automatically; modifications are recorded for
        # human inspection.
        assert record["trace_id"] == "rerun_trace_id"
        assert record["fixed_trace_id"] == "rerun_trace_id"
        assert record["source_trace_id"] == trace_id
        assert record["fork_step"] == 0
        assert record["modifications"] == {"prompt": "Be specific."}

    def test_endpoint_404s_for_unknown_fork(self, ui_env) -> None:
        client, _ = ui_env
        r = client.post(
            "/api/replay/forks/nonexistent/save-as-test",
            json={"input": "x", "expected_output": "y"},
        )
        assert r.status_code == 404
