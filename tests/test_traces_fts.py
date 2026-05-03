"""FTS5-backed trace search tests (Sprint 3).

Real SQLite + real FastAPI TestClient + real triggers. The triggers
fire on INSERT/UPDATE/DELETE in ``spans``, so every assertion below
exercises the actual sync path.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _insert_span(
    db,
    *,
    trace_id: str,
    span_id: str,
    name: str,
    attributes: dict | None = None,
    project_id: str = "",
) -> None:
    db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            span_id,
            trace_id,
            None,
            name,
            _now(),
            _now(),
            "OK",
            json.dumps(attributes or {}),
            "[]",
            project_id,
        ),
    )


@pytest.fixture
def app_db(temp_dir: Path):
    db_path = temp_dir / "local.db"
    db = init_local_db(db_path)
    db.close()
    app = build_app(db_path=str(db_path), no_auth=True)
    return app, db_path


@pytest.fixture
def client(app_db):
    app, _ = app_db
    return TestClient(app)


def _open(db_path: Path):
    return init_local_db(db_path)


# ---------------------------------------------------------------------------
# 1. The FTS table + triggers exist after init_local_db.
# ---------------------------------------------------------------------------


class TestFtsSchema:
    def test_span_fts_table_and_triggers_present(self, app_db) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            tbl = db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='span_fts'"
            )
            assert tbl, "span_fts virtual table not created"
            triggers = {
                r["name"]
                for r in db.fetchall(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            assert {"spans_fts_ai", "spans_fts_au", "spans_fts_ad"}.issubset(triggers)
            v = db.fetchone("PRAGMA user_version")
            assert v["user_version"] >= 6
        finally:
            db.close()


# ---------------------------------------------------------------------------
# 2. Insert → search by gen_ai.prompt finds the trace.
# ---------------------------------------------------------------------------


class TestInsertTrigger:
    def test_search_finds_inserted_prompt(
        self, client: TestClient, app_db
    ) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            _insert_span(
                db,
                trace_id="t-refund",
                span_id="s-1",
                name="llm.gpt",
                attributes={
                    "agent.name": "support",
                    "gen_ai.prompt": "What is your refund policy?",
                },
            )
            _insert_span(
                db,
                trace_id="t-shipping",
                span_id="s-2",
                name="llm.gpt",
                attributes={
                    "agent.name": "support",
                    "gen_ai.prompt": "When does shipping arrive?",
                },
            )
        finally:
            db.close()

        r = client.get("/api/traces", params={"q": "refund"})
        assert r.status_code == 200, r.text
        ids = [row["trace_id"] for row in r.json()["rows"]]
        assert ids == ["t-refund"]

    def test_search_matches_response_text_too(
        self, client: TestClient, app_db
    ) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            _insert_span(
                db,
                trace_id="t-r1",
                span_id="r-1",
                name="llm.gpt",
                attributes={
                    "gen_ai.response.text": "Refunds processed within 14 days.",
                },
            )
        finally:
            db.close()

        r = client.get("/api/traces", params={"q": "Refunds processed"})
        assert r.status_code == 200
        ids = [row["trace_id"] for row in r.json()["rows"]]
        assert ids == ["t-r1"]


# ---------------------------------------------------------------------------
# 3. Update span → old text gone, new text matches.
# ---------------------------------------------------------------------------


class TestUpdateTrigger:
    def test_update_resyncs_fts(self, client: TestClient, app_db) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            _insert_span(
                db,
                trace_id="t-up",
                span_id="u-1",
                name="llm.gpt",
                attributes={"gen_ai.prompt": "Tell me about quokkas."},
            )
        finally:
            db.close()

        # Old term matches.
        r = client.get("/api/traces", params={"q": "quokkas"})
        assert [row["trace_id"] for row in r.json()["rows"]] == ["t-up"]

        # Update the span's attributes.
        db = _open(db_path)
        try:
            db.execute(
                "UPDATE spans SET attributes = ? WHERE span_id = ?",
                (
                    json.dumps({"gen_ai.prompt": "Tell me about wombats."}),
                    "u-1",
                ),
            )
        finally:
            db.close()

        # Old term no longer matches.
        old = client.get("/api/traces", params={"q": "quokkas"}).json()
        assert [row["trace_id"] for row in old["rows"]] == []
        # New term does.
        new = client.get("/api/traces", params={"q": "wombats"}).json()
        assert [row["trace_id"] for row in new["rows"]] == ["t-up"]


# ---------------------------------------------------------------------------
# 4. Delete span → FTS row gone too.
# ---------------------------------------------------------------------------


class TestDeleteTrigger:
    def test_delete_drops_fts_row(self, app_db) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            _insert_span(
                db,
                trace_id="t-del",
                span_id="d-1",
                name="llm.gpt",
                attributes={"gen_ai.prompt": "ephemeralword"},
            )
            before = db.fetchone(
                "SELECT COUNT(*) AS n FROM span_fts WHERE span_id = ?",
                ("d-1",),
            )
            assert before["n"] == 1
            db.execute("DELETE FROM spans WHERE span_id = ?", ("d-1",))
            after = db.fetchone(
                "SELECT COUNT(*) AS n FROM span_fts WHERE span_id = ?",
                ("d-1",),
            )
            assert after["n"] == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# 5. FTS escaping — apostrophes and FTS metacharacters don't crash.
# ---------------------------------------------------------------------------


class TestEscaping:
    def test_apostrophe_does_not_crash(self, client: TestClient) -> None:
        # Returns 200 with possibly-empty rows; the key is that FTS5
        # doesn't throw a SyntaxError on the embedded apostrophe.
        r = client.get("/api/traces", params={"q": "O'Brien"})
        assert r.status_code == 200

    def test_asterisk_metachar_does_not_crash(self, client: TestClient) -> None:
        r = client.get("/api/traces", params={"q": "*"})
        assert r.status_code == 200

    def test_double_quote_does_not_crash(self, client: TestClient) -> None:
        r = client.get("/api/traces", params={"q": '"injection"'})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 6. Performance smoke — 1k spans + search returns within 1s.
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_thousand_spans_search_under_one_second(
        self, client: TestClient, app_db
    ) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            for i in range(1000):
                _insert_span(
                    db,
                    trace_id=f"t-{i}",
                    span_id=f"s-{i}",
                    name="llm.gpt",
                    attributes={
                        "gen_ai.prompt": (
                            "needle" if i == 555 else f"haystack token {i}"
                        ),
                    },
                )
        finally:
            db.close()

        start = time.perf_counter()
        r = client.get("/api/traces", params={"q": "needle"})
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        ids = [row["trace_id"] for row in r.json()["rows"]]
        assert ids == ["t-555"]
        # Tighter than the 2s spec budget — FTS5 should be sub-second
        # on 1k rows. Generous because CI shared runners can be slow.
        assert elapsed < 1.5, f"FTS query took {elapsed:.2f}s on 1k spans"


# ---------------------------------------------------------------------------
# 7. max_cost filter — added in Sprint 3.
# ---------------------------------------------------------------------------


class TestMaxCost:
    def test_max_cost_filters_out_expensive_traces(
        self, client: TestClient, app_db
    ) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            _insert_span(
                db,
                trace_id="cheap",
                span_id="c-1",
                name="agent.demo",
                attributes={
                    "agent.name": "x",
                    "fastaiagent.cost.total_usd": 0.01,
                },
            )
            _insert_span(
                db,
                trace_id="pricey",
                span_id="p-1",
                name="agent.demo",
                attributes={
                    "agent.name": "x",
                    "fastaiagent.cost.total_usd": 0.50,
                },
            )
        finally:
            db.close()

        r = client.get("/api/traces", params={"max_cost": 0.05})
        ids = sorted(row["trace_id"] for row in r.json()["rows"])
        assert ids == ["cheap"]

        # Both pass when no cap is set.
        r2 = client.get("/api/traces")
        ids2 = sorted(row["trace_id"] for row in r2.json()["rows"])
        assert ids2 == ["cheap", "pricey"]
