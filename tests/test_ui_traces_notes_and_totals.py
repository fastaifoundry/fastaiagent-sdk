"""Traces list totals, trace notes, and the runner/source fields (1.89.0).

Three defects, one fixture:

* ``GET /api/traces`` counted ``total`` **before** the per-trace filters
  (``agent``, ``thread_id``, ``runner_type``/``runner_name``, duration, cost,
  tokens) ran, so "17 traces" headed a page of 3 and later pages came back
  short or empty while matches remained.
* Trace notes had a write route and nothing else: no cap, no delete, no read on
  the detail payload, and a cleared (empty) note still counted as "noted" for
  ``curate_from_traces(filter="noted")``.
* A Playground root (``playground.run``) was badged ``agent``, and the trace
  detail did not say where the run came from (``source``).

Real SQLite, real FastAPI app via ``TestClient``, hand-seeded spans. No mocks.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

_BASE = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _insert(
    db: Any,
    *,
    span_id: str,
    trace_id: str,
    parent: str | None,
    name: str,
    start: datetime,
    duration_ms: int,
    attrs: dict[str, Any],
) -> None:
    db.execute(
        """INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time,
                              end_time, status, attributes, events)
           VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
        (
            span_id,
            trace_id,
            parent,
            name,
            start.isoformat(),
            (start + timedelta(milliseconds=duration_ms)).isoformat(),
            json.dumps(attrs),
        ),
    )


def _trace(
    db: Any,
    trace_id: str,
    i: int,
    *,
    root_name: str,
    root_attrs: dict[str, Any],
    duration_ms: int,
    cost: float | None = None,
    tokens: int | None = None,
) -> None:
    start = _BASE + timedelta(minutes=i)
    _insert(
        db,
        span_id=f"{trace_id}-root",
        trace_id=trace_id,
        parent=None,
        name=root_name,
        start=start,
        duration_ms=duration_ms,
        attrs=root_attrs,
    )
    if cost is not None or tokens is not None:
        llm: dict[str, Any] = {"gen_ai.request.model": "unpriced-test-model"}
        if tokens is not None:
            llm["gen_ai.usage.input_tokens"] = tokens // 2
            llm["gen_ai.usage.output_tokens"] = tokens - tokens // 2
        if cost is not None:
            llm["cost.total_usd"] = cost
        # Inherit the root's source, as provenance stamps every span in scope.
        if "fastaiagent.source" in root_attrs:
            llm["fastaiagent.source"] = root_attrs["fastaiagent.source"]
        _insert(
            db,
            span_id=f"{trace_id}-llm",
            trace_id=trace_id,
            parent=f"{trace_id}-root",
            name="llm.call",
            start=start + timedelta(milliseconds=1),
            duration_ms=max(duration_ms - 2, 1),
            attrs=llm,
        )


#: trace_id -> what it is. 17 traces: more than five pages of 3.
ALPHA = [f"alpha-{i}" for i in range(7)]  # agent alpha, 2s, $0.5, 1000 tok
ALPHA_T1 = ALPHA[:4]  # thread t1
BETA = [f"beta-{i}" for i in range(5)]  # chain beta, 100ms, $0.01, 10 tok
PLAY = [f"play-{i}" for i in range(3)]  # playground.run, source playground
KBS = [f"kbs-{i}" for i in range(2)]  # retrieval root, source kb_search
ALL = ALPHA + BETA + PLAY + KBS


def _seed(db_path: Path) -> None:
    db = init_local_db(db_path)
    try:
        i = 0
        for tid in ALPHA:
            attrs: dict[str, Any] = {"agent.name": "alpha"}
            if tid in ALPHA_T1:
                attrs["agent.thread_id"] = "t1"
            _trace(
                db,
                tid,
                i,
                root_name="agent.alpha",
                root_attrs=attrs,
                duration_ms=2000,
                cost=0.5,
                tokens=1000,
            )
            i += 1
        for tid in BETA:
            _trace(
                db,
                tid,
                i,
                root_name="chain.beta",
                root_attrs={"fastaiagent.runner.type": "chain", "chain.name": "beta"},
                duration_ms=100,
                cost=0.01,
                tokens=10,
            )
            i += 1
        for tid in PLAY:
            # An old Playground root: no runner.type, only the name and source.
            _trace(
                db,
                tid,
                i,
                root_name="playground.run",
                root_attrs={"fastaiagent.source": "playground"},
                duration_ms=100,
            )
            i += 1
        for tid in KBS:
            _trace(
                db,
                tid,
                i,
                root_name="retrieval.docs",
                root_attrs={
                    "fastaiagent.runner.type": "retrieval",
                    "fastaiagent.source": "kb_search",
                    "retrieval.kb_name": "docs",
                },
                duration_ms=100,
            )
            i += 1
    finally:
        db.close()


@pytest.fixture
def env(isolated_local_db: Path) -> tuple[TestClient, Path]:
    """The app and curation read the same file (``FASTAIAGENT_LOCAL_DB``)."""
    _seed(isolated_local_db)
    app = build_app(db_path=str(isolated_local_db), no_auth=True)
    return TestClient(app), isolated_local_db


def _all_pages(client: TestClient, query: str, page_size: int = 3) -> tuple[list[str], set[int]]:
    """Walk every page; returns (trace ids in order, the totals each page reported)."""
    ids: list[str] = []
    totals: set[int] = set()
    page = 1
    while True:
        r = client.get(f"/api/traces?page={page}&page_size={page_size}{query}")
        assert r.status_code == 200, r.text
        body = r.json()
        totals.add(body["total"])
        rows = body["rows"]
        if not rows:
            break
        ids.extend(row["trace_id"] for row in rows)
        expected_on_page = min(page_size, body["total"] - (page - 1) * page_size)
        assert len(rows) == expected_on_page, (query, page, len(rows), body["total"])
        page += 1
        assert page < 50, "pagination did not terminate"
    return ids, totals


# --------------------------------------------------------------------------- #
# total == the rows the filter keeps, across pages
# --------------------------------------------------------------------------- #

FILTER_CASES: list[tuple[str, list[str]]] = [
    ("", ALL),  # control: no filter
    ("&agent=alpha", ALPHA),
    ("&thread_id=t1", ALPHA_T1),
    ("&runner_type=chain", BETA),
    ("&runner_type=chain&runner_name=beta", BETA),
    ("&runner_name=beta", BETA),
    ("&runner_type=agent", ALPHA),
    ("&runner_type=playground", PLAY),
    ("&runner_type=retrieval", KBS),
    ("&min_duration_ms=1000", ALPHA),
    ("&max_duration_ms=500", BETA + PLAY + KBS),
    ("&min_cost=0.1", ALPHA),
    ("&max_cost=0.1", BETA),
    ("&min_tokens=100", ALPHA),
    ("&agent=alpha&thread_id=t1&min_cost=0.1", ALPHA_T1),
    ("&agent=nobody", []),
    ("&source=playground", PLAY),
    ("&source=kb_search", KBS),
    ("&production_only=true", ALPHA + BETA),
    ("&exclude_experiments=true", ALPHA + BETA),
    ("&production_only=true&max_duration_ms=500", BETA),
]


@pytest.mark.parametrize(
    ("query", "expected"), FILTER_CASES, ids=[q or "none" for q, _ in FILTER_CASES]
)
def test_total_counts_the_filtered_rows_on_every_page(
    env: tuple[TestClient, Path], query: str, expected: list[str]
) -> None:
    client, _ = env
    ids, totals = _all_pages(client, query)
    assert totals == {len(expected)}, f"{query}: totals {totals} for {len(expected)} rows"
    assert len(ids) == len(set(ids)), f"{query}: a trace was listed twice"
    assert set(ids) == set(expected)


def test_a_page_past_the_end_is_empty_with_the_filtered_total(
    env: tuple[TestClient, Path],
) -> None:
    client, _ = env
    body = client.get("/api/traces?agent=alpha&page=4&page_size=3").json()
    assert body["rows"] == []
    assert body["total"] == len(ALPHA)


def test_filtered_order_is_newest_first(env: tuple[TestClient, Path]) -> None:
    client, _ = env
    ids, _ = _all_pages(client, "&agent=alpha")
    assert ids == list(reversed(ALPHA))  # seeded oldest first


# --------------------------------------------------------------------------- #
# (g) runner_type and source on list + detail
# --------------------------------------------------------------------------- #


def test_playground_root_is_badged_playground_not_agent(env: tuple[TestClient, Path]) -> None:
    client, _ = env
    rows = {r["trace_id"]: r for r in client.get("/api/traces?page_size=100").json()["rows"]}
    assert rows["play-0"]["runner_type"] == "playground"
    assert rows["kbs-0"]["runner_type"] == "retrieval"
    assert rows["alpha-0"]["runner_type"] == "agent"
    detail = client.get("/api/traces/play-0").json()
    assert detail["runner_type"] == "playground"


@pytest.mark.parametrize(
    ("trace_id", "source"),
    [("play-0", "playground"), ("kbs-0", "kb_search"), ("alpha-0", None)],
)
def test_detail_carries_source(
    env: tuple[TestClient, Path], trace_id: str, source: str | None
) -> None:
    client, _ = env
    detail = client.get(f"/api/traces/{trace_id}").json()
    assert "source" in detail
    assert detail["source"] == source


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #


def _note_row(db_path: Path, trace_id: str) -> dict[str, Any] | None:
    db = init_local_db(db_path)
    try:
        return db.fetchone("SELECT * FROM trace_notes WHERE trace_id = ?", (trace_id,))
    finally:
        db.close()


def _noted_ids(client: TestClient, value: str = "true") -> set[str]:
    body = client.get(f"/api/traces?noted={value}&page_size=100").json()
    return {r["trace_id"] for r in body["rows"]}


def test_note_create_read_update(env: tuple[TestClient, Path]) -> None:
    client, _ = env
    assert client.get("/api/traces/alpha-0").json()["note"] is None

    r = client.post("/api/traces/alpha-0/notes", json={"note": "wrong refund amount"})
    assert r.status_code == 200, r.text
    first = r.json()["note"]
    assert first["text"] == "wrong refund amount"
    assert first["updated_at"]

    detail = client.get("/api/traces/alpha-0").json()
    assert detail["note"] == first

    r = client.post("/api/traces/alpha-0/notes", json={"note": "fixed in v2"})
    second = r.json()["note"]
    assert second["text"] == "fixed in v2"
    assert second["updated_at"] >= first["updated_at"]
    assert client.get("/api/traces/alpha-0").json()["note"]["text"] == "fixed in v2"

    row = {r["trace_id"]: r for r in client.get("/api/traces?page_size=100").json()["rows"]}
    assert row["alpha-0"]["noted"] is True
    assert row["alpha-1"]["noted"] is False


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_an_empty_note_deletes_the_note(env: tuple[TestClient, Path], blank: str) -> None:
    client, db_path = env
    client.post("/api/traces/beta-0/notes", json={"note": "look at this"})
    assert _note_row(db_path, "beta-0") is not None

    r = client.post("/api/traces/beta-0/notes", json={"note": blank})
    assert r.status_code == 200, r.text
    assert r.json()["note"] is None
    assert _note_row(db_path, "beta-0") is None
    assert client.get("/api/traces/beta-0").json()["note"] is None
    assert "beta-0" not in _noted_ids(client)


def test_delete_route_removes_the_note(env: tuple[TestClient, Path]) -> None:
    client, db_path = env
    client.post("/api/traces/beta-1/notes", json={"note": "flaky"})
    r = client.delete("/api/traces/beta-1/notes")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True}
    assert _note_row(db_path, "beta-1") is None
    assert client.get("/api/traces/beta-1").json()["note"] is None
    # Idempotent: deleting again reports there was nothing to delete.
    assert client.delete("/api/traces/beta-1/notes").json() == {"deleted": False}


def test_note_routes_404_for_an_unknown_trace(env: tuple[TestClient, Path]) -> None:
    client, _ = env
    assert client.delete("/api/traces/nope/notes").status_code == 404
    assert client.post("/api/traces/nope/notes", json={"note": "x"}).status_code == 404


def test_note_is_capped_at_20000_characters(env: tuple[TestClient, Path]) -> None:
    client, db_path = env
    assert client.post("/api/traces/alpha-2/notes", json={"note": "x" * 20000}).status_code == 200
    r = client.post("/api/traces/alpha-2/notes", json={"note": "y" * 20001})
    assert r.status_code == 422
    row = _note_row(db_path, "alpha-2")
    assert row is not None and row["note"] == "x" * 20000


def test_noted_filter_counts_only_real_notes(env: tuple[TestClient, Path]) -> None:
    client, db_path = env
    client.post("/api/traces/alpha-3/notes", json={"note": "keep"})
    client.post("/api/traces/play-1/notes", json={"note": "prompt drifted"})
    # A blank row written before 1.89.0 (saving an empty note stored it).
    db = init_local_db(db_path)
    try:
        db.execute(
            "INSERT INTO trace_notes (trace_id, note, updated_at) VALUES (?, ?, ?)",
            ("beta-2", "  ", _BASE.isoformat()),
        )
    finally:
        db.close()

    ids, totals = _all_pages(client, "&noted=true", page_size=1)
    assert set(ids) == {"alpha-3", "play-1"}
    assert totals == {2}
    assert client.get("/api/traces/beta-2").json()["note"] is None

    not_noted, totals = _all_pages(client, "&noted=false")
    assert set(not_noted) == set(ALL) - {"alpha-3", "play-1"}
    assert totals == {len(ALL) - 2}

    # Combined with a summary filter, the total is still the kept rows.
    ids, totals = _all_pages(client, "&noted=true&agent=alpha")
    assert ids == ["alpha-3"] and totals == {1}


def test_curate_noted_ignores_blank_notes(env: tuple[TestClient, Path]) -> None:
    from fastaiagent.eval.curate import curate_from_traces

    client, db_path = env
    db = init_local_db(db_path)
    try:
        # Curation reads agent spans with an input; give two alpha traces one.
        for tid in ("alpha-4", "alpha-5"):
            db.execute(
                "UPDATE spans SET attributes = ? WHERE span_id = ?",
                (
                    json.dumps(
                        {"agent.name": "alpha", "agent.input": f"q {tid}", "agent.output": "a"}
                    ),
                    f"{tid}-root",
                ),
            )
        db.execute(
            "INSERT INTO trace_notes (trace_id, note, updated_at) VALUES (?, ?, ?)",
            ("alpha-5", "", _BASE.isoformat()),
        )
    finally:
        db.close()
    client.post("/api/traces/alpha-4/notes", json={"note": "golden answer"})

    items = curate_from_traces(filter="noted")
    assert [it["input"] for it in items] == ["q alpha-4"]
    assert items[0]["note"] == "golden answer"
