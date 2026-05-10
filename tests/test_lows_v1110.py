"""Regression tests for the v1.11.1 Low-severity fixes.

* L1 — SQL LIKE escape on prompt name (``%``/``_``/``\\``).
* L3 — Filter preset JSON validated through a permissive Pydantic
  schema; unknown extra fields round-trip unchanged.
* L6 — ``fastaiagent traces purge`` CLI deletes scoped trace data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


# ---------------------------------------------------------------------------
# L1 — SQL LIKE escape on prompt name
# ---------------------------------------------------------------------------


def _seed_two_prompts_with_like_chars(db_path: Path) -> None:
    """Seed two prompts whose names share a LIKE-wildcard relationship.

    ``my_prompt`` (literal underscore) and ``myAprompt`` (where ``_``
    matches one char). Pre-L1 code escaped neither, so a search for
    ``my_prompt``'s linked traces would over-match ``myAprompt``'s
    span and inflate the count.
    """
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    try:
        for slug in ("my_prompt", "myAprompt"):
            db.execute(
                "INSERT INTO prompts (slug, latest_version, created_at, updated_at) "
                "VALUES (?, '1', ?, ?)",
                (slug, now, now),
            )
            db.execute(
                "INSERT INTO prompt_versions "
                "(slug, version, template, variables, fragments, metadata, "
                " created_at, created_by) VALUES (?, '1', 'x', '[]', '[]', '{}', ?, 'code')",
                (slug, now),
            )
        # ONE span belongs to ``myAprompt`` (the foil). Pre-L1, a
        # search keyed on ``my_prompt`` over-matched this span.
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, NULL, 'agent.x', ?, ?, 'OK', ?, '[]')""",
            (
                "span-foil",
                "trace-foil",
                now,
                now,
                json.dumps({"fastaiagent.prompt.name": "myAprompt"}),
            ),
        )
    finally:
        db.close()


def test_l1_like_escape_does_not_overmatch_underscore(tmp_path: Path) -> None:
    """``my_prompt`` must report 0 linked traces — its single span row
    belongs to ``myAprompt``. Pre-L1 it reported 1 because the literal
    ``_`` in the name matched the LIKE wildcard.
    """
    db_path = tmp_path / "local.db"
    _seed_two_prompts_with_like_chars(db_path)

    app = build_app(db_path=str(db_path), no_auth=True)
    with TestClient(app) as client:
        r = client.get("/api/prompts")
    assert r.status_code == 200, r.text
    by_name = {row["name"]: row for row in r.json()["rows"]}
    assert by_name["my_prompt"]["linked_trace_count"] == 0, (
        "L1 regression: ``my_prompt`` over-matched ``myAprompt`` via "
        "the unescaped LIKE wildcard"
    )
    assert by_name["myAprompt"]["linked_trace_count"] == 1


# ---------------------------------------------------------------------------
# L3 — Filter preset Pydantic schema (permissive)
# ---------------------------------------------------------------------------


def test_l3_filter_preset_known_keys_typed() -> None:
    from fastaiagent.ui.routes.filter_presets import FilterValues

    fv = FilterValues.model_validate(
        {"agent": "support-bot", "model": "gpt-4o", "has_error": True}
    )
    dumped = fv.model_dump(exclude_none=True)
    assert dumped == {"agent": "support-bot", "model": "gpt-4o", "has_error": True}


def test_l3_filter_preset_extra_keys_pass_through() -> None:
    """``extra='allow'`` keeps future or hand-edited keys flowing
    through unchanged so a preset stored under a previous schema still
    round-trips.
    """
    from fastaiagent.ui.routes.filter_presets import FilterValues

    fv = FilterValues.model_validate(
        {"agent": "x", "future_field": "future_value", "nested": {"k": "v"}}
    )
    dumped = fv.model_dump()
    assert dumped["future_field"] == "future_value"
    assert dumped["nested"] == {"k": "v"}


def test_l3_filter_preset_create_then_read_round_trip(tmp_path: Path) -> None:
    """Mirror what the UI does: POST a preset, list it back, confirm
    the filter dict survives.
    """
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    app = build_app(db_path=str(db_path), no_auth=True)
    with TestClient(app) as client:
        r = client.post(
            "/api/filter-presets",
            json={
                "name": "my preset",
                "filters": {"agent": "x", "custom_extra": "value"},
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["filters"]["agent"] == "x"
        listed = client.get("/api/filter-presets")
        assert listed.status_code == 200
        rows = listed.json()
        assert any(p["name"] == "my preset" for p in rows)
        # The custom_extra key flows through unchanged thanks to
        # ``extra='allow'`` on FilterValues.
        match = next(p for p in rows if p["name"] == "my preset")
        assert match["filters"]["custom_extra"] == "value"


# ---------------------------------------------------------------------------
# L6 — ``fastaiagent traces purge`` CLI
# ---------------------------------------------------------------------------


def _seed_two_traces(db_path: Path) -> None:
    """Insert one OLD root span (40 days ago) and one RECENT (today)."""
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc)
    old_iso = (now - timedelta(days=40)).isoformat()
    new_iso = now.isoformat()
    try:
        for trace_id, sid, start in [
            ("trace-old", "span-old", old_iso),
            ("trace-new", "span-new", new_iso),
        ]:
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events)
                   VALUES (?, ?, NULL, 'agent.x', ?, ?, 'OK', '{}', '[]')""",
                (sid, trace_id, start, start),
            )
    finally:
        db.close()


def test_l6_purge_deletes_only_older_than_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent.cli.traces import purge_traces

    db_path = tmp_path / "local.db"
    _seed_two_traces(db_path)
    # Point the resolved trace path at our temp file.
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    from fastaiagent._internal.config import reset_config

    reset_config()

    purge_traces(older_than_days=30, attachments=False, yes=True)

    with SQLiteHelper(db_path) as db:
        rows = db.fetchall("SELECT trace_id FROM spans ORDER BY trace_id")
    trace_ids = {r["trace_id"] for r in rows}
    assert trace_ids == {"trace-new"}, (
        "L6 regression: purge --older-than-days 30 must keep recent traces "
        f"and drop the old one. saw: {trace_ids}"
    )


def test_l6_purge_aborts_when_user_says_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--yes`` and a ``no`` at the prompt, purge must exit
    non-zero and leave the DB untouched.
    """
    import typer

    from fastaiagent.cli.traces import purge_traces

    db_path = tmp_path / "local.db"
    _seed_two_traces(db_path)
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    from fastaiagent._internal.config import reset_config

    reset_config()

    # Force the typer.confirm() prompt to return False.
    monkeypatch.setattr("typer.confirm", lambda *a, **kw: False)

    with pytest.raises(typer.Exit) as excinfo:
        purge_traces(older_than_days=None, attachments=False, yes=False)
    assert excinfo.value.exit_code == 1

    with SQLiteHelper(db_path) as db:
        n = db.fetchone("SELECT COUNT(*) AS n FROM spans")
    assert n is not None and n["n"] == 2, "DB must be untouched on abort"
