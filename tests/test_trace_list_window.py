"""``TraceStore.list_traces`` honours its arguments (1.81.0).

It accepted ``last_hours`` and ``**filters`` and ignored both: every call
returned the newest 100 traces of all time. ``fastaiagent traces list
--last-hours`` promised a window it never applied, ``run_extraction`` mined
traces from any date, and the replay examples' ``name_filter`` / ``limit``
did nothing. Real SQLite; spans are inserted with chosen timestamps.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fastaiagent.trace.storage import TraceStore


def _span(
    store: TraceStore,
    *,
    trace_id: str,
    name: str,
    hours_ago: float = 0.0,
    parent: str | None = None,
    attrs: dict | None = None,
) -> None:
    start = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    store._db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events)
           VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
        (
            f"{trace_id}:{name}",
            trace_id,
            parent,
            name,
            start.isoformat(),
            (start + timedelta(seconds=1)).isoformat(),
            json.dumps(attrs or {}),
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(db_path=str(tmp_path / "traces.db"))


def test_last_hours_excludes_older_traces(store):
    _span(store, trace_id="fresh", name="agent.a", hours_ago=1)
    _span(store, trace_id="stale", name="agent.a", hours_ago=240)

    assert {t.trace_id for t in store.list_traces()} == {"fresh"}  # default: 24h
    assert {t.trace_id for t in store.list_traces(last_hours=500)} == {"fresh", "stale"}
    assert {t.trace_id for t in store.list_traces(last_hours=None)} == {"fresh", "stale"}


def test_limit_and_name_filter(store):
    _span(store, trace_id="t1", name="agent.a", hours_ago=3)
    _span(store, trace_id="t2", name="chain.sales-sdr", hours_ago=2)
    _span(store, trace_id="t3", name="agent.a", hours_ago=1)

    assert [t.trace_id for t in store.list_traces(limit=1)] == ["t3"]  # newest first
    assert [t.trace_id for t in store.list_traces(name_filter="chain.sales")] == ["t2"]


def test_agent_name_matches_any_span_in_the_trace(store):
    """A swarm's agents run as child spans, so the filter looks at every span."""
    _span(store, trace_id="swarm-run", name="swarm.s")
    _span(
        store,
        trace_id="swarm-run",
        name="agent.alpha",
        parent="swarm-run:swarm.s",
        attrs={"agent.name": "alpha"},
    )
    _span(store, trace_id="beta-run", name="agent.beta", attrs={"agent.name": "beta"})

    assert [t.trace_id for t in store.list_traces(agent_name="alpha")] == ["swarm-run"]


def test_unknown_filters_warn(store):
    with pytest.warns(UserWarning, match="bogus"):
        store.list_traces(bogus=1)


def test_search_without_a_query_still_covers_all_time(store):
    _span(store, trace_id="stale", name="agent.a", hours_ago=240)
    assert [t.trace_id for t in store.search()] == ["stale"]


def test_cli_traces_list_applies_last_hours(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from fastaiagent._internal.config import reset_config
    from fastaiagent.cli.main import app

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    try:
        s = TraceStore(db_path=str(db))
        _span(s, trace_id="freshtrace-000", name="agent.fresh", hours_ago=0.2)
        _span(s, trace_id="staletrace-000", name="agent.stale", hours_ago=48)
        out = CliRunner().invoke(app, ["traces", "list", "--last-hours", "1"]).output
        # The table shows the first 12 characters of each trace id.
        assert "freshtrace-0" in out and "staletrace-0" not in out
    finally:
        reset_config()
