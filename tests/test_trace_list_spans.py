"""``TraceStore.list_spans`` — span-level reads for tailing as spans land.

Real SQLite, no mocking. Every assertion goes through the public method against
a store backed by a temp-file database, the same code path a caller gets.

Why this method exists: every "show spans as they arrive" use case previously
had to walk trace by trace. Two streaming demos assumed a method like this and
were dead because of it, and example 08 dropped to raw ``sqlite3``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from fastaiagent.trace.storage import SpanData, SpanRecord, TraceStore


def _insert(
    store: TraceStore,
    *,
    trace_id: str,
    span_id: str,
    name: str = "span",
    start: str | None = None,
    attributes: dict | None = None,
) -> None:
    """Write one span row directly, the way ``on_end`` does."""
    store._db.execute(
        "INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time, "
        "end_time, status, attributes, events, project_id, synced) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            span_id,
            trace_id,
            None,
            name,
            start or datetime.now(tz=timezone.utc).isoformat(),
            datetime.now(tz=timezone.utc).isoformat(),
            "OK",
            json.dumps(attributes or {}),
            "[]",
            "",
            0,
        ),
    )


@pytest.fixture
def store(tmp_path):
    s = TraceStore(db_path=str(tmp_path / "local.db"))
    yield s
    s.close()


def test_returns_span_records_oldest_first(store: TraceStore) -> None:
    for i in range(5):
        _insert(store, trace_id="t1", span_id=f"s{i}", name=f"n{i}")

    recs = store.list_spans()

    assert [r.span.name for r in recs] == ["n0", "n1", "n2", "n3", "n4"]
    assert all(isinstance(r, SpanRecord) for r in recs)
    assert all(isinstance(r.span, SpanData) for r in recs)
    assert [r.cursor for r in recs] == sorted(r.cursor for r in recs)


def test_cursor_resumes_without_repeating_or_skipping(store: TraceStore) -> None:
    """Paging with the returned cursor must read every span exactly once."""
    for i in range(10):
        _insert(store, trace_id="t1", span_id=f"s{i}", name=f"n{i}")

    seen: list[str] = []
    cursor = 0
    while True:
        page = store.list_spans(since=cursor, limit=3)
        if not page:
            break
        seen.extend(r.span.name for r in page)
        cursor = page[-1].cursor

    assert seen == [f"n{i}" for i in range(10)]
    assert len(seen) == len(set(seen)), "a span was read twice"


def test_spans_written_after_a_read_are_picked_up(store: TraceStore) -> None:
    """The tail case: new rows land between calls and must appear next call."""
    _insert(store, trace_id="t1", span_id="s0", name="first")
    first = store.list_spans()
    assert [r.span.name for r in first] == ["first"]

    _insert(store, trace_id="t1", span_id="s1", name="second")
    later = store.list_spans(since=first[-1].cursor)

    assert [r.span.name for r in later] == ["second"]


def test_ordering_is_insert_order_not_start_time(store: TraceStore) -> None:
    """A span that *started* earlier but was written later must not be skipped.

    This is why the cursor is a rowid and not a timestamp. A long span starts
    before a short one that finishes first, so it is written second while
    carrying the earlier ``start_time``. Ordering by ``start_time`` would step
    over it once the cursor had passed that instant.
    """
    now = datetime.now(tz=timezone.utc)
    _insert(store, trace_id="t1", span_id="short", name="short", start=now.isoformat())
    _insert(
        store,
        trace_id="t1",
        span_id="long",
        name="long",
        start=(now - timedelta(minutes=5)).isoformat(),
    )

    page = store.list_spans(limit=1)
    rest = store.list_spans(since=page[-1].cursor)

    assert [r.span.name for r in page] == ["short"]
    assert [r.span.name for r in rest] == ["long"], (
        "the late-written span with an earlier start_time was skipped"
    )


def test_trace_id_filter_scopes_to_one_trace(store: TraceStore) -> None:
    _insert(store, trace_id="t1", span_id="a")
    _insert(store, trace_id="t2", span_id="b")
    _insert(store, trace_id="t1", span_id="c")

    recs = store.list_spans(trace_id="t1")

    assert [r.span.span_id for r in recs] == ["a", "c"]


def test_execution_id_filter_matches_the_chain_attribute(store: TraceStore) -> None:
    """A durable run is followed across every trace it spans."""
    _insert(
        store,
        trace_id="t1",
        span_id="a",
        attributes={"chain.execution_id": "run-1"},
    )
    _insert(
        store,
        trace_id="t2",
        span_id="b",
        attributes={"chain.execution_id": "run-1"},
    )
    _insert(
        store,
        trace_id="t3",
        span_id="c",
        attributes={"chain.execution_id": "run-2"},
    )
    _insert(store, trace_id="t4", span_id="d", attributes={})

    recs = store.list_spans(execution_id="run-1")

    assert [r.span.span_id for r in recs] == ["a", "b"]


def test_limit_caps_the_page(store: TraceStore) -> None:
    for i in range(7):
        _insert(store, trace_id="t1", span_id=f"s{i}")

    assert len(store.list_spans(limit=3)) == 3
    assert len(store.list_spans(limit=100)) == 7


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_returns_nothing(store: TraceStore, limit: int) -> None:
    _insert(store, trace_id="t1", span_id="a")

    assert store.list_spans(limit=limit) == []


def test_empty_page_means_nothing_new_not_end_of_stream(store: TraceStore) -> None:
    _insert(store, trace_id="t1", span_id="a")
    page = store.list_spans()

    assert store.list_spans(since=page[-1].cursor) == []

    _insert(store, trace_id="t1", span_id="b")
    assert [r.span.span_id for r in store.list_spans(since=page[-1].cursor)] == ["b"]


def test_empty_store_returns_empty_list(store: TraceStore) -> None:
    assert store.list_spans() == []


def test_attributes_and_events_are_deserialized(store: TraceStore) -> None:
    _insert(store, trace_id="t1", span_id="a", attributes={"gen_ai.system": "openai"})

    (rec,) = store.list_spans()

    assert rec.span.attributes == {"gen_ai.system": "openai"}
    assert rec.span.events == []


def test_cursor_is_not_a_field_on_the_egress_model() -> None:
    """The wire payload must not gain a key for a local read cursor.

    ``trace.platform_export.to_wire`` dumps ``SpanData`` whole, so a field added
    there becomes a new key on the span payload the plane receives — a wire
    event under the cross-repo contract (CLAUDE.md §2.2), for something the
    plane has no use for.
    """
    assert "cursor" not in SpanData.model_fields
    assert "rowid" not in SpanData.model_fields
