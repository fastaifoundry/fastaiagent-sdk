"""A trace is labelled by its root span, not by whichever name sorts first.

``TraceStore.list_traces`` (and ``search``, and the Local UI's Home page) took
``MIN(name)`` over the trace's spans — the *lexicographic* minimum. For a swarm
whose spans are ``['swarm.pair', 'agent.alpha']`` that is ``agent.alpha``: the
trace list named a child agent as the run. Any trace whose root span sorts after
one of its children was mislabelled, which is most of them — ``agent.*`` and
``chain.*`` both sort before ``supervisor.*`` and ``swarm.*``.

It got worse in 1.67.0, which gave root spans to three paths that previously had
none: the more roots there are, the more often the root is the row being
discarded.

The fix prefers the root (``parent_span_id IS NULL``) and falls back to
``MIN(name)`` for a trace whose root was never captured — a sampled export, a
foreign ingest, or a crash between the child's ``on_end`` and the parent's.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fastaiagent.agent import Agent, Swarm
from fastaiagent.llm.client import LLMResponse
from fastaiagent.trace.storage import TraceStore

from .conftest import MockLLMClient


@pytest.fixture
def fresh_tracing(monkeypatch, tmp_path):
    """A private local.db AND a tracer provider pointed at it.

    ``LocalStorageProcessor`` captures its db path when the provider is built,
    and the provider is a module singleton — so any earlier test that ran an
    agent under ``isolated_local_db`` pins span WRITES to a temp file that
    ``TraceStore.default()`` no longer reads. Resetting both together is the
    only way to read back a span this test just wrote inside a full-suite run.
    """
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    yield
    otel.reset()
    reset_config()


def _summary(trace_id: str):
    rows = [t for t in TraceStore.default().list_traces() if t.trace_id == trace_id]
    assert rows, f"trace {trace_id} not listed"
    return rows[0]


# ---------------------------------------------------------------------------
# The real shape that is mislabelled
# ---------------------------------------------------------------------------


def test_a_swarm_trace_lists_under_its_swarm_root(fresh_tracing) -> None:
    """Measured before the fix: ``agent.alpha``, the lexicographic minimum."""
    alpha = Agent(name="alpha", llm=MockLLMClient([LLMResponse(content="alpha done")]))
    swarm = Swarm(name="pair", agents=[alpha], entrypoint="alpha")
    result = swarm.run("hi")

    spans = TraceStore.default().get_trace(result.trace_id or "").spans
    names = sorted(sp.name for sp in spans)
    assert "swarm.pair" in names and "agent.alpha" in names, names
    assert names[0] == "agent.alpha", (
        "the setup no longer reproduces the defect — the root must NOT be the "
        f"lexicographic minimum for this test to mean anything: {names}"
    )

    assert _summary(result.trace_id or "").name == "swarm.pair", (
        "the trace list named a child agent as the run"
    )


def test_a_supervisor_trace_lists_under_its_supervisor_root(fresh_tracing) -> None:
    from fastaiagent.agent.team import Supervisor, Worker

    worker = Agent(name="aide", llm=MockLLMClient([LLMResponse(content="worker output")]))
    boss = Supervisor(
        name="boss",
        llm=MockLLMClient([LLMResponse(content="final answer")]),
        workers=[Worker(agent=worker, role="aide", description="does the work")],
    )
    result = boss.run("hi")
    assert _summary(result.trace_id or "").name == "supervisor.boss"


# ---------------------------------------------------------------------------
# The fallback: a trace whose root was never captured
# ---------------------------------------------------------------------------


def _insert(store: TraceStore, *, trace_id: str, span_id: str, name: str, parent: str | None):
    now = datetime.now(tz=timezone.utc)
    store._db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events)
           VALUES (?, ?, ?, ?, ?, ?, 'OK', '{}', '[]')""",
        (
            span_id,
            trace_id,
            parent,
            name,
            (now - timedelta(seconds=1)).isoformat(),
            now.isoformat(),
        ),
    )


def test_a_trace_with_no_root_span_still_gets_a_name(tmp_path) -> None:
    """Orphan children — a sampled export, a foreign ingest, or a crash between
    the child's ``on_end`` and the parent's. The old ``MIN(name)`` is exactly
    the right answer here, so it stays as the fallback."""
    store = TraceStore(db_path=str(tmp_path / "orphans.db"))
    _insert(store, trace_id="t-orphan", span_id="s2", name="llm.openai", parent="ffff")
    _insert(store, trace_id="t-orphan", span_id="s3", name="tool.search", parent="ffff")

    listed = {t.trace_id: t.name for t in store.list_traces()}
    assert listed["t-orphan"] == "llm.openai", listed


def test_the_root_wins_over_an_alphabetically_smaller_child(tmp_path) -> None:
    store = TraceStore(db_path=str(tmp_path / "rooted.db"))
    _insert(store, trace_id="t-rooted", span_id="s1", name="swarm.pair", parent=None)
    _insert(store, trace_id="t-rooted", span_id="s2", name="agent.alpha", parent="s1")

    listed = {t.trace_id: t.name for t in store.list_traces()}
    assert listed["t-rooted"] == "swarm.pair", listed


def test_search_labels_a_trace_the_same_way_as_list(tmp_path) -> None:
    """``search`` carries a second copy of the same aggregate. Two copies of one
    query is how the first one got fixed and the second did not."""
    store = TraceStore(db_path=str(tmp_path / "searched.db"))
    # Both names match the query, so the WHERE clause cannot do the fix's job
    # for it by filtering the child out of the group.
    _insert(store, trace_id="t-search", span_id="s1", name="swarm.pair-run", parent=None)
    _insert(store, trace_id="t-search", span_id="s2", name="agent.alpha-run", parent="s1")

    hits = {t.trace_id: t.name for t in store.search("-run")}
    assert hits["t-search"] == "swarm.pair-run", hits


# ---------------------------------------------------------------------------
# The Local UI's Home page carries a third copy
# ---------------------------------------------------------------------------


def test_the_ui_home_recent_traces_label_matches(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    db_path = tmp_path / "ui.db"
    store = TraceStore(db_path=str(db_path))
    _insert(store, trace_id="t-ui", span_id="s1", name="swarm.pair", parent=None)
    _insert(store, trace_id="t-ui", span_id="s2", name="agent.alpha", parent="s1")

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    from fastaiagent._internal.config import reset_config

    reset_config()
    try:
        client = TestClient(build_app(db_path=str(db_path), no_auth=True))
        body = client.get("/api/overview").json()
    finally:
        reset_config()
    recent = {t["trace_id"]: t["name"] for t in body.get("recent_traces", [])}
    assert recent.get("t-ui") == "swarm.pair", recent
