"""``ForkedReplay.compare`` compares the run's turns, not every span (1.89.0).

A memory agent's original trace carries ``memory.read`` / ``memory.write``
spans; its rerun is rebuilt without memory, so those spans are missing and every
later index shifted. An identical recorded rerun therefore "diverged" at step 1.
``compare`` now aligns the root and its direct, non-memory children; tool calls
are diffed separately, so a blocked tool does not hide that the model's answers
matched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def traced_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET", "FASTAIAGENT_REPLAY_TOOL_POLICY"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def test_a_memory_agents_recorded_rerun_does_not_diverge(traced_db: Path) -> None:
    import fastaiagent as fa
    from fastaiagent.agent.memory import AgentMemory
    from fastaiagent.testing import TestModel
    from fastaiagent.trace.replay import Replay

    agent = fa.Agent(
        name="mem-agent",
        system_prompt="s",
        llm=TestModel(response="hello there"),
        memory=AgentMemory(),
    )
    original = agent.run("hi")
    replay = Replay.load(original.trace_id)
    assert any(s.span_name.startswith("memory.") for s in replay.steps()), (
        "precondition: the original carries memory spans"
    )
    forked = replay.fork_at(0).with_determinism("recorded")
    rerun = forked.rerun()
    comparison = forked.compare(rerun)

    assert comparison.compare_status == "ok"
    assert comparison.diverged_at is None, "an identical recorded rerun reported a divergence"
    assert comparison.new_diverged_at is None
    assert comparison.skipped_steps, "the memory spans should be skipped, not compared"
    assert all(replay.steps()[i].span_name.startswith("memory.") for i in comparison.skipped_steps)


def _coarse_clock(db: Path, trace_id: str) -> None:
    """Give every span of a trace one start time, as Windows' clock often does."""
    import sqlite3

    with sqlite3.connect(db) as conn:
        first = conn.execute(
            "SELECT MIN(start_time) FROM spans WHERE trace_id = ?", (trace_id,)
        ).fetchone()[0]
        conn.execute("UPDATE spans SET start_time = ? WHERE trace_id = ?", (first, trace_id))


def test_tied_start_times_keep_parents_first_and_do_not_diverge(traced_db: Path) -> None:
    """Failed on Windows CI: the tie fell to write order, children first."""
    import fastaiagent as fa
    from fastaiagent.agent.memory import AgentMemory
    from fastaiagent.testing import TestModel
    from fastaiagent.trace.replay import Replay
    from fastaiagent.trace.storage import TraceStore

    agent = fa.Agent(
        name="mem-agent",
        system_prompt="s",
        llm=TestModel(response="hello there"),
        memory=AgentMemory(),
    )
    original = agent.run("hi")
    _coarse_clock(traced_db, original.trace_id)
    trace = TraceStore.default().get_trace(original.trace_id)
    assert trace.name == "agent.mem-agent", "a child span was taken as the root"
    ids = [s.span_id for s in trace.spans]
    for s in trace.spans:
        if s.parent_span_id in ids:
            assert ids.index(s.parent_span_id) < ids.index(s.span_id), "child before parent"

    forked = Replay.load(original.trace_id).fork_at(0).with_determinism("recorded")
    rerun = forked.rerun()
    _coarse_clock(traced_db, rerun.trace_id)
    comparison = forked.compare(rerun)
    assert comparison.diverged_at is None, "an identical recorded rerun reported a divergence"


def test_a_blocked_tool_shows_in_tool_diffs_not_in_diverged_at(traced_db: Path) -> None:
    import fastaiagent as fa
    from fastaiagent.testing import FunctionModel
    from fastaiagent.trace.replay import Replay

    def responder(messages: list) -> Any:
        # First turn: call the tool. Once a tool result is in the history: answer.
        if not any(getattr(m, "role", None) == "tool" for m in messages):
            return "", [{"name": "lookup", "arguments": {"q": "x"}}]
        return "done"

    def lookup(q: str) -> str:
        """Look something up."""
        return f"found {q}"

    agent = fa.Agent(
        name="tool-agent",
        system_prompt="s",
        llm=FunctionModel(responder),
        tools=[fa.FunctionTool(name="lookup", fn=lookup, register=False)],
    )
    original = agent.run("go")
    forked = Replay.load(original.trace_id).fork_at(0).with_determinism("recorded")
    rerun = forked.rerun()
    assert rerun.tool_dispositions == {"lookup": "blocked"}
    comparison = forked.compare(rerun)

    assert comparison.tool_diffs, "the blocked call's changed result is not reported"
    diff = comparison.tool_diffs[0]
    assert diff.name == "lookup"
    assert diff.disposition == "blocked"
    assert comparison.tools_diverged_at == diff.original_step


def test_compare_reads_the_rerun_from_the_store_it_was_written_to(
    traced_db: Path, tmp_path: Path
) -> None:
    import fastaiagent as fa
    from fastaiagent.testing import TestModel
    from fastaiagent.trace.replay import Replay
    from fastaiagent.trace.storage import TraceStore

    agent = fa.Agent(name="sink-agent", system_prompt="s", llm=TestModel(response="ok"))
    original = agent.run("hi")
    sink = tmp_path / "reruns.db"
    forked = Replay.load(original.trace_id).fork_at(0).with_determinism("recorded")
    rerun = forked.with_trace_db(sink).rerun()

    assert rerun.trace_db_path == str(sink)
    TraceStore(db_path=str(sink)).get_trace(str(rerun.trace_id))  # it is there…
    with pytest.raises(Exception):  # noqa: B017 — …and not in the default store
        TraceStore.default().get_trace(str(rerun.trace_id))
    comparison = forked.compare(rerun)
    assert comparison.compare_status == "ok"
    assert comparison.diverged_at is None
