"""Tests for fastaiagent.trace.replay module."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import ReplayError
from fastaiagent.trace.replay import ForkedReplay, Replay, ReplayStep
from fastaiagent.trace.storage import SpanData, TraceData


def _make_trace(num_spans: int = 3) -> TraceData:
    spans = [
        SpanData(
            span_id=f"span_{i}",
            trace_id="trace_001",
            name=f"step_{i}",
            start_time=f"2025-01-01T00:00:0{i}Z",
            end_time=f"2025-01-01T00:00:0{i+1}Z",
            attributes={"step": i},
        )
        for i in range(num_spans)
    ]
    return TraceData(
        trace_id="trace_001",
        name="test-trace",
        start_time=spans[0].start_time,
        end_time=spans[-1].end_time,
        spans=spans,
    )


class TestReplay:
    def test_construction(self):
        trace = _make_trace()
        replay = Replay(trace)
        assert len(replay.steps()) == 3

    def test_summary(self):
        trace = _make_trace()
        replay = Replay(trace)
        summary = replay.summary()
        assert "trace_001" in summary
        assert "test-trace" in summary
        assert "step_0" in summary

    def test_steps(self):
        trace = _make_trace(5)
        replay = Replay(trace)
        steps = replay.steps()
        assert len(steps) == 5
        assert steps[0].step == 0
        assert steps[4].step == 4

    def test_inspect_valid_step(self):
        trace = _make_trace()
        replay = Replay(trace)
        step = replay.inspect(1)
        assert step.span_name == "step_1"
        assert step.step == 1

    def test_inspect_invalid_step(self):
        trace = _make_trace()
        replay = Replay(trace)
        with pytest.raises(ReplayError, match="out of range"):
            replay.inspect(10)

    def test_step_through(self):
        trace = _make_trace()
        replay = Replay(trace)
        steps = replay.step_through()
        assert len(steps) == 3

    def test_fork_at(self):
        trace = _make_trace(5)
        replay = Replay(trace)
        forked = replay.fork_at(step=2)
        assert isinstance(forked, ForkedReplay)

    def test_fork_at_invalid_step(self):
        trace = _make_trace()
        replay = Replay(trace)
        with pytest.raises(ReplayError, match="out of range"):
            replay.fork_at(step=99)


class TestForkedReplay:
    def test_modify_input(self):
        trace = _make_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1)
        result = forked.modify_input({"new": "input"})
        assert result is forked  # returns self for chaining

    def test_modify_prompt(self):
        trace = _make_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1)
        forked.modify_prompt("New prompt")
        assert forked._modifications["prompt"] == "New prompt"

    def test_modify_state(self):
        trace = _make_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=0)
        forked.modify_state({"key": "value"})
        assert forked._modifications["state"] == {"key": "value"}

    def test_rerun(self):
        trace = _make_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1)
        result = forked.rerun()
        assert result.trace_id == "trace_001"

    def test_compare(self):
        trace = _make_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1)
        result = forked.rerun()
        comparison = forked.compare(result)
        assert comparison.diverged_at == 1
