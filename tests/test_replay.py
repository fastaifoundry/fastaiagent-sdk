"""Tests for fastaiagent.trace.replay module."""

from __future__ import annotations

import json

import pytest

from fastaiagent._internal.errors import ReplayError
from fastaiagent.agent.agent import AgentResult
from fastaiagent.trace.replay import ForkedReplay, Replay
from fastaiagent.trace.storage import SpanData, TraceData


def _make_trace(num_spans: int = 3) -> TraceData:
    spans = [
        SpanData(
            span_id=f"span_{i}",
            trace_id="trace_001",
            name=f"step_{i}",
            start_time=f"2025-01-01T00:00:0{i}Z",
            end_time=f"2025-01-01T00:00:0{i + 1}Z",
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


def _make_agent_trace(num_child_spans: int = 2) -> TraceData:
    """Build a trace that looks like a real agent.run (with Phase A metadata)."""
    root = SpanData(
        span_id="span_root",
        trace_id="trace_002",
        name="agent.test-bot",
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-01T00:00:05Z",
        attributes={
            "agent.name": "test-bot",
            "agent.input": "hello",
            "agent.output": "original output",
            "agent.system_prompt": "You are a test agent.",
            "agent.config": json.dumps({"max_iterations": 5}),
            "agent.tools": json.dumps([]),
            "agent.guardrails": json.dumps([]),
            "agent.llm.provider": "openai",
            "agent.llm.model": "gpt-4o-mini",
            "agent.llm.config": json.dumps(
                {"provider": "openai", "model": "gpt-4o-mini"}
            ),
        },
    )
    children = [
        SpanData(
            span_id=f"span_child_{i}",
            trace_id="trace_002",
            parent_span_id="span_root",
            name=f"openai.chat.gpt-4o-mini",
            start_time=f"2025-01-01T00:00:0{i + 1}Z",
            end_time=f"2025-01-01T00:00:0{i + 2}Z",
            attributes={"gen_ai.system": "openai"},
        )
        for i in range(num_child_spans)
    ]
    spans = [root, *children]
    return TraceData(
        trace_id="trace_002",
        name="agent-trace",
        start_time=root.start_time,
        end_time=children[-1].end_time if children else root.end_time,
        spans=spans,
    )


@pytest.fixture
def stub_agent_arun(monkeypatch):
    """Replace Agent.arun with a coroutine that returns a canned AgentResult
    without touching any real LLM provider."""
    from fastaiagent.agent.agent import Agent

    async def _fake_arun(self, input, **kwargs):
        return AgentResult(
            output=f"stubbed-output for: {input}",
            tokens_used=1,
            latency_ms=1,
            trace_id="new_trace_id",
        )

    monkeypatch.setattr(Agent, "arun", _fake_arun)
    return _fake_arun


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

    def test_rerun(self, stub_agent_arun):
        trace = _make_agent_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1)
        result = forked.rerun()
        assert result.new_output is not None
        assert "stubbed-output" in result.new_output
        assert result.original_output == "original output"
        assert result.trace_id == "new_trace_id"

    def test_rerun_applies_prompt_modification(self, stub_agent_arun, monkeypatch):
        """Verify modify_prompt feeds into Agent reconstruction."""
        captured: dict = {}
        from fastaiagent.agent.agent import Agent

        original_from_dict = Agent.from_dict

        def _capturing_from_dict(data):
            captured["system_prompt"] = data.get("system_prompt")
            return original_from_dict(data)

        monkeypatch.setattr(Agent, "from_dict", _capturing_from_dict)

        trace = _make_agent_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1).modify_prompt("Overridden prompt")
        forked.rerun()
        assert captured["system_prompt"] == "Overridden prompt"

    def test_rerun_raises_when_trace_has_no_spans(self):
        empty_trace = TraceData(
            trace_id="trace_empty",
            name="empty",
            start_time="2025-01-01T00:00:00Z",
            end_time="2025-01-01T00:00:00Z",
            spans=[],
        )
        forked = ForkedReplay(original_trace=empty_trace, fork_point=0, steps=[])
        with pytest.raises(ReplayError, match="no spans"):
            forked.rerun()

    def test_compare(self, stub_agent_arun):
        trace = _make_agent_trace()
        replay = Replay(trace)
        forked = replay.fork_at(step=1)
        result = forked.rerun()
        comparison = forked.compare(result)
        assert comparison.diverged_at == 1
        assert len(comparison.original_steps) == len(trace.spans)
