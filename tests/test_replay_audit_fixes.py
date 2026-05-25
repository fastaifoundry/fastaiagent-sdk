"""Tests for the v1.14.1 audit-driven fixes to replay.

Covers:

#1. ``ReplayStep.input`` / ``ReplayStep.output`` populated from span
    attributes (was empty in v1.14.0 — UI diff cards were blank).
#4. ``save_as_test`` writes provenance fields
    (``source_trace_id`` / ``fixed_trace_id`` / ``fork_step`` /
    ``modifications``) in addition to the v1.13/v1.14.0 fields, and
    ``trace_id`` now means "the rerun's id" consistently.
#5. ``determinism="recorded"`` replays the full ordered sequence of
    captured LLM responses, not just the first one (multi-turn fix).

All tests use real classes and a deterministic ``LLMClient`` subclass
(not ``unittest.mock``) per the project's no-mocking rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent.llm.client import LLMClient, LLMResponse, _replay_recorded_response
from fastaiagent.trace.replay import (
    ForkedReplay,
    Replay,
    ReplayResult,
    ReplayStep,
    _partition_span_io,
)
from fastaiagent.trace.storage import SpanData, TraceData


def _trace_with_payloads(
    *,
    response_contents: list[str] | None = None,
) -> TraceData:
    """Construct a trace with one agent root span and N llm.* child
    spans, each carrying a different ``gen_ai.response.content`` so the
    multi-turn replay test can verify pop-from-front ordering."""
    responses = response_contents or ["first response"]
    root = SpanData(
        span_id="root",
        trace_id="t-1",
        name="agent.bot",
        start_time="2026-05-25T00:00:00Z",
        end_time="2026-05-25T00:00:09Z",
        attributes={
            "agent.name": "bot",
            "agent.input": "Look up ORD-1",
            "agent.output": "Final reply",
            "agent.system_prompt": "be helpful",
            "agent.config": json.dumps({"max_iterations": 5}),
            "agent.tools": json.dumps([]),
            "agent.guardrails": json.dumps([]),
            "agent.llm.provider": "openai",
            "agent.llm.model": "gpt-4o-mini",
            "agent.llm.config": json.dumps(
                {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k"}
            ),
        },
    )
    llm_spans = [
        SpanData(
            span_id=f"llm-{i}",
            trace_id="t-1",
            parent_span_id="root",
            name="llm.openai.gpt-4o-mini",
            # Use the index in the timestamp so sort-by-start_time is
            # stable and matches authoring order.
            start_time=f"2026-05-25T00:00:0{i + 1}Z",
            end_time=f"2026-05-25T00:00:0{i + 2}Z",
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.response.content": content,
                "gen_ai.response.finish_reason": "stop",
            },
        )
        for i, content in enumerate(responses)
    ]
    return TraceData(
        trace_id="t-1",
        name="multi-turn-trace",
        start_time=root.start_time,
        end_time=llm_spans[-1].end_time if llm_spans else root.end_time,
        spans=[root, *llm_spans],
    )


# ── #1: ReplayStep.input / output populated ────────────────────────────────


class TestStepInputOutputPopulated:
    def test_input_keys_land_in_step_input(self):
        attrs = {
            "agent.input": "hello",
            "gen_ai.request.messages": '[{"role": "user", "content": "hi"}]',
            "agent.name": "bot",  # not an input/output key
        }
        input_part, output_part = _partition_span_io(attrs)
        assert "agent.input" in input_part
        assert input_part["agent.input"] == "hello"
        # JSON-string payloads are decoded so the UI sees structured data.
        assert input_part["gen_ai.request.messages"] == [{"role": "user", "content": "hi"}]
        # Non-input/output keys stay out of both partitions.
        assert "agent.name" not in input_part
        assert "agent.name" not in output_part

    def test_output_keys_land_in_step_output(self):
        attrs = {
            "agent.output": "reply",
            "gen_ai.response.content": "model said this",
            "tool.output": '{"result": 42}',
        }
        input_part, output_part = _partition_span_io(attrs)
        assert output_part["agent.output"] == "reply"
        assert output_part["gen_ai.response.content"] == "model said this"
        assert output_part["tool.output"] == {"result": 42}
        assert not input_part

    def test_replay_build_steps_fills_input_output(self):
        # End-to-end: real ``Replay`` over a real ``TraceData`` —
        # confirm every step has populated input/output, not the
        # always-empty dict the UI saw pre-v1.14.1.
        trace = _trace_with_payloads()
        replay = Replay(trace)
        steps = replay.steps()
        # Root agent span — its agent.input/agent.output should be partitioned.
        assert steps[0].input == {"agent.input": "Look up ORD-1"}
        assert steps[0].output == {"agent.output": "Final reply"}
        # LLM child span — gen_ai.response.content lands in output.
        assert steps[1].output == {
            "gen_ai.response.content": "first response",
        }


# ── #4: save_as_test provenance ────────────────────────────────────────────


class TestSaveAsTestProvenance:
    def test_writes_all_provenance_fields(self, tmp_path: Path):
        rerun = ReplayResult(
            original_output="bad",
            new_output="good",
            steps_executed=1,
            trace_id="fixed-trace-9",
        )
        dataset = tmp_path / "regression.jsonl"
        rerun.save_as_test(
            dataset,
            input="What is X?",
            expected_output="good",
            source_trace_id="failure-trace-1",
            fork_step=3,
            modifications={"prompt": "Be specific.", "tool_overrides": ["search"]},
        )
        row = json.loads(dataset.read_text().strip())
        assert row["input"] == "What is X?"
        assert row["expected_output"] == "good"
        # v1.14.1: trace_id now ALWAYS means the rerun's id (no longer
        # overwritten when source_trace_id is passed). The failure id
        # lives in its own field.
        assert row["trace_id"] == "fixed-trace-9"
        assert row["fixed_trace_id"] == "fixed-trace-9"
        assert row["source_trace_id"] == "failure-trace-1"
        assert row["fork_step"] == 3
        assert row["modifications"] == {
            "prompt": "Be specific.",
            "tool_overrides": ["search"],
        }
        assert "created_at" in row

    def test_omitted_provenance_defaults_to_none_or_empty(self, tmp_path: Path):
        rerun = ReplayResult(new_output="x", trace_id="r")
        dataset = tmp_path / "r.jsonl"
        rerun.save_as_test(dataset, input="i", expected_output="x")
        row = json.loads(dataset.read_text().strip())
        # Caller didn't pass source_trace_id / fork_step / modifications —
        # they're still present but null / empty so downstream readers
        # can rely on the schema.
        assert row["source_trace_id"] is None
        assert row["fork_step"] is None
        assert row["modifications"] == {}


# ── #5: multi-turn recorded queue ──────────────────────────────────────────


class _CountingLLMClient(LLMClient):
    """Real ``LLMClient`` subclass that records how many times its
    provider call path actually ran. The recorded-mode short-circuit in
    ``acomplete`` should keep this at zero for every entry the queue
    can satisfy, and only fall through for extra calls beyond the
    queue's length.
    """

    def __init__(self) -> None:
        super().__init__(provider="openai", model="gpt-4o-mini", api_key="unused")
        self.live_calls = 0

    def _get_provider_fn(self):  # type: ignore[override]
        async def _boom(_messages, _tools=None, **_kwargs):
            self.live_calls += 1
            return LLMResponse(content="LIVE", finish_reason="stop")

        return _boom


@pytest.fixture(autouse=True)
def _reset_recorded_queue():
    # Some upstream test could leave a queue set; isolate.
    _replay_recorded_response.set(None)
    yield
    _replay_recorded_response.set(None)


class TestRecordedQueueMultiTurn:
    def test_all_llm_responses_returns_ordered_list(self):
        trace = _trace_with_payloads(response_contents=["turn-1", "turn-2", "turn-3"])
        forked = ForkedReplay(original_trace=trace, fork_point=0, steps=[])
        queue = forked._all_llm_responses()
        assert [r.content for r in queue] == ["turn-1", "turn-2", "turn-3"]

    def test_first_llm_response_still_works_as_alias(self):
        # Deprecated alias retained so anyone who monkey-patched the
        # v1.14.0 method in tests keeps working.
        trace = _trace_with_payloads(response_contents=["turn-1", "turn-2"])
        forked = ForkedReplay(original_trace=trace, fork_point=0, steps=[])
        first = forked._first_llm_response()
        assert first is not None
        assert first.content == "turn-1"

    @pytest.mark.asyncio
    async def test_pop_from_front_in_order(self):
        # Install a 3-entry queue, then call acomplete 3 times — the
        # responses must come back in order, never repeating.
        responses = [
            LLMResponse(content="alpha", finish_reason="stop"),
            LLMResponse(content="beta", finish_reason="stop"),
            LLMResponse(content="gamma", finish_reason="stop"),
        ]
        counting = _CountingLLMClient()
        token = _replay_recorded_response.set(list(responses))
        try:
            seen = []
            for _ in range(3):
                r = await counting.acomplete([])
                seen.append(r.content)
            assert seen == ["alpha", "beta", "gamma"]
            assert counting.live_calls == 0  # never hit the provider
        finally:
            _replay_recorded_response.reset(token)

    @pytest.mark.asyncio
    async def test_queue_drained_falls_through_to_live(self):
        # If the rerun makes more LLM calls than the original trace
        # captured (e.g. a tool override that triggers an extra turn),
        # the agent shouldn't deadlock — fall through to a live call.
        counting = _CountingLLMClient()
        token = _replay_recorded_response.set(
            [
                LLMResponse(content="only-recorded", finish_reason="stop"),
            ]
        )
        try:
            r1 = await counting.acomplete([])  # served from queue
            r2 = await counting.acomplete([])  # queue empty → live path
            assert r1.content == "only-recorded"
            assert r2.content == "LIVE"
            assert counting.live_calls == 1
        finally:
            _replay_recorded_response.reset(token)


# ── Sanity: ReplayStep API surface unchanged ───────────────────────────────


def test_replaystep_fields_are_still_optional_with_defaults():
    """Construction without ``input`` / ``output`` still works — the
    UI's hand-built step fixtures (and any third-party code that built
    ReplayStep directly in v1.14.0) shouldn't break."""
    step = ReplayStep(step=0, span_name="agent.x", span_id="s")
    assert step.input == {}
    assert step.output == {}
