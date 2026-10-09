"""Tests for ``ForkedReplay.with_determinism`` modes + partial tool
overrides + computed ``diverged_at``.

Per project rule [feedback_no_mocking]: no ``unittest.mock``. The
"recorded" determinism path is exercised through the real
``LLMClient.acomplete`` short-circuit — we install a recorded
response in the ContextVar and assert the agent never reaches a
provider HTTP call (verified via a counting subclass).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fastaiagent._internal.errors import ReplayError
from fastaiagent.agent.agent import Agent
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.tool import FunctionTool
from fastaiagent.trace.replay import (
    ForkedReplay,
    Replay,
    _first_divergence,
    _recorded_response_from_span,
)
from fastaiagent.trace.storage import SpanData, TraceData


class _CountingLLMClient(LLMClient):
    """LLMClient subclass that records how many times ``acomplete`` was
    routed to the *real* provider call path. Used to confirm that
    determinism='recorded' skips the HTTP layer entirely.
    """

    def __init__(self) -> None:
        super().__init__(provider="openai", model="gpt-4o-mini", api_key="not-used")
        self.live_calls = 0

    def _get_provider_fn(self):  # type: ignore[override]
        async def _boom(_messages, _tools=None, **_kwargs):
            self.live_calls += 1
            return LLMResponse(
                content="LIVE PROVIDER OUTPUT",
                finish_reason="stop",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        return _boom


def _trace_with_recorded_response(content: str = "captured response") -> TraceData:
    """Build a trace whose root span carries the agent reconstruction
    payload, plus a child llm span carrying ``gen_ai.response.content``.
    """
    root = SpanData(
        span_id="span_root",
        trace_id="trace_det_001",
        name="agent.replayed",
        start_time="2026-05-25T00:00:00Z",
        end_time="2026-05-25T00:00:01Z",
        attributes={
            "agent.name": "replayed-bot",
            "agent.input": "original question",
            "agent.output": "original answer",
            "agent.system_prompt": "be helpful",
            "agent.config": json.dumps({"max_iterations": 3}),
            "agent.tools": json.dumps([]),
            "agent.guardrails": json.dumps([]),
            "agent.llm.provider": "openai",
            "agent.llm.model": "gpt-4o-mini",
            "agent.llm.config": json.dumps(
                {"provider": "openai", "model": "gpt-4o-mini", "api_key": "not-used"}
            ),
        },
    )
    llm_span = SpanData(
        span_id="span_llm_0",
        trace_id="trace_det_001",
        parent_span_id="span_root",
        name="llm.openai.gpt-4o-mini",
        start_time="2026-05-25T00:00:00Z",
        end_time="2026-05-25T00:00:01Z",
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.response.content": content,
            "gen_ai.response.finish_reason": "stop",
        },
    )
    return TraceData(
        trace_id="trace_det_001",
        name="det-trace",
        start_time=root.start_time,
        end_time=llm_span.end_time,
        spans=[root, llm_span],
    )


class TestRecordedResponseHelper:
    def test_reconstructs_response_from_span(self):
        trace = _trace_with_recorded_response("hello world")
        llm_span = trace.spans[1]
        rec = _recorded_response_from_span(llm_span)
        assert rec is not None
        assert rec.content == "hello world"
        assert rec.finish_reason == "stop"

    def test_returns_none_when_no_response_content(self):
        empty_span = SpanData(
            span_id="x",
            trace_id="t",
            name="llm.openai",
            start_time="",
            end_time="",
            attributes={"gen_ai.system": "openai"},
        )
        assert _recorded_response_from_span(empty_span) is None

    def test_a_tool_call_only_turn_is_replayed(self):
        """1.84.0: OpenAI sends ``content: null`` on a turn that only calls tools.

        Such a span has no ``gen_ai.response.content``; before 1.84.0 it was
        skipped, so a recorded rerun jumped to the final answer without running
        any tool — and still matched the original output.
        """
        span = SpanData(
            span_id="x",
            trace_id="t",
            name="llm.openai.gpt-4o-mini",
            start_time="",
            end_time="",
            attributes={
                "gen_ai.response.tool_calls": json.dumps(
                    [{"id": "call_1", "name": "lookup", "arguments": {"q": "a"}}]
                ),
                "gen_ai.response.finish_reason": "tool_calls",
            },
        )
        rec = _recorded_response_from_span(span)
        assert rec is not None
        assert rec.content is None
        assert [(tc.id, tc.name, tc.arguments) for tc in rec.tool_calls] == [
            ("call_1", "lookup", {"q": "a"})
        ]
        assert rec.finish_reason == "tool_calls"

    def test_unreadable_tool_calls_with_no_content_is_not_a_turn(self):
        span = SpanData(
            span_id="x",
            trace_id="t",
            name="llm.openai",
            start_time="",
            end_time="",
            attributes={"gen_ai.response.tool_calls": "{not json"},
        )
        assert _recorded_response_from_span(span) is None


class TestDeterminismValidation:
    def test_unknown_mode_raises(self):
        forked = ForkedReplay(
            original_trace=_trace_with_recorded_response(),
            fork_point=0,
            steps=[],
        )
        with pytest.raises(ReplayError, match="Unknown determinism mode"):
            forked.with_determinism("wibble")  # type: ignore[arg-type]


class TestRecordedMode:
    """``determinism="recorded"`` skips the LLM HTTP call and returns the
    captured response. We confirm by counting calls against a
    ``_CountingLLMClient`` that fails the test if the live path runs.
    """

    @pytest.mark.asyncio
    async def test_recorded_mode_skips_provider_call(self, monkeypatch):
        counting = _CountingLLMClient()

        # Replace Agent.from_dict so reconstruction returns an agent that
        # already has the counting LLM client wired in.
        original_from_dict = Agent.from_dict

        def _from_dict_with_counting(data):
            agent = original_from_dict(data)
            agent.llm = counting
            return agent

        monkeypatch.setattr(Agent, "from_dict", _from_dict_with_counting)

        trace = _trace_with_recorded_response("captured!")
        replay = Replay(trace)
        forked = replay.fork_at(step=0).with_determinism("recorded")
        result = await forked.arerun()

        assert counting.live_calls == 0, "recorded mode must not hit the provider"
        # The agent loop sees the recorded LLMResponse; the recorded
        # content flows through to ``new_output`` (Agent stops when the
        # response has finish_reason='stop' and no tool calls).
        assert "captured!" in str(result.new_output)

    @pytest.mark.asyncio
    async def test_recorded_mode_raises_when_no_capture(self, monkeypatch):
        # Trace without a gen_ai.response.content attribute.
        root = SpanData(
            span_id="r",
            trace_id="t",
            name="agent.no_capture",
            start_time="",
            end_time="",
            attributes={
                "agent.name": "x",
                "agent.input": "y",
                "agent.llm.config": json.dumps(
                    {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k"}
                ),
                "agent.tools": "[]",
                "agent.guardrails": "[]",
                "agent.config": "{}",
                "agent.system_prompt": "",
            },
        )
        trace = TraceData(
            trace_id="t",
            name="x",
            start_time="",
            end_time="",
            spans=[root],
        )
        forked = Replay(trace).fork_at(step=0).with_determinism("recorded")
        with pytest.raises(ReplayError, match="no captured LLM response"):
            await forked.arerun()


class TestDeterministicMode:
    @pytest.mark.asyncio
    async def test_deterministic_mode_forces_temperature_and_seed(self, monkeypatch):
        # Capture the LLM client that the rerun ends up with — confirm
        # temperature=0 and seed=42 were forced.
        captured: dict[str, Any] = {}

        original_from_dict = Agent.from_dict

        def _from_dict_capturing(data):
            agent = original_from_dict(data)
            captured["agent"] = agent
            return agent

        monkeypatch.setattr(Agent, "from_dict", _from_dict_capturing)

        async def _fake_arun(self, input, **kwargs):
            from fastaiagent.agent.agent import AgentResult

            captured["temperature"] = self.llm.temperature
            captured["seed"] = self.llm.seed
            return AgentResult(output="ok", tokens_used=0, latency_ms=0, trace_id="new")

        monkeypatch.setattr(Agent, "arun", _fake_arun)

        trace = _trace_with_recorded_response()
        forked = Replay(trace).fork_at(step=0).with_determinism("deterministic")
        await forked.arerun()
        assert captured["temperature"] == 0
        assert captured["seed"] == 42


class TestPartialToolOverride:
    def test_with_tool_override_stores_by_name(self):
        forked = ForkedReplay(
            original_trace=_trace_with_recorded_response(),
            fork_point=0,
            steps=[],
        )

        def _impl(q: str) -> str:
            return q.upper()

        live = FunctionTool(name="search", fn=_impl)
        forked.with_tool_override("search", live)
        assert "search" in forked._tool_overrides
        assert forked._tool_overrides["search"] is live

    def test_with_tool_override_requires_non_empty_name(self):
        forked = ForkedReplay(
            original_trace=_trace_with_recorded_response(),
            fork_point=0,
            steps=[],
        )
        with pytest.raises(ReplayError, match="non-empty"):
            forked.with_tool_override("", object())

    def test_apply_tool_overrides_substitutes_matching_name(self):
        forked = ForkedReplay(
            original_trace=_trace_with_recorded_response(),
            fork_point=0,
            steps=[],
        )

        def _orig(q: str) -> str:
            return "orig"

        def _patched(q: str) -> str:
            return "patched"

        orig_search = FunctionTool(name="search", fn=_orig)
        orig_other = FunctionTool(name="other", fn=_orig)
        patched_search = FunctionTool(name="search", fn=_patched)

        forked.with_tool_override("search", patched_search)
        out = forked._apply_tool_overrides([orig_search, orig_other])
        assert out[0] is patched_search
        assert out[1] is orig_other  # untouched

    def test_apply_tool_overrides_appends_new_tools(self):
        forked = ForkedReplay(
            original_trace=_trace_with_recorded_response(),
            fork_point=0,
            steps=[],
        )

        def _impl(q: str) -> str:
            return q

        newcomer = FunctionTool(name="newcomer", fn=_impl)
        forked.with_tool_override("newcomer", newcomer)
        # No tool with name "newcomer" exists in the reconstructed list.
        out = forked._apply_tool_overrides([])
        assert out == [newcomer]


class TestFirstDivergence:
    def _step(self, idx: int, name: str, output: str = "") -> Any:
        from fastaiagent.trace.replay import ReplayStep

        return ReplayStep(
            step=idx,
            span_name=name,
            span_id=f"s{idx}",
            attributes={"agent.output": output} if output else {},
            timestamp="",
        )

    def test_no_divergence_returns_none(self):
        a = [self._step(0, "a", "x"), self._step(1, "b", "y")]
        b = [self._step(0, "a", "x"), self._step(1, "b", "y")]
        assert _first_divergence(a, b) is None

    def test_diverges_on_first_different_output(self):
        a = [self._step(0, "a", "x"), self._step(1, "b", "y")]
        b = [self._step(0, "a", "x"), self._step(1, "b", "Z")]
        assert _first_divergence(a, b) == 1

    def test_diverges_on_different_span_name(self):
        a = [self._step(0, "a", "x"), self._step(1, "b", "y")]
        b = [self._step(0, "a", "x"), self._step(1, "DIFFERENT", "y")]
        assert _first_divergence(a, b) == 1

    def test_diverges_when_one_is_longer(self):
        a = [self._step(0, "a"), self._step(1, "b")]
        b = [self._step(0, "a")]
        # The shorter list runs out at index 1.
        assert _first_divergence(a, b) == 1

    def test_returns_none_for_empty_lists(self):
        assert _first_divergence([], []) is None
        assert _first_divergence([self._step(0, "a")], []) is None
