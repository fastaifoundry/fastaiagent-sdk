"""``with_determinism("recorded", on_miss=…)`` — the drained-queue contract (1.48.0).

Before 1.48.0 a recorded rerun that made MORE LLM calls than the original
trace captured silently fell through to a live, billed provider call. Now:

* ``on_miss="live"`` (default — unchanged behavior) falls through but logs
  a prominent warning;
* ``on_miss="error"`` raises :class:`ReplayError` before any provider call.

Per project rule [feedback_no_mocking]: no ``unittest.mock``. The drained
queue is exercised through the real ``LLMClient.acomplete`` short-circuit,
with a counting subclass proving whether the provider path ran (the same
pattern as ``tests/test_replay_determinism.py``).
"""

from __future__ import annotations

import logging

import pytest

from fastaiagent._internal.errors import ReplayError
from fastaiagent.llm.client import (
    LLMClient,
    LLMResponse,
    _replay_on_miss,
    _replay_recorded_response,
)
from fastaiagent.llm.message import Message
from fastaiagent.trace.replay import ForkedReplay
from fastaiagent.trace.storage import TraceData


class _CountingLLMClient(LLMClient):
    """Counts trips into the real provider-call path."""

    def __init__(self) -> None:
        super().__init__(provider="openai", model="gpt-4o-mini", api_key="not-used")
        self.live_calls = 0

    def _get_provider_fn(self):  # type: ignore[override]
        async def _live(_messages, _tools=None, **_kwargs):
            self.live_calls += 1
            return LLMResponse(
                content="LIVE PROVIDER OUTPUT",
                finish_reason="stop",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        return _live


def _empty_trace() -> TraceData:
    return TraceData(trace_id="t", name="t", start_time="", end_time="", spans=[])


class TestOnMissValidation:
    def test_invalid_on_miss_raises(self) -> None:
        forked = ForkedReplay(original_trace=_empty_trace(), fork_point=0, steps=[])
        with pytest.raises(ReplayError, match="Unknown on_miss"):
            forked.with_determinism("recorded", on_miss="bogus")

    def test_valid_values_accepted(self) -> None:
        forked = ForkedReplay(original_trace=_empty_trace(), fork_point=0, steps=[])
        forked.with_determinism("recorded", on_miss="error")
        assert forked._on_miss == "error"
        forked.with_determinism("recorded", on_miss="live")
        assert forked._on_miss == "live"


class TestDrainedQueue:
    """An installed-but-empty queue means: replay active, responses exhausted."""

    @pytest.mark.asyncio
    async def test_on_miss_error_raises_before_any_provider_call(self) -> None:
        client = _CountingLLMClient()
        q_token = _replay_recorded_response.set([])
        m_token = _replay_on_miss.set("error")
        try:
            with pytest.raises(ReplayError, match="ran out of captured LLM responses"):
                await client.acomplete([Message(role="user", content="hi")])
        finally:
            _replay_on_miss.reset(m_token)
            _replay_recorded_response.reset(q_token)
        assert client.live_calls == 0

    @pytest.mark.asyncio
    async def test_default_live_falls_through_with_warning(self, caplog) -> None:
        client = _CountingLLMClient()
        q_token = _replay_recorded_response.set([])
        try:
            with caplog.at_level(logging.WARNING, logger="fastaiagent.llm.client"):
                response = await client.acomplete([Message(role="user", content="hi")])
        finally:
            _replay_recorded_response.reset(q_token)
        assert client.live_calls == 1
        assert response.content == "LIVE PROVIDER OUTPUT"
        assert any("ran out of captured LLM responses" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_queue_with_items_still_replays(self) -> None:
        client = _CountingLLMClient()
        recorded = LLMResponse(content="captured!", finish_reason="stop", usage={})
        q_token = _replay_recorded_response.set([recorded])
        m_token = _replay_on_miss.set("error")
        try:
            response = await client.acomplete([Message(role="user", content="hi")])
        finally:
            _replay_on_miss.reset(m_token)
            _replay_recorded_response.reset(q_token)
        assert response.content == "captured!"
        assert client.live_calls == 0
