"""Decisions API spans, egress, and recorded replay (1.84.0).

No mocks: real ``LLMClient.adecide`` against the local ``/v1/decisions`` stand-in,
spans read back from a private ``local.db`` through ``TraceStore``, and replay
through the real ``Replay`` / ``ForkedReplay.arerun``.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from fastaiagent._internal.errors import ReplayError
from fastaiagent.llm import Choice, DecisionResult, LLMClient, Predicate, PredicateAnswer
from fastaiagent.llm.client import _replay_on_miss, _replay_recorded_decisions
from fastaiagent.trace import otel
from fastaiagent.trace.redaction import apply_export_policy
from fastaiagent.trace.replay import _recorded_decision_from_span, _recorded_response_from_span
from tests._decisions_stub import INPUT_TOKENS

DUP = Predicate(name="duplicate", instructions="The customer reports a duplicate charge.")
DEPT = Choice(name="department", instructions="Which department?", options=["billing", "other"])


@pytest.fixture
def fresh_tracing(monkeypatch, tmp_path):
    from fastaiagent._internal.config import reset_config

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    yield tmp_path / "local.db"
    otel.reset()
    reset_config()


def _spans(db_path) -> list:
    from fastaiagent.trace.storage import TraceStore

    otel.get_tracer_provider().force_flush()
    store = TraceStore(str(db_path))
    out = []
    for trace in store.list_traces(limit=50):
        out.extend(store.get_trace(trace.trace_id).spans)
    return out


def _is_decision(s) -> bool:
    return ".decisions." in s.name


def _decision_span(db_path):
    spans = [s for s in _spans(db_path) if _is_decision(s)]
    assert len(spans) == 1, [s.name for s in _spans(db_path)]
    return spans[0]


class TestSpan:
    def test_attributes(self, decisions_stub, fresh_tracing) -> None:
        decisions_stub.client().decide("I was charged twice.", [DUP, DEPT])
        span = _decision_span(fresh_tracing)
        a = span.attributes
        # provider → API → model, next to chat spans like ``llm.openai.gpt-5.1``.
        # The stub is reached as provider="custom" (a gateway).
        assert span.name == "llm.custom.decisions.gpt-6-luna"
        assert a["gen_ai.system"] == "custom"
        assert a["gen_ai.request.model"] == "gpt-6-luna"
        assert a["gen_ai.usage.input_tokens"] == INPUT_TOKENS
        assert a["gen_ai.response.id"] == "req_stub_1"
        assert a["fastaiagent.cost.total_usd"] == pytest.approx(INPUT_TOKENS * 0.10 / 1e6)
        assert json.loads(a["fastaiagent.decision.input"]) == "I was charged twice."
        assert [q["name"] for q in json.loads(a["fastaiagent.decision.questions"])] == [
            "duplicate",
            "department",
        ]
        answers = json.loads(a["fastaiagent.decision.answers"])
        assert answers[0] == {"type": "predicate", "name": "duplicate", "probability": 0.9}
        assert a["fastaiagent.decision.refusals"] == 0

    def test_standard_otel_and_openinference_attributes(self, decisions_stub, fresh_tracing) -> None:
        """Any OTel GenAI or OpenInference backend reads it as an LLM call."""
        decisions_stub.client().decide("x", DUP)
        a = _decision_span(fresh_tracing).attributes
        # OTel GenAI semantic conventions. The operation is named after the API,
        # as ``chat`` is for /chat/completions; the provider is OpenAI's endpoint
        # even through a gateway.
        assert a["gen_ai.operation.name"] == "decisions"
        assert a["gen_ai.provider.name"] == "openai"
        assert a["gen_ai.response.model"] == "gpt-6-luna"
        assert a["gen_ai.usage.output_tokens"] == 0
        # OpenInference (Phoenix / Arize).
        assert a["openinference.span.kind"] == "LLM"
        assert a["llm.provider"] == "openai"
        assert a["llm.model_name"] == "gpt-6-luna"
        assert a["llm.token_count.prompt"] == INPUT_TOKENS
        assert a["llm.token_count.total"] == INPUT_TOKENS

    def test_a_decision_span_is_never_taken_for_a_chat_response(
        self, decisions_stub, fresh_tracing
    ) -> None:
        # Replay rebuilds a chat response from any span with
        # ``gen_ai.response.content``; a decision span must not carry it.
        decisions_stub.client().decide("x", DUP)
        span = _decision_span(fresh_tracing)
        assert "gen_ai.response.content" not in span.attributes
        assert _recorded_response_from_span(span) is None
        assert isinstance(_recorded_decision_from_span(span), DecisionResult)

    def test_refusals_are_counted(self, decisions_stub, fresh_tracing) -> None:
        decisions_stub.refuse.add("duplicate")
        decisions_stub.client().decide("x", [DUP, DEPT])
        assert _decision_span(fresh_tracing).attributes["fastaiagent.decision.refusals"] == 1

    def test_images_are_summarised_never_inlined(self, decisions_stub, fresh_tracing) -> None:
        from fastaiagent.multimodal.image import Image
        from tests.test_decisions import PNG

        decisions_stub.client().decide(["look", Image.from_bytes(PNG, "image/png")], DUP)
        recorded = _decision_span(fresh_tracing).attributes["fastaiagent.decision.input"]
        assert "base64" not in recorded
        assert json.loads(recorded)[1]["type"] == "image"
        # ...while the request itself carried the bytes.
        assert "base64" in json.dumps(decisions_stub.requests[-1]["body"])


class TestEgress:
    """CLAUDE.md §2.5: the evidence is user content; it must not egress when opted out."""

    def test_input_is_stripped_with_payloads_off(
        self, decisions_stub, fresh_tracing, monkeypatch
    ) -> None:
        decisions_stub.client().decide("my card number is 4111", DUP)
        attrs = dict(_decision_span(fresh_tracing).attributes)

        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        out = apply_export_policy(attrs)
        assert "fastaiagent.decision.input" not in out
        # Developer-authored structure and probabilities still flow.
        assert "fastaiagent.decision.questions" in out
        assert "fastaiagent.decision.answers" in out
        assert out["gen_ai.usage.input_tokens"] == INPUT_TOKENS

        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        assert "fastaiagent.decision.input" in apply_export_policy(attrs)

    def test_a_registered_otel_exporter_receives_a_standard_client_span(
        self, decisions_stub, fresh_tracing, monkeypatch
    ) -> None:
        """End to end through ``fastaiagent.trace.otel.add_exporter`` — the path a
        Datadog / Jaeger / Phoenix / Langfuse exporter takes — with payloads off."""
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.trace import SpanKind

        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        exporter = InMemorySpanExporter()
        otel.add_exporter(exporter)
        decisions_stub.client().decide("my card number is 4111", DUP)
        otel.get_tracer_provider().force_flush()

        spans = [s for s in exporter.get_finished_spans() if ".decisions." in s.name]
        assert len(spans) == 1
        span = spans[0]
        assert span.kind == SpanKind.CLIENT
        a = dict(span.attributes)
        assert a["gen_ai.operation.name"] == "decisions"
        assert a["gen_ai.provider.name"] == "openai"
        assert a["gen_ai.request.model"] == "gpt-6-luna"
        assert a["openinference.span.kind"] == "LLM"
        assert "fastaiagent.decision.answers" in a
        # The user's text never left.
        assert "fastaiagent.decision.input" not in a
        assert "4111" not in json.dumps(a, default=str)


class TestRecordedQueue:
    """The queue ``ForkedReplay`` installs, driven through the real ``adecide``."""

    def _run(self, client: LLMClient, queue, on_miss="error", questions=(DUP,)):
        async def go():
            q_token = _replay_recorded_decisions.set(queue)
            m_token = _replay_on_miss.set(on_miss)
            try:
                return await client.adecide("x", list(questions))
            finally:
                _replay_on_miss.reset(m_token)
                _replay_recorded_decisions.reset(q_token)

        return asyncio.run(go())

    def _captured(self, p: float) -> DecisionResult:
        return DecisionResult(answers=[PredicateAnswer(name="duplicate", probability=p)])

    def test_serves_captured_answers_without_a_call(self, decisions_stub) -> None:
        queue = [self._captured(0.11), self._captured(0.22)]
        client = decisions_stub.client()
        assert self._run(client, queue)["duplicate"].probability == 0.11
        assert self._run(client, queue)["duplicate"].probability == 0.22
        assert decisions_stub.requests == []

    def test_a_drained_queue_fails_loud_with_on_miss_error(self, decisions_stub) -> None:
        with pytest.raises(ReplayError, match="ran out of captured decisions"):
            self._run(decisions_stub.client(), [])
        assert decisions_stub.requests == []

    def test_different_questions_are_a_miss_not_a_wrong_answer(self, decisions_stub) -> None:
        with pytest.raises(ReplayError, match="different questions"):
            self._run(decisions_stub.client(), [self._captured(0.5)], questions=(DEPT,))

    def test_on_miss_live_warns_and_calls(self, decisions_stub, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="fastaiagent.llm.client"):
            r = self._run(decisions_stub.client(), [], on_miss="live")
        assert r["duplicate"].probability == 0.9
        assert len(decisions_stub.requests) == 1
        assert any("LIVE Decisions API call" in rec.message for rec in caplog.records)


class TestReplayRoundTrip:
    """Capture a real agent run that decides inside a tool, then rerun it offline."""

    def test_recorded_rerun_needs_no_endpoint(self, decisions_stub, fresh_tracing) -> None:
        from fastaiagent.agent import Agent
        from fastaiagent.tool import FunctionTool
        from fastaiagent.trace.replay import Replay

        decider = decisions_stub.client()

        async def route_ticket(text: str) -> str:
            """Route a support ticket to a department."""
            r = await decider.adecide(text, DEPT)
            return str(r["department"].choice)

        tool = FunctionTool(name="route_ticket_replay_rt", fn=route_ticket)
        decisions_stub.chat_replies = [
            {
                "role": "assistant",
                # A tool-call-only turn: OpenAI sends null content here.
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "route_ticket_replay_rt",
                            "arguments": json.dumps({"text": "charged twice"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Routed to billing."},
        ]
        chat = LLMClient(
            provider="custom",
            model="gpt-4o-mini",
            base_url=decisions_stub.base_url,
            api_key="stub-key",
        )
        agent = Agent(name="router-rt", llm=chat, tools=[tool], system_prompt="Route tickets.")
        result = agent.run("I was charged twice")
        assert "billing" in str(result.output)
        assert len([r for r in decisions_stub.requests if r["path"].endswith("/decisions")]) == 1

        trace_id = result.trace_id
        otel.get_tracer_provider().force_flush()
        decisions_stub.stop()  # nothing may reach the network from here on

        forked = Replay.load(trace_id).fork_at(step=0).with_determinism(
            "recorded", on_miss="error"
        )
        rerun = asyncio.run(forked.arerun())
        assert "billing" in str(rerun.new_output)
        # Not vacuous: the rerun really reached the tool and the decision was
        # served from the capture (a rerun that skipped the tool would also say
        # "billing", from the recorded final turn).
        served = [s for s in Replay.load(rerun.trace_id)._trace.spans if _is_decision(s)]
        assert len(served) == 1
        assert served[0].attributes.get("replay.mode") == "recorded"
