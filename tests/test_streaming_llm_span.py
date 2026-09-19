"""A streamed call must emit the same ``llm.*`` span a non-streamed one does.

Found by the 1.68.0 cross-repo sweep, and only visible from the plane side:
``LLMClient._stream_*`` created no span at all, so a streamed run had no
``gen_ai.usage.*`` anywhere in its trace. The control plane derives a run's
token total from LLM spans alone, so it reported **0** for the whole run.

Before 1.68.0 the SDK also reported 0, so the two agreed *by accident*. Making
``tokens_used`` correct made the divergence visible rather than causing it —
which is why this is a pre-existing gap rather than a regression.

The tests drive a real ``LLMClient`` over httpx against an in-process stub, for
the same reason ``test_agent_cost.py`` does: the span is emitted *inside* the
client, so a fake client that replaced ``astream`` would test nothing.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fastaiagent.llm import LLMClient
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.stream import TextDelta, Usage
from fastaiagent.trace import otel

PROMPT_TOKENS = 11
COMPLETION_TOKENS = 7


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class _StreamHandler(BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible streaming endpoint."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in ("Hel", "lo!"):
            self.wfile.write(
                _sse(
                    {
                        "choices": [{"delta": {"content": chunk}, "finish_reason": None}],
                    }
                )
            )
        self.wfile.write(
            _sse(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": PROMPT_TOKENS,
                        "completion_tokens": COMPLETION_TOKENS,
                        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                    },
                }
            )
        )
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None:  # silence the test log
        return


@pytest.fixture
def stub_stream():
    server = HTTPServer(("127.0.0.1", 0), _StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fresh_tracing(monkeypatch, tmp_path):
    """A private local.db and a tracer provider pointed at it."""
    from fastaiagent._internal.config import reset_config

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    yield tmp_path / "local.db"
    otel.reset()
    reset_config()


def _spans(db_path) -> list[dict]:
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore(str(db_path))
    out = []
    for trace in store.list_traces(limit=20):
        out.extend(store.get_trace(trace.trace_id).spans)
    return out


def _client(base_url: str) -> LLMClient:
    return LLMClient(
        provider="custom", model="gpt-4o-mini", base_url=base_url, api_key="stub", max_retries=0
    )


def test_a_streamed_call_emits_an_llm_span(stub_stream, fresh_tracing) -> None:
    """Red before the fix: a streamed call produced no ``llm.*`` span at all."""

    async def run() -> None:
        async for _ in _client(stub_stream).astream([UserMessage(content="hi")]):
            pass

    asyncio.run(run())
    otel.get_tracer_provider().force_flush()

    names = [s.name for s in _spans(fresh_tracing)]
    assert any(n.startswith("llm.") for n in names), f"no llm.* span emitted: {names}"


def test_the_streamed_span_carries_usage(stub_stream, fresh_tracing) -> None:
    """The attributes the plane actually reads.

    This is the whole point: the plane derives a run's token total from
    ``gen_ai.usage.*`` on LLM spans, so a span without them buys nothing.
    """

    async def run() -> None:
        async for _ in _client(stub_stream).astream([UserMessage(content="hi")]):
            pass

    asyncio.run(run())
    otel.get_tracer_provider().force_flush()

    llm = [s for s in _spans(fresh_tracing) if s.name.startswith("llm.")]
    assert llm, "no llm.* span"
    attrs = llm[0].attributes or {}
    assert attrs.get("gen_ai.usage.input_tokens") == PROMPT_TOKENS, attrs
    assert attrs.get("gen_ai.usage.output_tokens") == COMPLETION_TOKENS, attrs


def test_the_streamed_span_carries_cost(stub_stream, fresh_tracing) -> None:
    """A priced model gets the same cost attribute the non-streamed path sets."""

    async def run() -> None:
        async for _ in _client(stub_stream).astream([UserMessage(content="hi")]):
            pass

    asyncio.run(run())
    otel.get_tracer_provider().force_flush()

    llm = [s for s in _spans(fresh_tracing) if s.name.startswith("llm.")]
    attrs = llm[0].attributes or {}
    assert "fastaiagent.cost.total_usd" in attrs, attrs
    assert attrs["fastaiagent.cost.total_usd"] > 0


def test_the_streamed_span_carries_the_assembled_reply(stub_stream, fresh_tracing) -> None:
    """The deltas are reassembled, so the span shows what the model actually said."""

    async def run() -> None:
        async for _ in _client(stub_stream).astream([UserMessage(content="hi")]):
            pass

    asyncio.run(run())
    otel.get_tracer_provider().force_flush()

    llm = [s for s in _spans(fresh_tracing) if s.name.startswith("llm.")]
    attrs = llm[0].attributes or {}
    assert attrs.get("gen_ai.response.content") == "Hello!", attrs


def test_the_span_ends_when_the_consumer_abandons_the_stream(
    stub_stream, fresh_tracing
) -> None:
    """A consumer that breaks out early must not leak an unended span.

    An unended span never exports, so it is worse than no span: the run looks
    partially traced and nothing says why.
    """

    async def run() -> None:
        agen = _client(stub_stream).astream([UserMessage(content="hi")])
        async for event in agen:
            if isinstance(event, TextDelta):
                break  # walk away mid-stream
        await agen.aclose()

    asyncio.run(run())
    otel.get_tracer_provider().force_flush()

    llm = [s for s in _spans(fresh_tracing) if s.name.startswith("llm.")]
    assert llm, "the abandoned stream left no span — it was never ended"


def test_streaming_and_non_streaming_spans_have_the_same_name(
    stub_stream, fresh_tracing
) -> None:
    """Parity is the point: a console should not be able to tell them apart by
    shape, only by content."""

    async def run() -> None:
        async for _ in _client(stub_stream).astream([UserMessage(content="hi")]):
            pass

    asyncio.run(run())
    otel.get_tracer_provider().force_flush()

    llm = [s for s in _spans(fresh_tracing) if s.name.startswith("llm.")]
    assert llm[0].name == "llm.custom.gpt-4o-mini"


def test_usage_event_is_still_yielded_to_the_consumer(stub_stream, fresh_tracing) -> None:
    """Observing the stream must not consume it — the events a caller relied on
    before this change still arrive."""

    seen: list[object] = []

    async def run() -> None:
        async for event in _client(stub_stream).astream([UserMessage(content="hi")]):
            seen.append(event)

    asyncio.run(run())

    assert any(isinstance(e, Usage) for e in seen), [type(e).__name__ for e in seen]
    text = "".join(e.text for e in seen if isinstance(e, TextDelta))
    assert text == "Hello!"
