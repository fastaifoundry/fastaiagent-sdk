"""OpenAI / Anthropic auto-tracers keep a streamed call's span open (1.89.0, D7).

The auto-tracers wrapped only ``create``. With ``stream=True``, ``create``
returns the stream object as soon as the headers arrive, and the span closed
right there: it ended OK with no usage and no content, and an error raised
while the caller iterated the stream (a dropped connection, a provider
``error`` event part-way) happened outside any span. The failure never reached
the trace. ``AsyncOpenAI`` / ``AsyncAnthropic`` were not traced at all.

No mocks: the real ``openai`` and ``anthropic`` SDK clients talk over a real
socket to :mod:`tests._sse_provider`, which streams scripted SSE and can cut
the connection mid-body. Spans are read from the real OTel SDK through an
``InMemorySpanExporter`` (status descriptions and exception events are not in
``local.db``) and from ``local.db``.

Sweep: provider × sync/async × clean stream / error event mid-stream / dropped
connection mid-stream / error event as the first chunk / HTTP error before the
stream / non-stream call.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests._sse_provider import (
    ECHOED_KEY,
    Reply,
    SSEProvider,
    anthropic_error_event,
    anthropic_stream,
    error_body,
    openai_error_event,
    openai_stream,
)

CLIENT_KEY = "sk-local-client-key-0123456789abcdef"


@pytest.fixture(scope="module")
def provider() -> Any:
    server = SSEProvider().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def spans(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fresh tracer provider with an in-memory exporter beside ``local.db``."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()
    exporter = InMemorySpanExporter()
    otel.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield exporter
    finally:
        otel.reset()
        reset_config()


@pytest.fixture
def traced() -> Any:
    from fastaiagent.integrations import anthropic as anthropic_int
    from fastaiagent.integrations import openai as openai_int

    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    openai_int.enable()
    anthropic_int.enable()
    try:
        yield
    finally:
        openai_int.disable()
        anthropic_int.disable()


# -- the two providers behind one interface ----------------------------------------


class _OpenAI:
    name = "openai"
    span_prefix = "openai.chat."

    @staticmethod
    def stream(*texts: str) -> list[Any]:
        return openai_stream(*texts)

    error_event = staticmethod(openai_error_event)

    @staticmethod
    def client(url: str, is_async: bool) -> Any:
        import openai

        cls = openai.AsyncOpenAI if is_async else openai.OpenAI
        return cls(api_key=CLIENT_KEY, base_url=f"{url}/v1", max_retries=0)

    @staticmethod
    def create(client: Any, stream: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return client.chat.completions.create(**kwargs)

    @staticmethod
    def completion_body() -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "whole answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

    clean_usage = (11, 7)


class _Anthropic:
    name = "anthropic"
    span_prefix = "anthropic.messages."

    @staticmethod
    def stream(*texts: str) -> list[Any]:
        return anthropic_stream(*texts)

    error_event = staticmethod(anthropic_error_event)

    @staticmethod
    def client(url: str, is_async: bool) -> Any:
        import anthropic

        cls = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
        return cls(api_key=CLIENT_KEY, base_url=url, max_retries=0)

    @staticmethod
    def create(client: Any, stream: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        if stream:
            kwargs["stream"] = True
        return client.messages.create(**kwargs)

    @staticmethod
    def completion_body() -> dict[str, Any]:
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "whole answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 13, "output_tokens": 9},
        }

    clean_usage = (13, 9)


PROVIDERS = [_OpenAI, _Anthropic]


def _drive(p: Any, url: str, is_async: bool, *, stream: bool = True) -> list[Any]:
    """Make the call and read the whole stream; returns the chunks read."""
    client = p.client(url, is_async)
    if not is_async:
        result = p.create(client, stream)
        return list(result) if stream else [result]

    async def go() -> list[Any]:
        result = await p.create(client, stream)
        if not stream:
            return [result]
        return [c async for c in result]

    return asyncio.run(go())


def _the_span(exporter: Any, p: Any) -> Any:
    found = [s for s in exporter.get_finished_spans() if s.name.startswith(p.span_prefix)]
    assert len(found) == 1, (
        f"expected one ended {p.span_prefix}* span, got {[s.name for s in found]}"
    )
    return found[0]


def _assert_no_key_anywhere(span: Any) -> None:
    blob = json.dumps(
        {
            "status": span.status.description,
            "events": [dict(e.attributes or {}) for e in span.events],
            "attrs": dict(span.attributes or {}),
        },
        default=str,
    )
    assert ECHOED_KEY not in blob, "the echoed key reached the span"
    assert CLIENT_KEY not in blob, "the client's key reached the span"


MODES = [False, True]
MODE_IDS = ["sync", "async"]


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_a_clean_stream_ends_ok_with_usage_and_content(
    p: Any, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=p.stream("Hel", "lo!")))
    chunks = _drive(p, provider.url, is_async)
    assert chunks, "the stream yielded nothing"
    span = _the_span(spans, p)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code != StatusCode.ERROR
    attrs = dict(span.attributes)
    assert (attrs.get("gen_ai.usage.input_tokens"), attrs.get("gen_ai.usage.output_tokens")) == (
        p.clean_usage
    )
    assert attrs.get("gen_ai.response.content") == "Hello!"
    assert attrs.get("gen_ai.response.finish_reason") in ("stop", "end_turn")


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_an_error_event_mid_stream_ends_the_span_error(
    p: Any, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    events = p.stream("Hel", "lo!")
    provider.script(Reply(events=[*events[:3], p.error_event()]))
    with pytest.raises(Exception) as caught:
        _drive(p, provider.url, is_async)
    span = _the_span(spans, p)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
    assert type(caught.value).__name__ in (span.status.description or "")
    exc_events = [e for e in span.events if e.name == "exception"]
    assert exc_events, "the exception was not recorded on the span"
    assert "[REDACTED]" in exc_events[0].attributes["exception.message"]
    _assert_no_key_anywhere(span)
    # What was read before the failure is kept.
    assert dict(span.attributes).get("gen_ai.response.content")


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_a_dropped_connection_mid_stream_ends_the_span_error(
    p: Any, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=p.stream("Hel", "lo!"), cut_after=3))
    with pytest.raises(Exception):
        _drive(p, provider.url, is_async)
    span = _the_span(spans, p)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
    assert [e for e in span.events if e.name == "exception"]


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_an_error_event_as_the_first_chunk_ends_the_span_error(
    p: Any, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=[p.error_event()]))
    with pytest.raises(Exception):
        _drive(p, provider.url, is_async)
    span = _the_span(spans, p)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
    _assert_no_key_anywhere(span)


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_an_http_error_before_the_stream_is_error_and_scrubbed(
    p: Any, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(status=401, body=error_body()))
    with pytest.raises(Exception):
        _drive(p, provider.url, is_async)
    span = _the_span(spans, p)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
    _assert_no_key_anywhere(span)


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_a_non_stream_call_is_still_traced(
    p: Any, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(body=p.completion_body()))
    _drive(p, provider.url, is_async, stream=False)
    attrs = dict(_the_span(spans, p).attributes)
    assert attrs.get("gen_ai.response.content") == "whole answer"
    assert (attrs["gen_ai.usage.input_tokens"], attrs["gen_ai.usage.output_tokens"]) == (
        p.clean_usage
    )


@pytest.mark.parametrize("p", PROVIDERS, ids=lambda p: p.name)
def test_a_stream_closed_early_ends_the_span_without_error(
    p: Any, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=p.stream("a", "b", "c")))
    stream = p.create(p.client(provider.url, False), True)
    with stream:
        next(iter(stream))
    span = _the_span(spans, p)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code != StatusCode.ERROR


def test_the_openai_stream_helper_is_traced_and_errors_surface(
    provider: Any, spans: Any, traced: Any
) -> None:
    """``client.chat.completions.stream(...)`` calls ``create(stream=True)``."""
    p = _OpenAI
    provider.script(Reply(events=[*openai_stream("Hel", "lo!")[:2], openai_error_event()]))
    client = p.client(provider.url, False)
    with pytest.raises(Exception):
        with client.chat.completions.stream(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        ) as s:
            for _ in s:
                pass
    from opentelemetry.trace import StatusCode

    assert _the_span(spans, p).status.status_code == StatusCode.ERROR


def test_the_stream_object_is_the_providers_own(provider: Any, spans: Any, traced: Any) -> None:
    import openai

    provider.script(Reply(events=openai_stream("x")))
    stream = _OpenAI.create(_OpenAI.client(provider.url, False), True)
    assert isinstance(stream, openai.Stream)
    list(stream)


def test_the_stream_span_lands_in_local_db_as_error(
    provider: Any, spans: Any, traced: Any, tmp_path: Any
) -> None:
    from fastaiagent.trace import otel
    from fastaiagent.trace.storage import TraceStore

    provider.script(Reply(events=openai_stream("Hel", "lo!"), cut_after=2))
    with pytest.raises(Exception):
        _drive(_OpenAI, provider.url, False)
    otel.get_tracer_provider().force_flush()
    store = TraceStore(str(tmp_path / "local.db"))
    rows = [
        s
        for t in store.list_traces(limit=5)
        for s in store.get_trace(t.trace_id).spans
        if s.name.startswith("openai.chat.")
    ]
    assert [r.status for r in rows] == ["ERROR"]


def test_disable_restores_sync_and_async_create() -> None:
    anthropic = pytest.importorskip("anthropic")
    openai = pytest.importorskip("openai")
    from fastaiagent.integrations import anthropic as anthropic_int
    from fastaiagent.integrations import openai as openai_int

    targets = [
        (openai.resources.chat.completions.Completions, openai_int),
        (openai.resources.chat.completions.AsyncCompletions, openai_int),
        (anthropic.resources.messages.Messages, anthropic_int),
        (anthropic.resources.messages.AsyncMessages, anthropic_int),
    ]
    before = [cls.create for cls, _ in targets]
    openai_int.enable()
    anthropic_int.enable()
    assert all(cls.create is not b for (cls, _), b in zip(targets, before, strict=True))
    openai_int.disable()
    anthropic_int.disable()
    assert all(cls.create is b for (cls, _), b in zip(targets, before, strict=True))


# -- anthropic's messages.stream(...) helper (MessageStreamManager) --------------------
#
# It posts the request itself rather than through ``create``, so until 1.89.0 it
# was not auto-traced at all.


def _helper_kwargs() -> dict[str, Any]:
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _drive_helper(url: str, is_async: bool, *, stop_after: int | None = None) -> list[Any]:
    """Use ``messages.stream(...)`` as a context manager; read all events, or
    leave the ``with`` block after ``stop_after`` events."""
    client = _Anthropic.client(url, is_async)
    seen: list[Any] = []
    if not is_async:
        with client.messages.stream(**_helper_kwargs()) as stream:
            for event in stream:
                seen.append(event)
                if stop_after is not None and len(seen) >= stop_after:
                    break
        return seen

    async def go() -> list[Any]:
        async with client.messages.stream(**_helper_kwargs()) as stream:
            async for event in stream:
                seen.append(event)
                if stop_after is not None and len(seen) >= stop_after:
                    break
        return seen

    return asyncio.run(go())


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
def test_the_anthropic_stream_helper_clean_stream_has_usage_and_content(
    is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=anthropic_stream("Hel", "lo!")))
    assert _drive_helper(provider.url, is_async)
    span = _the_span(spans, _Anthropic)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code != StatusCode.ERROR
    attrs = dict(span.attributes)
    assert (attrs["gen_ai.usage.input_tokens"], attrs["gen_ai.usage.output_tokens"]) == (13, 9)
    assert attrs["gen_ai.response.content"] == "Hello!"
    assert attrs["gen_ai.response.finish_reason"] == "end_turn"


def test_the_anthropic_stream_helper_final_message_path(
    provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=anthropic_stream("Hel", "lo!")))
    client = _Anthropic.client(provider.url, False)
    with client.messages.stream(**_helper_kwargs()) as stream:
        assert stream.get_final_text() == "Hello!"
    assert dict(_the_span(spans, _Anthropic).attributes)["gen_ai.response.content"] == "Hello!"


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("failure", ["error_event", "dropped", "error_first", "http_401"])
def test_the_anthropic_stream_helper_failures_end_the_span_error(
    failure: str, is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    events = anthropic_stream("Hel", "lo!")
    reply = {
        "error_event": Reply(events=[*events[:3], anthropic_error_event()]),
        "dropped": Reply(events=events, cut_after=3),
        "error_first": Reply(events=[anthropic_error_event()]),
        "http_401": Reply(status=401, body=error_body()),
    }[failure]
    provider.script(reply)
    with pytest.raises(Exception):
        _drive_helper(provider.url, is_async)
    span = _the_span(spans, _Anthropic)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
    assert [e for e in span.events if e.name == "exception"]
    _assert_no_key_anywhere(span)


@pytest.mark.parametrize("is_async", MODES, ids=MODE_IDS)
def test_the_anthropic_stream_helper_early_exit_ends_clean(
    is_async: bool, provider: Any, spans: Any, traced: Any
) -> None:
    provider.script(Reply(events=anthropic_stream("a", "b", "c")))
    assert len(_drive_helper(provider.url, is_async, stop_after=2)) == 2
    span = _the_span(spans, _Anthropic)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code != StatusCode.ERROR
    assert dict(span.attributes)["gen_ai.usage.input_tokens"] == 13


def test_disable_restores_the_anthropic_stream_helper() -> None:
    anthropic = pytest.importorskip("anthropic")
    from fastaiagent.integrations import anthropic as anthropic_int

    classes = [anthropic.resources.messages.Messages, anthropic.resources.messages.AsyncMessages]
    before = [cls.stream for cls in classes]
    anthropic_int.enable()
    assert all(cls.stream is not b for cls, b in zip(classes, before, strict=True))
    anthropic_int.disable()
    assert all(cls.stream is b for cls, b in zip(classes, before, strict=True))
