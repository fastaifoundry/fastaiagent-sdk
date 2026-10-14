"""A provider error that echoes the API key leaks it nowhere — P9 (1.89.0).

OpenAI answers a bad key with ``"Incorrect API key provided: sk-…"``, and every
``LLMProviderError`` used to embed the provider's body verbatim. From there OTel
copied it into the span's exception event and status description, local.db
stored it, the UI showed it, the Playground logged it with ``exc_info=True``,
and the exporter sent it to the plane (which copies ``exception.message`` into
``trace.error``) whenever payloads were on. Gemini also put the key in the URL,
so httpx's own INFO log line carried it.

The fix scrubs where the exception is built (``LLMClient._provider_error``), and
falls back to scrubbing exception text when spans are stored and exported, for
exceptions the SDK did not build. No mocks: a real localhost server plays the
provider (``CaptureServer`` with scripted bodies), and the real ``httpx``
transport, retry loop, tracing and storage run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any

import pytest

from fastaiagent._internal.errors import LLMProviderError
from fastaiagent.llm import LLMClient, UserMessage

#: Shaped like a gateway token, not an OpenAI key, so only the *known-secret*
#: pass can catch it: the secret-pattern pass would miss it.
KEY = "gw-tok-0123456789ABCDEFghijKLMNOP"

#: Shaped like a real OpenAI key: caught even when the SDK does not know it.
OPENAI_SHAPED = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


def _openai_401(key: str = KEY) -> dict[str, Any]:
    return {
        "error": {
            "message": f"Incorrect API key provided: {key}. You can find your API key at "
            "https://platform.openai.com/account/api-keys.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }


def _anthropic_401(key: str = KEY) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": "authentication_error", "message": f"invalid x-api-key: {key}"},
        "request_id": "req_011CABC",
    }


def _gemini_400(key: str = KEY) -> dict[str, Any]:
    return {
        "error": {
            "code": 400,
            "message": f"API key not valid. Please pass a valid API key. ({key})",
            "status": "INVALID_ARGUMENT",
        }
    }


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test states the keys it means to have; nothing from the shell."""
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "FASTAIAGENT_API_KEY",
        "FASTAIAGENT_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# The scrubber itself
# ---------------------------------------------------------------------------


class TestScrubSecrets:
    def test_a_known_secret_is_replaced_and_the_rest_kept(self) -> None:
        from fastaiagent._internal.scrub import REDACTED, scrub_secrets

        out = scrub_secrets(f"Incorrect API key provided: {KEY}.", [KEY])
        assert KEY not in out
        assert out == f"Incorrect API key provided: {REDACTED}."

    def test_url_encoded_forms_too(self) -> None:
        from urllib.parse import quote

        from fastaiagent._internal.scrub import scrub_secrets

        secret = "tok/with+special=chars&1234"
        out = scrub_secrets(f"GET /v1?key={quote(secret, safe='')}", [secret])
        assert quote(secret, safe="") not in out

    def test_a_half_masked_echo_is_replaced(self) -> None:
        from fastaiagent._internal.scrub import scrub_secrets

        out = scrub_secrets("Incorrect API key provided: sk-proj-****abcd.")
        assert "****abcd" not in out
        assert "Incorrect API key provided" in out

    def test_a_key_shaped_value_is_replaced_without_being_known(self) -> None:
        from fastaiagent._internal.scrub import scrub_secrets

        assert OPENAI_SHAPED not in scrub_secrets(f"bad key {OPENAI_SHAPED}")

    def test_short_values_are_not_replaced_verbatim(self) -> None:
        """A 3-character 'secret' would otherwise mangle ordinary words."""
        from fastaiagent._internal.scrub import scrub_secrets

        assert scrub_secrets("the cat sat", ["cat"]) == "the cat sat"


# ---------------------------------------------------------------------------
# Every provider path, against a real socket
# ---------------------------------------------------------------------------


def _raise(coro: Any) -> LLMProviderError:
    with pytest.raises(LLMProviderError) as exc:
        asyncio.run(coro)
    return exc.value


async def _drain(gen: Any) -> None:
    async for _ in gen:
        pass


def _assert_scrubbed(err: LLMProviderError, key: str = KEY) -> None:
    assert key not in str(err), f"the key is in the exception message: {err}"
    assert key not in repr(err.args)
    assert err.provider_message is None or key not in err.provider_message


class TestProviderPaths:
    def test_openai_complete(self, capture_server: Any) -> None:
        capture_server.set_response_sequence([(401, _openai_401())])
        client = LLMClient(provider="openai", model="m", base_url=capture_server.url, api_key=KEY)

        err = _raise(client.acomplete([UserMessage("hi")]))

        _assert_scrubbed(err)
        assert err.status_code == 401
        assert err.provider == "openai"
        assert err.provider_code == "invalid_api_key"
        assert err.provider_message and "Incorrect API key provided" in err.provider_message

    def test_openai_stream(self, capture_server: Any) -> None:
        capture_server.set_response_sequence([(401, _openai_401())])
        client = LLMClient(provider="openai", model="m", base_url=capture_server.url, api_key=KEY)

        err = _raise(_drain(client.astream([UserMessage("hi")])))

        _assert_scrubbed(err)

    def test_the_env_key_is_scrubbed_too(
        self, capture_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``api_key=`` on the client: it fell back to the env var, and the
        echo of *that* is what has to go."""
        monkeypatch.setenv("OPENAI_API_KEY", KEY)
        capture_server.set_response_sequence([(401, _openai_401())])
        client = LLMClient(provider="openai", model="m", base_url=capture_server.url)

        _assert_scrubbed(_raise(client.acomplete([UserMessage("hi")])))

    def test_anthropic_complete_and_stream(self, capture_server: Any) -> None:
        capture_server.set_response_sequence([(401, _anthropic_401()), (401, _anthropic_401())])
        client = LLMClient(
            provider="anthropic", model="m", base_url=capture_server.url, api_key=KEY
        )

        err = _raise(client.acomplete([UserMessage("hi")]))
        _assert_scrubbed(err)
        assert err.provider_code == "authentication_error"
        assert err.request_id == "req_011CABC"

        _assert_scrubbed(_raise(_drain(client.astream([UserMessage("hi")]))))

    def test_ollama_complete_and_stream(
        self, capture_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ollama sends no key, but a proxy in front of it can echo one from the env."""
        monkeypatch.setenv("OPENAI_API_KEY", KEY)
        body = {"error": f"upstream rejected token {KEY}"}
        capture_server.set_response_sequence([(502, body), (502, body)])
        client = LLMClient(provider="ollama", model="m", base_url=capture_server.url)

        err = _raise(client.acomplete([UserMessage("hi")]))
        _assert_scrubbed(err)
        assert err.provider_message and "upstream rejected token" in err.provider_message

        _assert_scrubbed(_raise(_drain(client.astream([UserMessage("hi")]))))

    def test_gemini_sends_the_key_as_a_header_not_in_the_url(self, capture_server: Any) -> None:
        capture_server.set_response_sequence([(400, _gemini_400()), (400, _gemini_400())])
        client = LLMClient(provider="gemini", model="m", base_url=capture_server.url, api_key=KEY)

        err = _raise(client.acomplete([UserMessage("hi")]))
        _assert_scrubbed(err)
        assert err.provider_code == "INVALID_ARGUMENT"
        _assert_scrubbed(_raise(_drain(client.astream([UserMessage("hi")]))))

        assert len(capture_server.requests) == 2
        for request in capture_server.requests:
            assert KEY not in request["path"], f"the key is in the URL: {request['path']}"
            assert "key=" not in request["path"]
            headers = {k.lower(): v for k, v in request["headers"].items()}
            assert headers.get("x-goog-api-key") == KEY

    def test_decisions(self, capture_server: Any) -> None:
        from fastaiagent.llm.decisions import Choice

        capture_server.set_response_sequence([(401, _openai_401())])
        client = LLMClient(provider="openai", model="m", base_url=capture_server.url, api_key=KEY)

        err = _raise(client.adecide("x", [Choice(instructions="pick", options=["a", "b"])]))
        _assert_scrubbed(err)


# ---------------------------------------------------------------------------
# Every channel downstream of a failed traced run
# ---------------------------------------------------------------------------


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _all_rows(db_path: Any) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return "\n".join(
            repr(row)
            for t in tables
            for row in conn.execute(f'SELECT * FROM "{t}"')  # noqa: S608 — names from sqlite_master
        )
    finally:
        conn.close()


def _stored_spans() -> list[Any]:
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore.default()
    conn = sqlite3.connect(str(store.db_path)) if hasattr(store, "db_path") else None
    if conn is None:  # pragma: no cover — TraceStore always exposes its path
        return []
    try:
        ids = [r[0] for r in conn.execute("SELECT DISTINCT trace_id FROM spans")]
    finally:
        conn.close()
    return [s for tid in ids for s in store.get_trace(tid).spans]


class TestTracedRun:
    def test_no_channel_carries_the_echoed_key(
        self, traced_db: Any, capture_server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        import fastaiagent as fa
        from fastaiagent.trace import otel
        from fastaiagent.trace.platform_export import to_wire

        third_party = InMemorySpanExporter()
        otel.add_span_processor(SimpleSpanProcessor(otel._EgressFilteredExporter(third_party)))
        capture_server.set_response_sequence([(401, _openai_401())])
        agent = fa.Agent(
            name="p9",
            system_prompt="s",
            llm=LLMClient(provider="openai", model="m", base_url=capture_server.url, api_key=KEY),
        )

        with pytest.raises(LLMProviderError):
            agent.run("hi")

        rows = _all_rows(traced_db)
        assert "Incorrect API key provided" in rows, (
            "the provider's error was not recorded, so this test checked nothing"
        )
        assert KEY not in rows, "local.db holds the echoed key"

        spans = _stored_spans()
        assert spans
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        assert KEY not in json.dumps([to_wire(s) for s in spans], default=str)

        finished = third_party.get_finished_spans()
        assert finished
        exported = json.dumps(
            [
                {
                    "status": getattr(s.status, "description", None),
                    "events": [dict(e.attributes or {}) for e in s.events],
                }
                for s in finished
            ],
            default=str,
        )
        assert KEY not in exported, "an add_exporter sink received the echoed key"

    def test_an_exception_the_sdk_did_not_build_is_scrubbed_when_stored(
        self, traced_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback: user code formats an env-held key into its own exception."""
        from fastaiagent.trace import otel

        monkeypatch.setenv("OPENAI_API_KEY", KEY)
        tracer = otel.get_tracer("p9-test")
        with pytest.raises(RuntimeError):
            with tracer.start_as_current_span("user.op"):
                raise RuntimeError(f"gateway refused token {KEY}")

        rows = _all_rows(traced_db)
        assert "gateway refused token" in rows
        assert KEY not in rows


# ---------------------------------------------------------------------------
# The Playground's server log
# ---------------------------------------------------------------------------


class TestPlaygroundLog:
    def test_the_failure_log_line_carries_no_key(
        self,
        tmp_path: Any,
        capture_server: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from fastaiagent._internal.config import reset_config
        from fastaiagent.ui.db import init_local_db
        from fastaiagent.ui.server import build_app

        db_path = tmp_path / "local.db"
        init_local_db(db_path).close()
        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
        reset_config()
        capture_server.set_response_sequence([(401, _openai_401())])
        client = TestClient(build_app(db_path=str(db_path), no_auth=True))

        with caplog.at_level(logging.WARNING):
            r = client.post(
                "/api/playground/run",
                json={
                    "provider": "custom",
                    "model": "m",
                    "prompt_template": "hi",
                    "variables": {},
                    "base_url": capture_server.url,
                    "api_key": KEY,
                },
            )
        reset_config()

        assert r.status_code == 502, r.text
        assert "Playground /run LLM call failed" in caplog.text
        assert "Incorrect API key provided" in caplog.text, "the log lost the provider's reason"
        assert KEY not in caplog.text, "the server log carries the key"
        assert KEY not in r.text
