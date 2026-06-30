"""Tests for delegating LLMClient calls to an injected OpenAI-SDK client.

No mocks: these use the *real* ``openai`` SDK clients (sync, async, and
``AzureOpenAI``) pointed at a tiny in-process stub HTTP server. This is the
path that lets the SDK reach Azure OpenAI behind a corporate gateway — the
pre-built ``AzureOpenAI`` client handles the classic deployments URL,
``api_version``, Azure AD token refresh, and ``verify=False`` http_client,
while LLMClient builds the body, emits spans, and parses the response.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import UserMessage

openai = pytest.importorskip("openai")


_LAST: dict[str, Any] = {}


def _completion_json(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stub-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello from stub"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _sse_chunks(model: str) -> bytes:
    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        c = {
            "id": "chatcmpl-stub-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(c)}\n\n"

    parts = [
        chunk({"role": "assistant", "content": "hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop"),
        "data: [DONE]\n\n",
    ]
    return "".join(parts).encode()


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.split("?")[0].endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode() or "{}")
        _LAST["body"] = body
        _LAST["path"] = self.path
        _LAST["auth"] = self.headers.get("Authorization")
        model = body.get("model", "stub-model")

        if body.get("stream"):
            payload = _sse_chunks(model)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload)
            return

        payload = json.dumps(_completion_json(model)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def stub_url():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def test_sync_openai_client_delegation(stub_url: str) -> None:
    client = openai.OpenAI(base_url=stub_url, api_key="x")
    llm = LLMClient(provider="azure", model="my-deployment", openai_client=client)

    resp = llm.complete([UserMessage("hi")])
    assert resp.content == "hello from stub"
    assert resp.usage.get("total_tokens") == 8
    assert resp.finish_reason == "stop"
    # The request actually went through the injected client.
    assert _LAST["body"]["model"] == "my-deployment"


def test_azure_openai_client_uses_deployments_url(stub_url: str) -> None:
    # The real AzureOpenAI client builds the classic deployments + api-version
    # URL — exactly the shape fastaiagent's native path could not produce.
    client = openai.AzureOpenAI(
        azure_endpoint=stub_url, api_version="2024-10-21", api_key="x"
    )
    llm = LLMClient(provider="azure", model="my-deployment", openai_client=client)

    resp = llm.complete([UserMessage("hi")])
    assert resp.content == "hello from stub"
    assert "/openai/deployments/my-deployment/chat/completions" in _LAST["path"]
    assert "api-version=2024-10-21" in _LAST["path"]


def test_azure_ad_token_provider_is_used(stub_url: str) -> None:
    # Mirrors the user's managed-identity setup: AzureOpenAI is given a
    # token-provider callable instead of an api_key. The openai SDK calls it
    # to mint a Bearer token per request (this is where token *refresh* would
    # happen). Verify the resulting token actually reaches the wire.
    calls = {"n": 0}

    def token_provider() -> str:
        calls["n"] += 1
        return "managed-identity-token-123"

    client = openai.AzureOpenAI(
        azure_endpoint=stub_url,
        api_version="2025-01-01-preview",
        azure_ad_token_provider=token_provider,
    )
    llm = LLMClient(provider="azure", model="gpt-5.1", openai_client=client)

    resp = llm.complete([UserMessage("hi")])
    assert resp.content == "hello from stub"
    assert _LAST["auth"] == "Bearer managed-identity-token-123"
    assert calls["n"] >= 1  # the provider was invoked to mint the token


async def test_async_openai_client_delegation(stub_url: str) -> None:
    client = openai.AsyncOpenAI(base_url=stub_url, api_key="x")
    llm = LLMClient(provider="azure", model="dep", openai_client=client)
    assert llm._openai_client_is_async is True

    resp = await llm.acomplete([UserMessage("hi")])
    assert resp.content == "hello from stub"


def test_sync_detected_as_not_async(stub_url: str) -> None:
    client = openai.OpenAI(base_url=stub_url, api_key="x")
    llm = LLMClient(provider="azure", model="dep", openai_client=client)
    assert llm._openai_client_is_async is False


async def test_astream_via_injected_client(stub_url: str) -> None:
    from fastaiagent.llm.stream import StreamDone, TextDelta

    client = openai.OpenAI(base_url=stub_url, api_key="x")
    llm = LLMClient(provider="azure", model="dep", openai_client=client)

    text = ""
    saw_done = False
    async for ev in llm.astream([UserMessage("hi")]):
        if isinstance(ev, TextDelta):
            text += ev.text
        elif isinstance(ev, StreamDone):
            saw_done = True
    assert text == "hello"
    assert saw_done


def test_sync_stream_collects_response(stub_url: str) -> None:
    # ``stream()`` collects the streamed deltas into one LLMResponse.
    client = openai.OpenAI(base_url=stub_url, api_key="x")
    llm = LLMClient(provider="azure", model="dep", openai_client=client)
    resp = llm.stream([UserMessage("hi")])
    assert resp.content == "hello"
