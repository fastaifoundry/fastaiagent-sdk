"""A real localhost HTTP server that streams scripted OpenAI / Anthropic SSE (1.89.0).

No mocks: the real ``openai`` / ``anthropic`` SDK clients (and LangChain's
``ChatOpenAI``) talk to it over a real socket. Each request is answered by the
next scripted :class:`Reply`:

* a JSON body with any status (an error before the first chunk);
* an SSE stream, which can carry a provider error event part-way;
* an SSE stream **cut** part-way: the server sends some chunks with
  ``Transfer-Encoding: chunked`` and then closes the socket without the final
  chunk, so ``httpx`` raises ``RemoteProtocolError`` mid-iteration. That is
  what a dropped connection looks like to a client.

The last reply repeats once the script drains.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

#: A key-shaped string the error replies echo back, to prove it is scrubbed.
ECHOED_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd"


@dataclass
class Reply:
    status: int = 200
    #: SSE events as ``(event_name_or_None, payload)``; ``payload`` may be a
    #: dict (JSON-encoded) or a raw string such as ``"[DONE]"``.
    events: list[tuple[str | None, Any]] = field(default_factory=list)
    #: A non-stream body, sent as JSON when set.
    body: Any = None
    #: Close the socket after this many events, without ending the body.
    cut_after: int | None = None


def sse(event: str | None, payload: Any) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {data}\n\n".encode()


class SSEProvider:
    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests: list[dict[str, Any]] = []
        self._server: HTTPServer | None = None

    def script(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests = []

    def _next(self) -> Reply:
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0] if self.replies else Reply(body={})

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> SSEProvider:
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return

            def _chunk(self, raw: bytes) -> None:
                self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                self.wfile.flush()

            def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                provider.requests.append({"path": self.path, "body": body})
                reply = provider._next()
                if reply.body is not None or not reply.events:
                    payload = json.dumps(reply.body or {}).encode()
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                    self.close_connection = True
                    return
                self.send_response(reply.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                for i, (name, payload) in enumerate(reply.events):
                    if reply.cut_after is not None and i >= reply.cut_after:
                        # Drop the connection mid-body: no terminating chunk.
                        self.close_connection = True
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        return
                    self._chunk(sse(name, payload))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                self.close_connection = True

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# -- OpenAI chat-completions shapes -------------------------------------------


def openai_chunk(content: str | None = None, *, finish: str | None = None, **extra: Any) -> dict:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    return {
        "id": "chatcmpl-sse",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }


def openai_usage_chunk(prompt: int = 11, completion: int = 7) -> dict:
    return {
        "id": "chatcmpl-sse",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def openai_stream(*texts: str, usage: bool = True) -> list[tuple[str | None, Any]]:
    events: list[tuple[str | None, Any]] = [(None, openai_chunk(t)) for t in texts]
    events.append((None, openai_chunk(finish="stop")))
    if usage:
        events.append((None, openai_usage_chunk()))
    events.append((None, "[DONE]"))
    return events


def openai_error_event() -> tuple[str | None, Any]:
    return (
        None,
        {"error": {"message": f"upstream failed for key {ECHOED_KEY}", "type": "server_error"}},
    )


# -- Anthropic messages shapes ------------------------------------------------


def anthropic_stream(*texts: str) -> list[tuple[str | None, Any]]:
    events: list[tuple[str | None, Any]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_sse",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 13, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for t in texts:
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": t},
                },
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


def anthropic_error_event() -> tuple[str | None, Any]:
    return (
        "error",
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": f"Overloaded for key {ECHOED_KEY}"},
        },
    )


def error_body() -> dict[str, Any]:
    """A provider error body that echoes a key, as OpenAI does on a 401."""
    return {
        "error": {
            "message": f"Incorrect API key provided: {ECHOED_KEY}",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        },
        "type": "error",
    }
