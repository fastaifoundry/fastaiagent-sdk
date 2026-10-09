"""An in-process stand-in for OpenAI's ``POST /v1/decisions``.

No mocks: tests drive the real ``LLMClient.adecide`` (httpx, retries, spans,
pricing) — or a real ``openai`` SDK client — against this tiny HTTP server, the
pattern ``tests/test_llm_injected_client.py`` established. Its reply shapes copy
a live response captured on 2026-10-09 (see the 1.84.0 notes): unnamed questions
come back with ``"name": null``, choice values keep their JSON type, and
``output_tokens`` is 0.

By default every question gets a deterministic answer:

* predicate → ``probability`` from ``stub.probabilities[name]`` (default 0.9)
* choice    → ``stub.choices[name]`` if set, else the first option
* score     → all weight on ``stub.levels[name]`` (default the last level)

``stub.refuse`` names questions to answer with a refusal. ``stub.script`` is a
queue of ``(status, body)`` pairs served before the default — for errors and
malformed replies.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

INPUT_TOKENS = 100


class DecisionsStub:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.probabilities: dict[str | None, float] = {}
        self.choices: dict[str | None, Any] = {}
        self.levels: dict[str | None, int] = {}
        self.refuse: set[str | None] = set()
        self.script: list[tuple[int, Any]] = []
        #: Assistant messages ``/chat/completions`` hands out in order (last
        #: repeats) — for flows that mix a chat model with decisions.
        self.chat_replies: list[dict[str, Any]] = [{"role": "assistant", "content": "ok"}]
        self.chat_calls = 0
        self.base_url = ""
        self._server: HTTPServer | None = None

    # -- answers ----------------------------------------------------------

    def answer(self, q: dict[str, Any]) -> dict[str, Any]:
        name = q.get("name")
        if name in self.refuse:
            return {"type": "refusal", "name": name}
        if q["type"] == "predicate":
            return {
                "type": "predicate",
                "name": name,
                "probability": self.probabilities.get(name, 0.9),
            }
        if q["type"] == "choice":
            values = [c["value"] for c in q["choices"]]
            pick = self.choices.get(name, values[0])
            return {
                "type": "choice",
                "name": name,
                "choice": pick,
                "probabilities": [
                    {"value": v, "probability": 1.0 if v == pick and type(v) is type(pick) else 0.0}
                    for v in values
                ],
                "confidence": 1.0,
            }
        labels = [lvl["label"] for lvl in q["levels"]]
        idx = self.levels.get(name, len(labels) - 1)
        return {
            "type": "score",
            "name": name,
            "score": float(idx),
            "probabilities": [
                {"value": i, "label": lbl, "probability": 1.0 if i == idx else 0.0}
                for i, lbl in enumerate(labels)
            ],
            "confidence": 1.0,
        }

    def reply(self, body: dict[str, Any]) -> tuple[int, Any]:
        if self.script:
            return self.script.pop(0)
        return 200, {
            "model": body.get("model", "gpt-6-luna"),
            "answers": [self.answer(q) for q in body.get("questions", [])],
            "usage": {
                "input_tokens": INPUT_TOKENS,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": INPUT_TOKENS,
            },
        }

    def chat_reply(self, body: dict[str, Any]) -> dict[str, Any]:
        idx = min(self.chat_calls, len(self.chat_replies) - 1)
        self.chat_calls += 1
        message = self.chat_replies[idx]
        return {
            "id": f"chatcmpl-stub-{self.chat_calls}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "gpt-4o-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    # -- server -----------------------------------------------------------

    def start(self) -> None:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode() or "{}")
                stub.requests.append(
                    {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
                )
                path = self.path.split("?")[0]
                if path.endswith("/decisions"):
                    status, payload = stub.reply(body)
                elif path.endswith("/chat/completions"):
                    status, payload = 200, stub.chat_reply(body)
                else:
                    status, payload = 404, {"error": {"message": "not found"}}
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("x-request-id", f"req_stub_{len(stub.requests)}")
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def llm_kwargs(self, **extra: Any) -> dict[str, Any]:
        """``LLMClient`` kwargs pointing at this stub — also a guardrail ``config["llm"]``."""
        return {
            "provider": "custom",
            "model": "gpt-6-luna",
            "base_url": self.base_url,
            "api_key": "stub-key",
            **extra,
        }

    def client(self, **extra: Any) -> Any:
        from fastaiagent.llm import LLMClient

        return LLMClient(**self.llm_kwargs(**extra))


def started_stub() -> Iterator[DecisionsStub]:
    """Body of the ``decisions_stub`` fixture registered in ``tests/conftest.py``."""
    stub = DecisionsStub()
    stub.start()
    try:
        yield stub
    finally:
        stub.stop()
