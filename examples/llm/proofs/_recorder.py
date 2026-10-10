"""A local endpoint that records what LLMClient sends and answers with a canned reply.

It speaks the two wire shapes the SDK builds: ``POST …/chat/completions``
(OpenAI-compatible, used by ``openai``, ``custom``, ``azure`` and every preset)
and ``POST …/messages`` (Anthropic). The request bodies it records are the real
bytes the client put on the wire; only the model behind it is canned.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Recorder:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.statuses = list(statuses or [])
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> str:
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.loads(raw or b"{}")
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): ("***" if k.lower() in ("authorization", "x-api-key") else v)
                                    for k, v in self.headers.items()
                                    if k.lower() in ("authorization", "x-api-key", "anthropic-version")},
                        "body": body,
                    }
                )
                status = recorder.statuses.pop(0) if recorder.statuses else 200
                if status != 200:
                    payload = {"error": {"message": f"canned {status}", "type": "server_error"}}
                else:
                    payload = recorder._reply(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @staticmethod
    def _reply(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path.endswith("/messages"):  # Anthropic shape
            return {
                "id": "rec", "model": body.get("model", ""), "role": "assistant",
                "content": [{"type": "text", "text": "recorded"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        return {  # OpenAI-compatible shape
            "id": "rec", "model": body.get("model", ""),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "recorded"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]
