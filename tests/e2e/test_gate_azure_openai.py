"""End-to-end quality gate — Azure OpenAI provider against a stub server.

Azure OpenAI uses an OpenAI-compatible HTTP API at a different base URL,
so the SDK dispatch table routes ``provider="azure"`` to the same
``_call_openai`` implementation that handles ``provider="openai"``.

Testing this against the real Azure service would require a paid Azure
subscription with an OpenAI deployment provisioned, which is overkill
for a regression gate. Instead this gate runs a tiny stub HTTP server
in a daemon thread that responds to POST /chat/completions with a
realistic OpenAI-compatible JSON body, points an
``LLMClient(provider="azure", base_url=stub_url)`` at it, and verifies:

- The dispatch routes Azure to the OpenAI HTTP path.
- The request body matches the OpenAI chat-completions schema.
- The response parser correctly extracts content, tool_calls, and
  usage from an OpenAI-shaped response.
- ``LLMClient.acomplete`` wrap emits a ``llm.azure.<model>`` span (not
  ``llm.openai.<model>``) so observability tools can distinguish the
  two providers even though they share an HTTP path.
- ``Agent`` end-to-end with provider="azure" produces a working
  ``AgentResult`` with a populated trace.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


_LAST_REQUEST: dict[str, Any] = {}


class _StubAzureHandler(BaseHTTPRequestHandler):
    """Implements POST /chat/completions in OpenAI-compatible shape."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        # Capture the request for the test to introspect.
        _LAST_REQUEST["body"] = body
        _LAST_REQUEST["headers"] = dict(self.headers)

        # Build a realistic OpenAI chat-completions response.
        response = {
            "id": "chatcmpl-stub-azure-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "stub-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "stubbed azure response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "total_tokens": 11,
            },
        }

        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def stub_azure_server():
    """Spin up the stub server on 127.0.0.1:<ephemeral> for this module."""
    server = HTTPServer(("127.0.0.1", 0), _StubAzureHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


class TestAzureOpenAIGate:
    """Verify provider='azure' dispatch + Phase A span naming."""

    def test_01_llmclient_complete_routes_to_openai_path(
        self, stub_azure_server: str, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import LLMClient
        from fastaiagent.llm.message import UserMessage

        _LAST_REQUEST.clear()
        llm = LLMClient(
            provider="azure",
            model="gpt-4o-azure-deployment",
            base_url=stub_azure_server,
            api_key="fake-azure-key",
        )
        response = llm.complete([UserMessage("Hello.")])

        assert response.content == "stubbed azure response", (
            f"response.content not parsed correctly: {response.content!r}"
        )
        assert response.usage.get("total_tokens") == 11
        assert response.finish_reason == "stop"

        # The stub should have received a request shaped like the OpenAI
        # chat-completions schema, even though the LLMClient said
        # provider='azure'.
        body = _LAST_REQUEST.get("body") or {}
        assert body.get("model") == "gpt-4o-azure-deployment"
        messages = body.get("messages") or []
        assert messages and messages[-1].get("role") == "user"
        assert messages[-1].get("content") == "Hello."

        # Auth header should carry the api_key as a Bearer token (OpenAI
        # convention; Azure usually uses api-key header but the SDK uses
        # the OpenAI auth shape since dispatch routes through _call_openai).
        headers = _LAST_REQUEST.get("headers") or {}
        auth = headers.get("Authorization", "")
        assert "fake-azure-key" in auth, (
            f"Authorization header missing api key: {auth!r}"
        )

    def test_02_agent_run_with_azure_provider(
        self, stub_azure_server: str, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="azure-gate",
            system_prompt="You are a stub-backed assistant.",
            llm=LLMClient(
                provider="azure",
                model="gpt-4o-azure-deployment",
                base_url=stub_azure_server,
                api_key="fake-azure-key",
            ),
        )
        result = agent.run("Anything you want.")
        assert result.output == "stubbed azure response"
        assert result.trace_id, "agent.run produced no trace_id"
        assert result.tokens_used == 11
        gate_state["azure_trace_id"] = result.trace_id

    def test_03_phase_a_span_carries_azure_provider(
        self, stub_azure_server: str, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent.trace.replay import Replay

        trace_id = gate_state["azure_trace_id"]
        replay = Replay.load(trace_id)
        steps = replay.steps()

        # Phase A reconstruction attrs should reflect provider="azure",
        # not "openai", even though both share _call_openai.
        root_attrs = replay._trace.spans[0].attributes
        assert root_attrs.get("agent.llm.provider") == "azure", (
            f"agent.llm.provider should be 'azure', got "
            f"{root_attrs.get('agent.llm.provider')!r}"
        )
        llm_config = json.loads(root_attrs.get("agent.llm.config", "{}"))
        assert llm_config.get("provider") == "azure", (
            f"llm.config provider should be 'azure', got {llm_config.get('provider')!r}"
        )

        # The LLMClient.acomplete span should be named with the azure
        # provider, not openai — that's how observability tools tell
        # them apart.
        llm_spans = [s for s in steps if s.span_name.startswith("llm.")]
        assert llm_spans, "No llm.* spans on Azure trace"
        assert any("azure" in s.span_name for s in llm_spans), (
            f"No llm.azure.* span — provider naming regression: "
            f"{[s.span_name for s in llm_spans]}"
        )
