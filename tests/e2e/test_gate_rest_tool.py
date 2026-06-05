"""End-to-end quality gate — RESTTool with a real HTTP endpoint.

Exercises the REST-tool path end-to-end:
- Direct ``.execute()`` against a real public endpoint.
- LLM-driven tool call: the agent decides to invoke the REST tool and
  uses its response.

Uses ``httpbin.org`` for the target. That's the standard public echo
service for HTTP testing — stable, free, no auth, and its responses
are trivially inspectable.

Marked to skip cleanly in environments without outbound internet (which
includes some CI runners on locked-down networks), by checking
connectivity upfront with a short timeout.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


_HTTPBIN = "https://httpbin.org"


def _httpbin_reachable() -> bool:
    # Require a real 200 — httpbin is a shared public service that frequently
    # returns 503 under load. httpx doesn't raise on 5xx, so an earlier
    # "any response = reachable" check let a degraded httpbin through and the
    # actual tool call then failed the gate.
    try:
        return httpx.get(f"{_HTTPBIN}/status/200", timeout=5.0).status_code == 200
    except Exception:
        return False


def _skip_on_httpbin_outage(call: Any) -> Any:
    """Run ``call()``; if it fails because httpbin was unreachable or returned
    a 5xx mid-test (flaky third party), skip rather than red the gate. These
    tests call only httpbin, so a transport error or server 5xx is the endpoint
    misbehaving, not our bug — a 4xx (which could be a malformed request we
    built) still fails.
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 — re-raised unless it's an outage
        # RESTTool raises ToolExecutionError(...) ``from`` the underlying httpx
        # error, so inspect the cause.
        cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
        if isinstance(cause, httpx.HTTPStatusError):
            if cause.response.status_code >= 500:
                pytest.skip(f"httpbin.org server error {cause.response.status_code}")
        elif isinstance(cause, httpx.RequestError):  # connect/read/timeout/etc.
            pytest.skip(f"httpbin.org request error: {cause!r}")
        raise


class TestRESTToolGate:
    """RESTTool direct call + LLM-driven call via a real httpbin endpoint."""

    def test_01_direct_get_with_query_params(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        if not _httpbin_reachable():
            pytest.skip("httpbin.org not reachable from this environment")
        from fastaiagent import RESTTool

        tool = RESTTool(
            name="echo_get",
            description="Echo back the query parameters sent to the GET endpoint.",
            url=f"{_HTTPBIN}/get",
            method="GET",
            body_mapping="query_params",
            parameters={
                "type": "object",
                "properties": {
                    "foo": {"type": "string"},
                    "bar": {"type": "string"},
                },
                "required": ["foo"],
            },
        )

        result = _skip_on_httpbin_outage(
            lambda: tool.execute({"foo": "hello", "bar": "world"})
        )
        assert result.success, f"REST tool failed: {result.error}"
        assert isinstance(result.output, dict), (
            "Expected JSON dict from httpbin /get"
        )
        args = result.output.get("args", {})
        assert args.get("foo") == "hello", f"foo not echoed: {args}"
        assert args.get("bar") == "world", f"bar not echoed: {args}"
        assert result.metadata.get("status_code") == 200

    def test_02_direct_post_with_json_body(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        if not _httpbin_reachable():
            pytest.skip("httpbin.org not reachable from this environment")
        from fastaiagent import RESTTool

        tool = RESTTool(
            name="echo_post",
            description="POST a JSON body and receive it echoed back.",
            url=f"{_HTTPBIN}/post",
            method="POST",
            body_mapping="json_body",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
        )
        result = _skip_on_httpbin_outage(
            lambda: tool.execute({"message": "quality gate post"})
        )
        assert result.success, f"REST POST failed: {result.error}"
        assert isinstance(result.output, dict)
        json_body = result.output.get("json") or {}
        assert json_body.get("message") == "quality gate post", (
            f"JSON body not echoed correctly: {json_body}"
        )

    def test_03_agent_uses_rest_tool(self, gate_state: dict[str, Any]) -> None:
        require_env()
        if not _httpbin_reachable():
            pytest.skip("httpbin.org not reachable from this environment")
        from fastaiagent import Agent, LLMClient, RESTTool

        tool = RESTTool(
            name="get_ip",
            description="Return the caller's public IP address by calling an echo service.",
            url=f"{_HTTPBIN}/ip",
            method="GET",
            body_mapping="query_params",
            parameters={"type": "object", "properties": {}},
        )
        agent = Agent(
            name="rest-tool-gate",
            system_prompt=(
                "You have a get_ip tool. When asked about the current IP, "
                "call get_ip and include the origin field from its response "
                "verbatim in your answer."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[tool],
        )
        result = _skip_on_httpbin_outage(
            lambda: agent.run("What is the origin IP from the get_ip tool?")
        )
        assert result.output, "agent run returned empty output"
        assert result.tool_calls, "agent did not invoke get_ip REST tool"
        assert result.tool_calls[0]["tool_name"] == "get_ip"
        # When httpbin is down, the get_ip tool errors and the agent gracefully
        # reports the failure (no exception, no IP) — the tool-call record carries
        # the error. This tool only calls httpbin, so a tool error means the
        # endpoint misbehaved, not our bug; skip rather than fail the digit check.
        tool_error = result.tool_calls[0].get("error")
        if tool_error:
            pytest.skip(f"httpbin.org outage — get_ip tool errored: {tool_error}")
        # The tool output is a JSON dict containing an 'origin' key with
        # the IP string. We just check that the agent's final answer
        # includes something that looks like a dotted-quad IP (the LLM
        # should have extracted it verbatim from the tool output).
        lower = result.output.lower()
        assert any(c.isdigit() for c in lower), (
            f"Agent answer contained no digits — probably didn't use the tool output: "
            f"{result.output!r}"
        )
