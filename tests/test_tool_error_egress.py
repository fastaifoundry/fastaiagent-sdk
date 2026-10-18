"""A tool's error text is payload, and stays home when payloads are off (1.90.0, F2).

``tool.error`` holds ``str(exception)`` or ``ToolResult.error`` — text that
routinely quotes the tool's arguments or the upstream response (``"no account
for jane@acme.com"``). It was set unconditionally by the agent loop and the
runner and was missing from ``SENSITIVE_ATTR_KEYS``, so it left the machine
with ``FASTAIAGENT_TRACE_PAYLOADS=0`` while ``tool.args`` / ``tool.result``, the
same content, were held back (CLAUDE.md §2.5).

Real Agent, real tool, real OTel provider writing a private ``local.db``; the
stored spans go through the platform exporter's own ``to_wire``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool
from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.testing import FunctionModel
from fastaiagent.tool.base import ToolResult

CANARY = "jane.doe-CANARY-7c41@acme.com"


@pytest.fixture
def traced_db(monkeypatch, tmp_path) -> Iterator[str]:
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    try:
        yield str(db)
    finally:
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()


def _raises(email: str) -> str:
    raise ToolExecutionError(f"no account for {email}")


class _ErrorResultTool(FunctionTool):
    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        return ToolResult(error=f"upstream said: unknown user {arguments['email']}")


def _responder(tool_name: str):
    def respond(messages):
        if any(m.role.value == "tool" for m in messages):
            return "Could not find that account."
        return "", [{"name": tool_name, "arguments": {"email": CANARY}}]

    return respond


def _tool_span_on_the_wire(db: str, trace_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.platform_export import to_wire
    from fastaiagent.trace.storage import TraceStore

    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    spans = TraceStore(db).get_trace(trace_id).spans
    tool = next(s for s in spans if s.name.startswith("tool."))
    return tool.attributes, to_wire(tool)


@pytest.mark.parametrize(
    "tool",
    [
        FunctionTool(name="lookup", fn=_raises),
        _ErrorResultTool(name="lookup", fn=_raises),
    ],
    ids=["raised", "error-result"],
)
@pytest.mark.parametrize("payloads", ["0", "1"])
def test_tool_error_follows_the_payload_gate(tool, payloads, traced_db, monkeypatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", payloads)
    agent = Agent(name="lookup-bot", llm=FunctionModel(_responder("lookup")), tools=[tool])

    result = agent.run("find jane")

    local, wire = _tool_span_on_the_wire(traced_db, result.trace_id)
    # Local capture is always full fidelity.
    assert CANARY in local["tool.error"]
    assert local["tool.status"] == "error"
    # Structure always flows, so the plane still sees *that* the tool failed.
    assert wire["attributes"]["tool.status"] == "error"
    if payloads == "0":
        assert "tool.error" not in wire["attributes"]
        assert CANARY not in json.dumps(wire), "the tool's error text egressed"
    else:
        assert CANARY in wire["attributes"]["tool.error"]


def test_the_registry_names_tool_error() -> None:
    from fastaiagent.trace.redaction import is_sensitive_attr

    assert is_sensitive_attr("tool.error")


# --- F2b: the same text on the *events* channel ------------------------------
#
# The CrewAI integration records a tool or LLM failure that is not an exception
# object as an event ``{"name": "tool.error" | "llm.error", "attributes":
# {"message": <error text>}}``. The event gate only knew ``exception.message`` /
# ``exception.stacktrace``, so this message egressed with payloads off.


@pytest.mark.parametrize("event_name", ["tool.error", "llm.error"])
@pytest.mark.parametrize("payloads", ["0", "1"])
def test_integration_error_events_follow_the_payload_gate(event_name, payloads, monkeypatch):
    from fastaiagent.trace.platform_export import to_wire
    from fastaiagent.trace.storage import SpanData

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", payloads)
    span = SpanData(
        span_id="s1",
        trace_id="t1",
        parent_span_id=None,
        name="crewai.tool.lookup",
        span_type="tool",
        start_time="2026-10-18T00:00:00+00:00",
        end_time="2026-10-18T00:00:01+00:00",
        status="ERROR",
        attributes={"tool.name": "lookup"},
        events=[
            {
                "name": event_name,
                "timestamp": "2026-10-18T00:00:00+00:00",
                "attributes": {"message": f"no account for {CANARY}"},
            }
        ],
    )
    wire = to_wire(span)
    assert wire["events"][0]["name"] == event_name  # that it failed always flows
    if payloads == "0":
        assert CANARY not in json.dumps(wire)
    else:
        assert CANARY in wire["events"][0]["attributes"]["message"]
