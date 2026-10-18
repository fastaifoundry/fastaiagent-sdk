"""MCP ``expose_tools`` calls run through the governed tool-call path (1.90.0, F3).

``FastAIAgentMCPServer(agent, expose_tools=True)`` lets an MCP client call the
agent's tools by name. Since 1.78.0 that asked governance first; then it called
``tool.aexecute`` directly, so an MCP client could run a tool with arguments a
``tool_call`` guardrail would have blocked, and receive a result a
``tool_result`` guardrail would have masked — the agent's own guardrails,
skipped by the one caller that is not the agent. No middleware, no tool span,
no ``Tool.timeout`` / retries either.

Real MCP protocol over the upstream SDK's in-memory transport, real Agent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from fastaiagent import Agent  # noqa: E402
from fastaiagent.agent.middleware import AgentMiddleware  # noqa: E402
from fastaiagent.guardrail.guardrail import (  # noqa: E402
    Guardrail,
    GuardrailPosition,
    GuardrailType,
)
from fastaiagent.testing import TestModel  # noqa: E402
from fastaiagent.tool.function import FunctionTool  # noqa: E402
from fastaiagent.tool.mcp_server import FastAIAgentMCPServer  # noqa: E402

SSN = "123-45-6789"
SSN_RE = r"\b\d{3}-\d{2}-\d{4}\b"


def _regex(position: GuardrailPosition, action: str, name: str) -> Guardrail:
    return Guardrail(
        name=name,
        guardrail_type=GuardrailType.regex,
        position=position,
        config={"pattern": SSN_RE, "should_match": False},
        action=action,
    )


async def _call(agent: Agent, tool: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    from mcp.shared.memory import create_connected_server_and_client_session

    server = FastAIAgentMCPServer(agent, expose_tools=True)
    async with create_connected_server_and_client_session(server._server) as client:
        result = await client.call_tool(tool, arguments=arguments)
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    return bool(result.isError), text


def _lookup_tool(ran: list[str], reply: str = "ok") -> FunctionTool:
    def lookup(query: str) -> str:
        """Look a customer up."""
        ran.append(query)
        return reply

    return FunctionTool(name="lookup", fn=lookup)


@pytest.mark.asyncio
async def test_a_tool_call_guardrail_blocks_an_mcp_call() -> None:
    ran: list[str] = []
    agent = Agent(
        name="desk",
        llm=TestModel(response="unused"),
        tools=[_lookup_tool(ran)],
        guardrails=[_regex(GuardrailPosition.tool_call, "block", "no-ssn-in-args")],
    )

    is_error, text = await _call(agent, "lookup", {"query": f"ssn {SSN}"})

    assert is_error, f"a call the agent's guardrail blocks ran over MCP: {text!r}"
    assert ran == [], "the tool ran"
    assert "no-ssn-in-args" in text


@pytest.mark.asyncio
async def test_a_tool_result_mask_applies_to_an_mcp_call() -> None:
    ran: list[str] = []
    agent = Agent(
        name="desk",
        llm=TestModel(response="unused"),
        tools=[_lookup_tool(ran, reply=f"customer ssn {SSN}")],
        guardrails=[_regex(GuardrailPosition.tool_result, "mask", "mask-ssn")],
    )

    is_error, text = await _call(agent, "lookup", {"query": "jane"})

    assert not is_error, text
    assert ran == ["jane"]
    assert SSN not in text, f"the unmasked result reached the MCP client: {text!r}"
    assert "[REDACTED]" in text


@pytest.mark.asyncio
async def test_a_tool_result_block_is_an_error_without_the_payload() -> None:
    agent = Agent(
        name="desk",
        llm=TestModel(response="unused"),
        tools=[_lookup_tool([], reply=f"customer ssn {SSN}")],
        guardrails=[_regex(GuardrailPosition.tool_result, "block", "no-ssn-out")],
    )

    is_error, text = await _call(agent, "lookup", {"query": "jane"})

    assert is_error
    assert SSN not in text
    assert "no-ssn-out" in text


@pytest.mark.asyncio
async def test_wrap_tool_middleware_wraps_an_mcp_call() -> None:
    seen: list[dict[str, Any]] = []

    class Recorder(AgentMiddleware):
        name = "recorder"

        async def wrap_tool(self, ctx, tool, args, call_next):
            seen.append(dict(args))
            return await call_next(tool, {"query": args["query"].upper()})

    ran: list[str] = []
    agent = Agent(
        name="desk",
        llm=TestModel(response="unused"),
        tools=[_lookup_tool(ran)],
        middleware=[Recorder()],
    )

    is_error, _ = await _call(agent, "lookup", {"query": "jane"})

    assert not is_error
    assert seen == [{"query": "jane"}]
    assert ran == ["JANE"], "the middleware's rewrite was not applied"


@pytest.mark.asyncio
async def test_the_tool_timeout_applies_to_an_mcp_call() -> None:
    async def slow(query: str) -> str:
        """Slow lookup."""
        await asyncio.sleep(0.5)
        return "late"

    agent = Agent(
        name="desk",
        llm=TestModel(response="unused"),
        tools=[FunctionTool(name="slow", fn=slow, timeout=0.05)],
    )

    _, text = await _call(agent, "slow", {"query": "jane"})

    assert "timed out" in text, text


@pytest.mark.asyncio
async def test_an_ungoverned_mcp_call_is_unchanged() -> None:
    """The control: no guardrails, no middleware — same output as before."""
    agent = Agent(
        name="desk", llm=TestModel(response="unused"), tools=[_lookup_tool([], reply="found")]
    )
    assert await _call(agent, "lookup", {"query": "jane"}) == (False, "found")
