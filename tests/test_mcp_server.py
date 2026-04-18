"""Tests for ``fastaiagent.tool.mcp_server``.

Unit tests cover the tool-definition shape, name sanitization, and the
``describe()`` introspection helper. A full-protocol end-to-end test spins
up a real MCP server in-process using the upstream ``mcp`` SDK's
in-memory transport and drives it with the SDK's :class:`ClientSession`
— so the handshake, ``tools/list``, ``tools/call``, ``prompts/list``, and
``prompts/get`` paths are all exercised against the real protocol with no
mocking.

The ``agent.arun`` path uses :class:`tests.conftest.MockLLMClient` because
live API calls from inside a server subprocess would make the tests flaky
— this is test infrastructure, not mocking of the feature under test.
"""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp")

from fastaiagent import Agent, Chain  # noqa: E402 — after importorskip
from fastaiagent.llm.client import LLMResponse  # noqa: E402
from fastaiagent.tool.function import FunctionTool  # noqa: E402
from fastaiagent.tool.mcp_server import (  # noqa: E402
    FastAIAgentMCPServer,
    _sanitize_name,
)
from tests.conftest import MockLLMClient  # noqa: E402

# ---------------------------------------------------------------------------
# Unit: name sanitization, describe()
# ---------------------------------------------------------------------------


def test_sanitize_name_replaces_non_alnum() -> None:
    assert _sanitize_name("research-assistant") == "research_assistant"
    assert _sanitize_name("my agent") == "my_agent"
    assert _sanitize_name("ok_name_2") == "ok_name_2"
    assert _sanitize_name("") == "fastaiagent"


def test_describe_reports_primary_only_by_default() -> None:
    a = Agent(name="demo", llm=MockLLMClient())
    server = FastAIAgentMCPServer(a)
    desc = server.describe()
    assert desc["transport"] == "stdio"
    assert desc["target_name"] == "demo"
    assert len(desc["tools"]) == 1
    assert desc["tools"][0]["primary"] is True
    assert desc["tools"][0]["name"] == "demo"


def test_describe_expose_tools_surfaces_inner_tools() -> None:
    def hello(who: str) -> str:
        return f"hello {who}"

    hello_tool = FunctionTool(
        name="hello",
        fn=hello,
        description="Say hello",
        parameters={
            "type": "object",
            "properties": {"who": {"type": "string"}},
            "required": ["who"],
        },
    )
    a = Agent(name="demo", llm=MockLLMClient(), tools=[hello_tool])
    server = FastAIAgentMCPServer(a, expose_tools=True)
    names = [t["name"] for t in server.describe()["tools"]]
    assert "demo" in names
    assert "hello" in names


def test_describe_hyphens_sanitized_in_primary_name() -> None:
    a = Agent(name="research-bot", llm=MockLLMClient())
    desc = FastAIAgentMCPServer(a).describe()
    # Primary tool name must be a valid MCP identifier.
    assert desc["tools"][0]["name"] == "research_bot"


def test_chain_as_mcp_server_exposes_chain() -> None:
    c = Chain(name="pipeline")
    server = c.as_mcp_server()
    desc = server.describe()
    assert desc["target_name"] == "pipeline"
    # Chains don't expose system prompts.
    assert desc["expose_system_prompt"] is False


def test_unsupported_transport_raises_on_run() -> None:
    import asyncio

    a = Agent(name="demo", llm=MockLLMClient())
    server = FastAIAgentMCPServer(a, transport="sse")
    with pytest.raises(NotImplementedError, match="sse"):
        asyncio.run(server.run())


# ---------------------------------------------------------------------------
# Full-protocol end-to-end via MCP's in-memory transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_server_full_protocol_roundtrip() -> None:
    """Start a real MCP server in-process and drive it with a real MCP client.

    Exercises: initialize handshake, tools/list, tools/call (primary),
    prompts/list, prompts/get — all via the upstream ``mcp`` SDK's protocol
    machinery. No mocking of the protocol itself.
    """
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_connected_server_and_client_session

    agent = Agent(
        name="research-bot",
        system_prompt="Answer briefly.",
        llm=MockLLMClient(
            responses=[LLMResponse(content="hello from the agent", finish_reason="stop")]
        ),
    )
    server = FastAIAgentMCPServer(agent)

    async with create_connected_server_and_client_session(server._server) as client:
        client: ClientSession
        # Handshake happens inside the context manager.

        # tools/list
        tools_resp = await client.list_tools()
        tool_names = {t.name for t in tools_resp.tools}
        assert "research_bot" in tool_names

        # tools/call primary
        call_resp = await client.call_tool(
            "research_bot", arguments={"input": "hi"}
        )
        assert call_resp.content, "expected at least one content part"
        text_parts = [c.text for c in call_resp.content if hasattr(c, "text")]
        assert any("hello from the agent" in t for t in text_parts)

        # prompts/list
        prompts_resp = await client.list_prompts()
        prompt_names = {p.name for p in prompts_resp.prompts}
        assert "research_bot_system" in prompt_names

        # prompts/get
        prompt = await client.get_prompt("research_bot_system")
        assert prompt.messages
        text_parts = [
            m.content.text
            for m in prompt.messages
            if hasattr(m.content, "text")
        ]
        assert "Answer briefly." in " ".join(text_parts)


@pytest.mark.asyncio
async def test_mcp_server_call_tool_with_unknown_name_errors() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    agent = Agent(
        name="demo",
        llm=MockLLMClient(responses=[LLMResponse(content="unused", finish_reason="stop")]),
    )
    server = FastAIAgentMCPServer(agent)
    async with create_connected_server_and_client_session(server._server) as client:
        result = await client.call_tool("not_a_real_tool", arguments={})
        # MCP surfaces errors via isError on the result, not an exception.
        assert result.isError is True


@pytest.mark.asyncio
async def test_mcp_server_exposes_inner_tools_when_flag_on() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    def echo(text: str) -> str:
        return f"echoed: {text}"

    echo_tool = FunctionTool(
        name="echo",
        fn=echo,
        description="Echo text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    agent = Agent(
        name="demo",
        llm=MockLLMClient(responses=[LLMResponse(content="unused", finish_reason="stop")]),
        tools=[echo_tool],
    )
    server = FastAIAgentMCPServer(agent, expose_tools=True)
    async with create_connected_server_and_client_session(server._server) as client:
        tools = (await client.list_tools()).tools
        assert "echo" in {t.name for t in tools}
        assert "demo" in {t.name for t in tools}

        result = await client.call_tool("echo", arguments={"text": "ping"})
        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        assert any("echoed: ping" in t for t in text_parts)


@pytest.mark.asyncio
async def test_mcp_server_chain_roundtrip() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    # Trivial chain with a transformer node so aexecute produces output.
    chain = Chain(name="echo-chain")
    chain.add_node("echo", type="transformer", template="CHAIN SAW: {{input.input}}")

    server = chain.as_mcp_server()
    async with create_connected_server_and_client_session(server._server) as client:
        tools = (await client.list_tools()).tools
        assert "echo_chain" in {t.name for t in tools}
        result = await client.call_tool(
            "echo_chain", arguments={"input": "knock knock"}
        )
        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        assert text_parts  # chain produced some output
