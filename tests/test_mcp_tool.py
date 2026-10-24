"""``MCPTool`` against real MCP servers built with the official ``mcp`` library.

Until 1.92.0 ``MCPTool`` spoke a hand-written JSON-RPC dialect that official
servers refuse (406 / 400), and its tests ran against a stub that accepted it.
Every test here talks to a server made by the library MCP hosts use: FastMCP over
streamable HTTP in-process (``official_mcp_server``), the same tools over stdio
(``tests/_mcp_servers.py``), and our own ``as_mcp_server``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from fastaiagent import Agent  # noqa: E402
from fastaiagent._internal.errors import GuardrailBlockedError, ToolExecutionError  # noqa: E402
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult  # noqa: E402
from fastaiagent.guardrail.guardrail import GuardrailType  # noqa: E402
from fastaiagent.testing.models import FunctionModel  # noqa: E402
from fastaiagent.tool.base import Tool  # noqa: E402
from fastaiagent.tool.mcp import MCPTool  # noqa: E402
from tests._mcp_servers import STDIO_COMMAND  # noqa: E402

# --- streamable HTTP ------------------------------------------------------------


async def test_from_server_returns_one_tool_per_remote_tool(official_mcp_server: Any) -> None:
    tools = await MCPTool.afrom_server(official_mcp_server.url)

    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"add", "fail", "notify", "whoami"}
    add = by_name["add"]
    assert add.description == "Add two numbers."
    assert add.parameters["required"] == ["a", "b"]
    assert add.server_url == official_mcp_server.url
    assert add.tool_name == "add"


def test_from_server_sync(official_mcp_server: Any) -> None:
    assert len(MCPTool.from_server(official_mcp_server.url)) == 4


async def test_call_returns_the_text_result(official_mcp_server: Any) -> None:
    tool = MCPTool(name="add", server_url=official_mcp_server.url)

    result = await tool.aexecute({"a": 2, "b": 3})

    assert result.output == "5"
    assert official_mcp_server.calls == [{"tool": "add", "a": 2, "b": 3}]


async def test_a_tool_error_comes_back_as_an_error_result(official_mcp_server: Any) -> None:
    result = await MCPTool(name="fail", server_url=official_mcp_server.url).aexecute(
        {"reason": "no"}
    )

    assert result.output is None
    assert "refused: no" in (result.error or "")


async def test_the_auth_token_reaches_the_server(official_mcp_server: Any) -> None:
    url = official_mcp_server.url
    with_token = await MCPTool(name="whoami", server_url=url, auth_token="s3cret").aexecute({})
    without = await MCPTool(name="whoami", server_url=url).aexecute({})

    assert with_token.output == "Bearer s3cret"
    assert without.output == "<none>"


def _calls_tool(name: str, arguments: dict[str, Any]) -> FunctionModel:
    def route(messages: list[Any]) -> Any:
        last = messages[-1]
        if last.role.value == "tool":
            return f"tool said: {last.content}"
        return "", [{"name": name, "arguments": arguments}]

    return FunctionModel(route)


async def test_an_agent_calls_the_mcp_tool(official_mcp_server: Any) -> None:
    tools = await MCPTool.afrom_server(official_mcp_server.url)
    agent = Agent(name="a", llm=_calls_tool("notify", {"text": "hi"}), tools=tools)

    result = await agent.arun("notify the customer", trace=False)

    assert result.output == "tool said: sent: hi"
    assert official_mcp_server.calls == [{"tool": "notify", "text": "hi"}]


async def test_a_tool_call_guardrail_stops_the_call_before_the_server(
    official_mcp_server: Any,
) -> None:
    def no_notify(text: str) -> GuardrailResult:
        return GuardrailResult(passed="notify" not in text, message="no notifications")

    agent = Agent(
        name="a",
        llm=_calls_tool("notify", {"text": "hi"}),
        tools=[MCPTool(name="notify", server_url=official_mcp_server.url)],
        guardrails=[
            Guardrail(
                name="no_notify",
                guardrail_type=GuardrailType.code,
                position=GuardrailPosition.tool_call,
                fn=no_notify,
            )
        ],
    )

    with pytest.raises(GuardrailBlockedError):
        await agent.arun("notify the customer", trace=False)
    assert official_mcp_server.calls == []


# --- failures (the SSRF guard is in tests/test_mcp_ssrf.py) ---------------------


async def test_an_unreachable_server_fails(unused_tcp_port: int) -> None:
    with pytest.raises(ToolExecutionError, match="MCP tool 'x' failed"):
        await MCPTool(name="x", server_url=f"http://127.0.0.1:{unused_tcp_port}/mcp").aexecute({})


async def test_a_server_that_is_not_mcp_fails_instead_of_hanging(
    capture_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The official client waits for a handshake reply forever unless told not to.
    import importlib

    monkeypatch.setattr(importlib.import_module("fastaiagent.tool.mcp"), "_HANDSHAKE_SECONDS", 1.0)

    with pytest.raises(ToolExecutionError, match="MCP tool 'x' failed"):
        await MCPTool(name="x", server_url=f"{capture_server.url}/mcp").aexecute({})


# --- stdio ----------------------------------------------------------------------


async def test_a_local_server_over_stdio() -> None:
    tools = await MCPTool.afrom_server(command=STDIO_COMMAND)
    add = next(t for t in tools if t.name == "add")

    assert add.command == STDIO_COMMAND
    assert (await add.aexecute({"a": 4, "b": 5})).output == "9"


async def test_mcp_tool_calls_our_own_mcp_server(tmp_path: Path) -> None:
    """Our server is stdio and the client used to be HTTP-only, so the docs
    said one could not call the other. Now it can."""
    script = tmp_path / "served.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            from fastaiagent import Agent, LLMClient
            from fastaiagent.tool import FunctionTool

            def double(n: int) -> int:
                \"\"\"Double a number.\"\"\"
                return n * 2

            agent = Agent(
                name="calc",
                llm=LLMClient(provider="openai", model="gpt-5-mini", api_key="sk-unused"),
                tools=[FunctionTool(name="double", fn=double)],
            )
            asyncio.run(agent.as_mcp_server(expose_tools=True).run())
            """
        )
    )
    tools = await MCPTool.afrom_server(command=[sys.executable, str(script)])
    double = next(t for t in tools if t.name == "double")

    assert {t.name for t in tools} == {"calc", "double"}
    assert (await double.aexecute({"n": 21})).output == "42"


# --- what a payload can carry ---------------------------------------------------


def test_a_payload_carries_no_command_env_or_token() -> None:
    local = MCPTool(name="x", command=["npx", "server"], env={"K": "v"}, auth_token="t")
    remote = MCPTool(name="y", server_url="https://mcp.example.com/mcp", auth_token="t")

    assert local.to_dict()["config"] == {"server_url": "", "tool_name": "x"}
    assert remote.to_dict()["config"] == {
        "server_url": "https://mcp.example.com/mcp",
        "tool_name": "y",
    }


async def test_a_local_tool_rebuilt_from_a_payload_refuses_to_run() -> None:
    rebuilt = Tool.from_dict(MCPTool(name="x", command=STDIO_COMMAND).to_dict())

    assert isinstance(rebuilt, MCPTool) and rebuilt.command is None
    with pytest.raises(ToolExecutionError, match="cannot run after being rebuilt"):
        await rebuilt.aexecute({})


def test_server_url_and_command_are_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        MCPTool(name="x", server_url="http://localhost/mcp", command=["srv"])


# --- without the extra ----------------------------------------------------------


def test_without_the_extra_the_error_names_the_install() -> None:
    """``import fastaiagent`` must not need ``mcp``; using MCPTool says what to
    install. ``sys.modules["mcp"] = None`` makes ``import mcp`` fail."""
    code = textwrap.dedent(
        """
        import asyncio, sys
        sys.modules["mcp"] = None
        import fastaiagent
        from fastaiagent.tool.mcp import MCPTool
        try:
            asyncio.run(MCPTool(name="x", server_url="http://localhost:1/mcp").aexecute({}))
        except Exception as e:
            print("call:", e)
        try:
            MCPTool.from_server("http://localhost:1/mcp")
        except ImportError as e:
            print("from_server:", e)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.count('pip install "fastaiagent[mcp]"') == 2, out
