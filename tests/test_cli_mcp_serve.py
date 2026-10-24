"""``fastaiagent mcp serve`` end to end: the real CLI in a subprocess, driven by
the official ``mcp`` client over stdio.

Before 1.92.0 the command passed ``expose_tools=`` to every target, and
``Chain.as_mcp_server`` takes no such parameter, so serving any Chain died
with ``TypeError`` before the handshake.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

_CLI = "from fastaiagent.cli.main import app; app()"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "target.py"
    path.write_text(textwrap.dedent(body))
    return path


async def _list_and_call(
    spec: str, tool: str, arguments: dict, *flags: str
) -> tuple[set[str], str]:
    params = StdioServerParameters(
        command=sys.executable, args=["-c", _CLI, "mcp", "serve", spec, *flags]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            result = await session.call_tool(tool, arguments)
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return names, text


@pytest.mark.asyncio
async def test_mcp_serve_serves_a_chain(tmp_path: Path) -> None:
    target = _write(
        tmp_path,
        """
        from fastaiagent import Chain

        chain = Chain(name="echo-chain")
        chain.add_node("echo", type="transformer", template="CHAIN SAW: {{input.input}}")
        """,
    )
    names, text = await _list_and_call(f"{target}:chain", "echo_chain", {"input": "knock knock"})
    assert names == {"echo_chain"}
    assert "CHAIN SAW: knock knock" in text


@pytest.mark.asyncio
async def test_mcp_serve_exposes_an_agents_tools(tmp_path: Path) -> None:
    # Calling an inner tool needs no model, so the key is never used.
    target = _write(
        tmp_path,
        """
        from fastaiagent import Agent, LLMClient
        from fastaiagent.tool import FunctionTool

        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        agent = Agent(
            name="calc",
            llm=LLMClient(provider="openai", model="gpt-5-mini", api_key="sk-unused"),
            tools=[FunctionTool(name="add", fn=add)],
        )
        """,
    )
    names, text = await _list_and_call(f"{target}:agent", "add", {"a": 2, "b": 3}, "--expose-tools")
    assert names == {"calc", "add"}
    assert text.strip() == "5"
