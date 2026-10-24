"""End-to-end quality gate — a live model uses tools from a real MCP server.

The server is built with the official ``mcp`` library (``tests/_mcp_servers.py``).
Until 1.92.0 this gate ran against a hand-written JSON-RPC stub that accepted the
dialect ``MCPTool`` spoke, while every official server refused it. The protocol
paths are covered offline in ``tests/test_mcp_tool.py``; this gate adds what only
a live model shows: the schemas ``MCPTool.from_server`` reads are good enough for
the model to pick the right tool and call it with the right arguments.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


def test_a_live_model_uses_tools_from_an_mcp_server(official_mcp_server: Any) -> None:
    require_env()
    from fastaiagent import Agent, LLMClient, MCPTool

    agent = Agent(
        name="mcp-gate",
        system_prompt="Use your tools for arithmetic. Reply with the number only.",
        llm=LLMClient(provider="openai", model="gpt-5-mini", reasoning="minimal"),
        tools=MCPTool.from_server(official_mcp_server.url),
    )

    result = agent.run("What is 1234 + 4321?", trace=False)

    assert official_mcp_server.calls == [{"tool": "add", "a": 1234, "b": 4321}]
    assert "5555" in result.output
