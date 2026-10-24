"""Example 116: use an MCP server's tools in an agent (1.92.0).

    MCPTool.from_server(server_url=...)   a remote server, over streamable HTTP
    MCPTool.from_server(command=[...])    a local server the SDK starts (stdio)

Both connect with the official ``mcp`` client, list the server's tools and return
one ``MCPTool`` per tool, with its name, description and input schema filled in.
Guardrails, governance and tracing apply to every call, as for any tool.

The server here is example 32 — a fastaiagent agent served over MCP — started as a
local subprocess. Any MCP server works the same way, for instance the official
filesystem server: ``command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.

Install:
    pip install 'fastaiagent[mcp,openai]'

Usage:
    zsh -lc 'python examples/116_mcp_tool.py'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastaiagent import Agent, LLMClient, MCPTool

SERVER = Path(__file__).with_name("32_mcp_expose_agent.py")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (example 32's server needs a model key too).")

    # A stdio server gets a minimal environment (PATH, HOME, ...); ``env`` adds to
    # it. Example 32 needs the key, so pass it on.
    tools = MCPTool.from_server(
        command=[sys.executable, str(SERVER)],
        env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
    )
    print("Tools on the server:", [t.name for t in tools])

    lookup = next(t for t in tools if t.name == "research_lookup")
    agent = Agent(
        name="curious",
        system_prompt="Answer with the cached fact from research_lookup. One sentence.",
        llm=LLMClient(provider="openai", model="gpt-5-mini", reasoning="minimal"),
        tools=[lookup],
    )
    result = agent.run("Tell me something about octopuses.")
    print("Tool calls:", [c["tool_name"] for c in result.tool_calls])
    print("Answer:", result.output)


if __name__ == "__main__":
    main()
