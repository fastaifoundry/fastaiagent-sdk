"""Example 32: Expose an Agent as an MCP server.

Run this file directly to start an MCP server on stdio. Register it with
Claude Desktop / Cursor / Continue / Zed via their MCP config, or drive it
from another fastaiagent process via ``MCPTool``.

Install:
    pip install 'fastaiagent[mcp-server]'

Run directly (expects Claude Desktop or another stdio MCP client to pipe
stdin/stdout):
    python examples/32_mcp_expose_agent.py

Claude Desktop config snippet
-----------------------------
    {
      "mcpServers": {
        "research-assistant": {
          "command": "python",
          "args": ["/absolute/path/to/examples/32_mcp_expose_agent.py"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import os
import sys

from fastaiagent import Agent, FunctionTool, LLMClient


def _pick_llm() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        return LLMClient(provider="openai", model="gpt-4o-mini")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
    print(
        "Set OPENAI_API_KEY or ANTHROPIC_API_KEY before starting the MCP server.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- A small search-like tool the agent can use ---------------------------


def lookup(topic: str) -> str:
    """A pretend research cache."""
    facts = {
        "octopus": "Octopuses have three hearts and eight arms with chromatophore skin.",
        "bridge": (
            "Modern suspension bridges use stranded steel cables in "
            "compression-resistant towers."
        ),
        "coffee": "Arabica beans contain about half the caffeine of robusta per gram.",
    }
    return facts.get(topic.lower(), f"No cached fact for {topic!r}.")


research_tool = FunctionTool(
    name="research_lookup",
    fn=lookup,
    description="Look up a cached fact about a topic.",
    parameters={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
)


# --- Build the agent -------------------------------------------------------


agent = Agent(
    name="research_assistant",
    system_prompt=(
        "You are a concise research assistant. When the user asks about a "
        "topic you recognize, call research_lookup(topic) to get a cached "
        "fact and include it in your answer. Always be brief — one or two "
        "sentences."
    ),
    llm=_pick_llm(),
    tools=[research_tool],
)


async def main() -> None:
    # expose_tools=True lets MCP clients call research_lookup directly too;
    # omit or set False to keep the surface to a single primary tool.
    server = agent.as_mcp_server(transport="stdio", expose_tools=True)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
