# Expose an Agent or Chain as an MCP Server

Since 0.6.0, any `Agent` or `Chain` can be exposed as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server. Once exposed, any MCP-compatible runtime — Claude Desktop, Cursor, Continue, Zed, or another fastaiagent agent via `MCPTool` — can invoke it as a tool.

The complement to [MCP Tools](mcp-tools.md) (which lets fastaiagent **consume** MCP servers), this page covers the **serve** side.

## Install

```bash
pip install 'fastaiagent[mcp-server]'
```

The `mcp` Python package is an optional extra — importing `FastAIAgentMCPServer` before installing it raises a clear `ImportError` with install instructions.

## Quick start — expose an Agent over stdio

```python
# my_agent.py
from fastaiagent import Agent, LLMClient

agent = Agent(
    name="research_assistant",
    system_prompt="Research the user's question and summarize concisely.",
    llm=LLMClient(provider="openai", model="gpt-4o"),
)

if __name__ == "__main__":
    import asyncio
    asyncio.run(agent.as_mcp_server(transport="stdio").run())
```

Run it as a one-shot:

```bash
python my_agent.py
```

Or via the CLI:

```bash
fastaiagent mcp serve my_agent.py:agent
```

## Register with Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on your platform:

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "python",
      "args": ["/absolute/path/to/my_agent.py"]
    }
  }
}
```

Restart Claude Desktop. The `research_assistant` tool is now available — Claude will call it for research-shaped questions and show the output in the conversation.

## Register with Cursor / Continue / Zed

These MCP clients accept the same stdio command/args pattern. See the client's own MCP config docs; the important line is the `command` + `args` that runs your Python file.

## API

### `Agent.as_mcp_server(...)`

```python
agent.as_mcp_server(
    transport="stdio",
    expose_tools=False,
    expose_system_prompt=True,
    tool_name=None,
    tool_description=None,
) -> FastAIAgentMCPServer
```

| Parameter | Default | Description |
|---|---|---|
| `transport` | `"stdio"` | Only `"stdio"` ships in 0.6.0. `"sse"` / `"streamable-http"` are tracked as follow-ups. |
| `expose_tools` | `False` | If `True`, each of the agent's own tools is also listed as an individual MCP tool. Default keeps the surface to one primary tool. |
| `expose_system_prompt` | `True` | Expose the system prompt via MCP's Prompts mechanism so clients can introspect "what kind of agent is this?". |
| `tool_name` | `None` | Override the primary tool name. Default sanitizes `agent.name` to `[A-Za-z0-9_]`. |
| `tool_description` | `None` | Override the primary tool description. Default includes the first line of the system prompt. |

Returns a `FastAIAgentMCPServer` — call `await server.run()` (or use `asyncio.run(server.run())`) to start the stdio loop. The method blocks until stdin closes (which is how local MCP hosts signal shutdown).

### `Chain.as_mcp_server(...)`

Same shape as `Agent.as_mcp_server`, minus the `expose_tools` / `expose_system_prompt` flags (chains have neither concept).

```python
chain.as_mcp_server(transport="stdio", tool_name=None, tool_description=None)
```

The chain is exposed as a single MCP tool that takes `{"input": "..."}` and returns the chain's final output as text.

### CLI — `fastaiagent mcp serve`

```bash
fastaiagent mcp serve <target> [--transport stdio] [--expose-tools] [--name NAME]
```

- `<target>` is either a file path or a dotted module path, followed by `:attr_name`:
  - `path/to/my_agent.py:agent`
  - `mypkg.agents:research_bot`
- `--transport` defaults to `stdio`.
- `--expose-tools` turns on individual tool surfacing.
- `--name` overrides the primary tool name.

## What the MCP client sees

| MCP primitive | What it contains |
|---|---|
| **Tool** (one, or more with `expose_tools=True`) | The primary tool takes `{"input": "..."}` and returns the agent's final output as text. When `expose_tools=True`, each of the agent's own tools is also listed with its original JSON Schema. |
| **Prompt** (one, when `expose_system_prompt=True`) | Named `<tool>_system`, contains the agent's resolved system prompt. Useful for clients that show "what does this server do?" tooltips. |
| **Resources** | Not currently exposed. Tracked as a follow-up — map `LocalKB` namespaces to MCP resources. |

## Example — agent with KB, memory, and tools — as one MCP server

Everything composes:

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.agent import ComposableMemory, AgentMemory, SummaryBlock
from fastaiagent.kb import LocalKB

llm = LLMClient(provider="openai", model="gpt-4o")

kb = LocalKB(name="product-docs")
kb.add("docs/")

agent = Agent(
    name="support_bot",
    system_prompt="Answer product questions from the KB. Escalate if unsure.",
    llm=llm,
    tools=[kb.as_tool()],
    memory=ComposableMemory(
        blocks=[SummaryBlock(llm=llm, keep_last=10, summarize_every=5)],
        primary=AgentMemory(max_messages=20),
    ),
)

if __name__ == "__main__":
    import asyncio
    asyncio.run(agent.as_mcp_server(transport="stdio").run())
```

From Claude Desktop this looks like one tool called `support_bot`. Claude calls it, the agent runs its KB-search tool internally, summarizes with the memory block it has configured, and returns text. Claude never sees the internal complexity.

## Troubleshooting

- **`ImportError: mcp`** — you haven't installed the extra. `pip install 'fastaiagent[mcp-server]'`.
- **Claude Desktop doesn't list the server** — ensure the `command` path is absolute and `python` resolves to an interpreter that has `fastaiagent` installed. Use an absolute `python` path if you use multiple environments.
- **Agent output looks blank** — the primary tool requires a non-empty `input` argument. Empty input returns an explicit error string.
- **Need to see what the server will expose before shipping** — call `server.describe()` to get a plain dict summary of the tools and prompts, without starting the stdio loop.

## Not yet implemented (tracked for follow-up)

- `transport="sse"` and `transport="streamable-http"` — the MCP spec's two remote transports. Currently raise `NotImplementedError` on `run()`.
- MCP resources — mapping `LocalKB` namespaces to MCP resources so clients can browse them.
- Auth middleware on remote transports — will compose with [`AgentMiddleware`](../agents/middleware.md).

---

## Next Steps

- [MCP Tools](mcp-tools.md) — consume MCP servers from a fastaiagent agent
- [Tools Overview](index.md) — the rest of the tool system
- [Agents](../agents/index.md) — the agents you're exposing
