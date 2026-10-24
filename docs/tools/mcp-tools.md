# MCPTool

Use the tools of any [Model Context Protocol](https://modelcontextprotocol.io) server — a remote
one over HTTP, or a local one the SDK starts for you — as fastaiagent tools. `MCPTool` uses the
official `mcp` client library, so it works with any server built to the spec.

> **See also:** [Expose an Agent or Chain as an MCP Server](mcp-server.md) — the other direction:
> Claude Desktop, Cursor and other MCP hosts (including another fastaiagent agent) call your agent.

```bash
pip install "fastaiagent[mcp]"
```

## Load every tool a server offers

`MCPTool.from_server(...)` connects, lists the server's tools and returns one `MCPTool` per tool,
with its name, description and input schema filled in:

```python
from fastaiagent import Agent, LLMClient, MCPTool

# A remote server (streamable HTTP)
tools = MCPTool.from_server("https://mcp.example.com/mcp", auth_token="...")

# A local server, started as a subprocess (stdio)
tools = MCPTool.from_server(
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp/docs"]
)

agent = Agent(name="assistant", llm=LLMClient(provider="openai", model="gpt-5-mini"), tools=tools)
```

Inside async code, use `await MCPTool.afrom_server(...)`. Extra keyword arguments (`timeout`,
`max_retries`, `replay_class`, `deferred`, …) are applied to every tool.

A local server gets a minimal environment (the official client's default: `PATH`, `HOME` and a
few more); `env={...}` adds to it — for anything else it needs, such as an API key. Runnable:
[`examples/116_mcp_tool.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/116_mcp_tool.py).

## One tool by hand

```python
search = MCPTool(
    name="search",
    server_url="https://mcp.example.com/mcp",
    tool_name="search_files",          # the server's name, if different from `name`
    auth_token="...",                  # sent as Authorization: Bearer ...
    description="Search the codebase",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
result = await search.aexecute({"query": "authentication"})
```

Pass `server_url=` **or** `command=` (and optionally `env=`), not both.

## What a call does

Each call opens a session, runs the MCP handshake, calls the tool and closes the session. For a
local server that means one process start per call.

| Server reply | `ToolResult` |
|---|---|
| Text content | `output` = the text (several blocks joined by newlines) |
| Image content | `output` contains `[image: <mime type>]` |
| Embedded resource with text | its text |
| No content, structured content only | `output` = the structured content |
| `isError: true` | `error` = the text |
| Server unreachable, not MCP, or a refused address | raises `ToolExecutionError`; for a reply that is not MCP, the error includes what the client made of it (for example `Error parsing JSON response`) instead of the `mcp` library logging a traceback |

The handshake and `tools/list` give up after 30 seconds, so a URL that is not an MCP server fails
instead of hanging. A call waits up to the tool's `timeout`, or 300 seconds when none is set.

## Trust Layer

MCP tools are ordinary tools to the agent: `tool_call` and `tool_result` [guardrails](../guardrails/index.md),
governance and tracing apply to every call, and a `tool_call` guardrail that blocks stops the call
before anything reaches the server.

## Security

- **Addresses.** Every HTTP request — and every redirect hop — passes the SDK's SSRF check: cloud
  metadata and private networks are refused, loopback is allowed for local servers. Set
  `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1` to reach intranet servers. See [Security](../security.md).
- **What a payload carries.** `to_dict()` (the agent push, the runner payload, traces) carries only
  `server_url` and `tool_name`. `auth_token`, `command` and `env` are never serialized. A local
  (`command=`) tool rebuilt from a payload therefore has nothing to run and refuses with an error —
  a payload can never start a process.

> **Replay safety.** Pass `replay_class="read_only" | "idempotent" | "side_effecting"` to mark how
> [Agent Replay](../replay/index.md) treats this tool — default `side_effecting`, never auto-inferred.
> See [Replay safety](index.md#replay-safety-replay_class).

## Upgrading from 1.91

Before 1.92.0, `MCPTool` spoke a hand-written JSON-RPC dialect that official MCP servers refuse
(`406` / `400`); it worked only against servers written the same way.

- Install the extra: `pip install "fastaiagent[mcp]"` (`fastaiagent[mcp-server]` still works).
- `discover_tools()` is gone: use `MCPTool.from_server(...)`, which returns ready-to-use tools.
- New: local servers via `command=`.

---

## Next Steps

- [FunctionTool](function-tools.md) — Wrap Python functions as tools
- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Tools Overview](index.md) — All tool types at a glance
