# MCPTool

Connects to a Model Context Protocol (MCP) server via JSON-RPC 2.0.

## Basic Usage

```python
from fastaiagent import MCPTool

file_search = MCPTool(
    name="file_search",
    description="Search files in the codebase",
    server_url="http://localhost:3000/mcp",
    tool_name="search_files",
    auth_token="my-token",  # Optional Bearer token
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "file_pattern": {"type": "string"},
        },
        "required": ["query"],
    },
)

result = await file_search.aexecute({"query": "authentication", "file_pattern": "*.py"})
```

## Discovering MCP Tools

List available tools on an MCP server:

```python
tools = await file_search.discover_tools()
for t in tools:
    print(f"{t['name']}: {t.get('description', '')}")
```

---

## Next Steps

- [FunctionTool](function-tools.md) — Wrap Python functions as tools
- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Tools Overview](index.md) — All tool types at a glance
