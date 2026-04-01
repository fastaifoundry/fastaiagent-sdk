# RESTTool

Calls an HTTP API endpoint. No Python function needed — just configure the URL and method.

## Basic Usage

```python
from fastaiagent import RESTTool

weather_api = RESTTool(
    name="weather",
    description="Get weather forecast for a city",
    url="https://api.weather.example.com/v1/forecast",
    method="GET",
    headers={"X-API-Key": "my-key"},
    body_mapping="query_params",  # Send arguments as URL query parameters
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    },
)

# Executes: GET https://api.weather.example.com/v1/forecast?city=Paris&days=3
result = await weather_api.aexecute({"city": "Paris", "days": 3})
```

## Body Mapping Options

| Mode | Behavior | Use Case |
|------|----------|----------|
| `query_params` | Arguments sent as URL query parameters | GET requests |
| `json_body` | Arguments sent as JSON request body | POST/PUT requests |
| `path_params` | Arguments replace `{placeholders}` in URL | RESTful paths |

### Path Parameters Example

```python
order_api = RESTTool(
    name="get_order",
    url="https://api.example.com/orders/{order_id}",
    method="GET",
    body_mapping="path_params",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)
# Executes: GET https://api.example.com/orders/ORD-123
```

---

## Next Steps

- [FunctionTool](function-tools.md) — Wrap Python functions as tools
- [MCPTool](mcp-tools.md) — Connect to MCP servers
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Tools Overview](index.md) — All tool types at a glance
