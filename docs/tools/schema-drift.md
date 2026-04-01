# Schema Drift Detection

Detect when tool responses no longer match their declared schema — catches API changes before they break your agents.

## Validating a Single Response

```python
from fastaiagent.tool.schema import validate_schema, detect_drift

# Validate a single response
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
    },
    "required": ["name", "price"],
}

violations = validate_schema(schema, {"name": "Widget", "price": "not-a-number"})
for v in violations:
    print(f"{v.field}: {v.message}")
    # price: Expected number, got string
```

## Detecting Drift Across Multiple Responses

```python
# Detect drift across multiple responses
report = detect_drift("product_api", schema, [
    {"name": "A", "price": 10.0},
    {"name": "B", "price": 20.0},
    {"name": "C", "price": "free"},  # drift!
])

print(report.drift_detected)   # True
print(report.violations)        # 1 violation
print(report.summary)           # "Drift detected for 'product_api': 1 violations..."
```

## Error Handling

```python
from fastaiagent._internal.errors import SchemaDriftError

try:
    result = tool.execute({"query": "test"})
except SchemaDriftError as e:
    print(f"Schema drift detected: {e}")
```

---

## Next Steps

- [FunctionTool](function-tools.md) — Wrap Python functions as tools
- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [MCPTool](mcp-tools.md) — Connect to MCP servers
- [Tools Overview](index.md) — All tool types at a glance
