# Code-first nodes (`@node`, typed I/O)

Beyond agent and tool nodes, you can write a node as a plain Python function and
give it typed inputs/outputs. Everything here is **additive** — chains that
don't use it behave exactly as before.

## `@node`

Decorate a function and add it to a chain. Its type hints become the node's
input schema, validated at the node boundary:

```python
from fastaiagent import Chain, node

@node(output_key="category")
def classify(text: str) -> str:
    return "support" if "help" in text else "sales"

chain = Chain("router")
chain.add_node("classify", node=classify, input_mapping={"text": "{{state.input}}"})
chain.execute({"input": "I need help"})   # -> state["category"] == "support"
```

## `output_key`

By default a node's non-dict output is stored under `_<node_id>_output`, and a
dict output is merged into state. Pass `output_key` to store the node's output
under a name you choose — clearer and collision-free:

```python
chain.add_node(
    "dbl", tool=double_tool, type=NodeType.tool,
    input_mapping={"x": "{{state.n}}"}, output_key="doubled",
)
# state["doubled"] holds the tool's return value
```

`output_key` works on any node type, not just `@node` ones.

## `input_schema` / `output_schema`

Attach optional JSON schemas to validate a node's resolved inputs and its output
at the boundary. A violation raises `ChainError` naming the offending field:

```python
@node(
    output_key="user",
    output_schema={
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    },
)
def make_user(name: str) -> dict:
    return {"id": f"u-{name}"}
```

`@node` derives `input_schema` from the function's type hints automatically (pass
`validate_input=False` to skip it). You can also set `input_schema=` /
`output_schema=` / `output_key=` directly on `add_node` for any node.

## What's intentionally out

This is a tight slice. Sub-DAGs / composite nodes, multi-node transactions, and a
unified Chain / Swarm / Supervisor node API are out of scope by design — those
three remain separate models.

See `examples/72_node_framework.py`.
