# Chains

A Chain is a directed graph workflow where nodes execute agents, tools, or logic, and edges define the flow between them. Chains support cycles (retry loops), typed state, checkpointing with resume, human-in-the-loop approval, parallel execution, and conditional branching.

## Quick Start

```python
from fastaiagent import Agent, Chain, LLMClient

summarizer = Agent(
    name="summarizer",
    system_prompt="Summarize the input in one sentence.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)
translator = Agent(
    name="translator",
    system_prompt="Translate the input to French. Output only the French text.",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
)

chain = Chain("summarize-and-translate")
chain.add_node("summarize", agent=summarizer)
chain.add_node("translate", agent=translator)
chain.connect("summarize", "translate")

result = chain.execute({"message": "Python is a popular programming language"})
print(result.output)
print(result.node_results)     # {"summarize": {...}, "translate": {...}}
print(result.execution_id)     # UUID for checkpointing/resume
```

## Node Types

| Type | Purpose | Example |
|------|---------|---------|
| `agent` | Run an agent (default) | `chain.add_node("research", agent=my_agent)` |
| `tool` | Execute a tool directly | `chain.add_node("fetch", tool=my_tool)` |
| `condition` | Branch based on state | See [Conditional Branching](#conditional-branching) |
| `transformer` | Render a template | `chain.add_node("fmt", type=NodeType.transformer, template="Hello {{state.name}}")` |
| `parallel` | Run multiple agents concurrently | See [Parallel Execution](#parallel-execution) |
| `hitl` | Pause for human approval | See [Human-in-the-Loop](hitl.md) |
| `start` / `end` | Explicit entry/exit points | `chain.add_node("in", type=NodeType.start)` |

```python
from fastaiagent.chain import NodeType

chain.add_node("classify", agent=classifier_agent)
chain.add_node("transform", type=NodeType.transformer, template="Category: {{state.category}}")
chain.add_node("approve", type=NodeType.hitl)
```

## Connecting Nodes

### Simple Edges

```python
chain.connect("a", "b")  # a → b
chain.connect("b", "c")  # b → c
```

### Conditional Edges

Route flow based on state values:

```python
chain.connect("classify", "billing_agent", condition="category == billing")
chain.connect("classify", "tech_agent", condition="category == technical")
chain.connect("classify", "general_agent")  # default fallback
```

Condition expressions support: `==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`, `startswith`. Values are resolved from chain state using `{{path.to.value}}` templates.

## Typed State

Validate the chain state at every step using JSON Schema:

```python
chain = Chain(
    "typed-pipeline",
    state_schema={
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "category": {"type": "string"},
            "priority": {"type": "integer"},
        },
        "required": ["message"],
    },
)
chain.add_node("classify", agent=classifier)

# This works — message is present
result = chain.execute({"message": "My order is late", "priority": 1})

# This raises ChainStateValidationError — missing required field
result = chain.execute({"priority": 1})
```

State is validated:
1. Before execution starts (initial state)
2. After each node updates the state

### Accessing State in Nodes

Each node receives the current state in its context. Agent nodes receive the input as their prompt. Transformer nodes can template against the full state:

```python
chain.add_node(
    "format_output",
    type=NodeType.transformer,
    template="Customer {{state.name}} (priority: {{state.priority}}): {{node_results.classify.output}}",
)
```

## Parallel Execution

Run multiple agents concurrently within a single node:

```python
chain = Chain("parallel-pipeline")
chain.add_node("start", type=NodeType.start)
chain.add_node(
    "parallel_research",
    type=NodeType.parallel,
    agents=[researcher_1, researcher_2, researcher_3],  # Run all 3 in parallel
)
chain.add_node("merge", type=NodeType.transformer, template="Results: {{node_results.parallel_research.outputs}}")
chain.connect("start", "parallel_research")
chain.connect("parallel_research", "merge")
```

Parallel nodes use `asyncio.gather()` for concurrent execution.

## Conditional Branching

Route execution based on state or previous node output:

```python
chain = Chain("routing-pipeline")
chain.add_node("classify", agent=classifier_agent)
chain.add_node(
    "router",
    type=NodeType.condition,
    conditions=[
        {"expression": "category == billing", "handle": "billing"},
        {"expression": "category == technical", "handle": "technical"},
    ],
)
chain.add_node("billing_agent", agent=billing_agent)
chain.add_node("tech_agent", agent=tech_agent)

chain.connect("classify", "router")
chain.connect("router", "billing_agent", condition="category == billing")
chain.connect("router", "tech_agent", condition="category == technical")
```

## Chain Validation

Validate chain structure before execution:

```python
errors = chain.validate()
if errors:
    for e in errors:
        print(f"Error: {e}")
else:
    print("Chain is valid")
```

Checks for:
- Missing edge targets (referencing nonexistent nodes)
- Orphaned nodes (no incoming or outgoing edges)
- Cyclic edges without `max_iterations`

## Tool Node State Behavior

When a tool node executes, its return value is wrapped in `{"output": <return_value>, "error": <error_or_None>}` and merged into chain state. This means each successive tool node **overwrites** `state.output` with its own wrapped result.

If you need to thread a value across multiple tool nodes (e.g., a `seed_value` that step A produces and step C reads), put it on the **top-level state** via `initial_state` to `chain.execute()` or `modified_state` to `chain.resume()` — not as a return value from a tool node. Top-level state keys persist because nothing overwrites them.

```python
# This is fragile — step_c can't reliably read step_a's output
# because step_b's output overwrites state.output

# This is reliable — seed_value persists at the top level
result = chain.execute({"seed_value": "original", "message": "go"})
# In the tool node, read via input_mapping:
#   input_mapping={"seed": "{{state.seed_value}}"}
```

Agent nodes do not have this wrapping quirk — their output is stored under `_{node_id}_output` in state, preserving it across nodes.

## ChainResult

Every chain execution returns a `ChainResult`:

| Field | Type | Description |
|-------|------|-------------|
| `output` | `Any` | Final node's output |
| `final_state` | `dict` | Chain state after all nodes complete |
| `execution_id` | `str` | UUID for checkpointing and resume |
| `node_results` | `dict` | Map of node_id -> output for each executed node |

```python
result = chain.execute({"message": "Hello"})

# Access individual node outputs
for node_id, output in result.node_results.items():
    print(f"{node_id}: {output}")
```

## Serialization

Chains serialize to a ReactFlow-compatible JSON format (used by the platform visual editor):

```python
# Serialize
data = chain.to_dict()
# {
#   "name": "my-pipeline",
#   "nodes": [{"id": "a", "type": "agent", "label": "A", "position": {"x": 0, "y": 0}, ...}],
#   "edges": [{"source": "a", "target": "b", "is_cyclic": false, ...}],
#   "state_schema": {...}
# }

# Restore
chain = Chain.from_dict(data)
```

This canonical format can be used to serialize/deserialize chains for storage or transfer.

## Sync vs Async

```python
# Sync
result = chain.execute({"input": "data"})

# Async
result = await chain.aexecute({"input": "data"})

# Async resume
result = await chain.resume(execution_id="<id>")
```

## Error Handling

```python
from fastaiagent._internal.errors import (
    ChainError,                   # Base chain error
    ChainCycleError,              # Cycle exceeded max_iterations
    ChainCheckpointError,         # Checkpoint save/load failed
    ChainStateValidationError,    # State failed schema validation
)

try:
    result = chain.execute({"input": "data"})
except ChainCycleError as e:
    print(f"Cycle limit hit: {e}")
except ChainStateValidationError as e:
    print(f"Invalid state: {e}")
except ChainCheckpointError as e:
    print(f"Checkpoint error: {e}")
```

## Complete Example

A support pipeline with classification, conditional routing, retry loop, and approval:

```python
from fastaiagent import Agent, Chain, LLMClient
from fastaiagent.chain import NodeType

llm = LLMClient(provider="openai", model="gpt-4.1")

chain = Chain(
    "support-pipeline",
    state_schema={
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "category": {"type": "string"},
        },
        "required": ["message"],
    },
)

# Nodes
chain.add_node("classify", agent=Agent(name="classifier",
    system_prompt="Classify the support request. Set category.", llm=llm))
chain.add_node("research", agent=Agent(name="researcher",
    system_prompt="Research the issue thoroughly.", llm=llm))
chain.add_node("draft", agent=Agent(name="drafter",
    system_prompt="Draft a helpful response.", llm=llm))
chain.add_node("review", type=NodeType.hitl)
chain.add_node("send", agent=Agent(name="sender",
    system_prompt="Finalize and send the response.", llm=llm))

# Flow
chain.connect("classify", "research")
chain.connect("research", "draft")
chain.connect("draft", "review")
chain.connect("review", "send")

result = chain.execute(
    {"message": "My order hasn't arrived"},
    hitl_handler=lambda n, c, s: True,  # Auto-approve for demo
)
print(result.output)
```

---

## Next Steps

- [Cyclic Workflows](cyclic-workflows.md) — Retry loops, max iterations, and exit conditions
- [Checkpointing](checkpointing.md) — Save and resume chain execution
- [Human-in-the-Loop](hitl.md) — Pause chains for human approval
- [Agents](../agents/index.md) — Core agent documentation
- [Platform Sync](../platform/index.md) — Push chains to the platform visual editor
