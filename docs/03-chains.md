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
| `hitl` | Pause for human approval | See [Human-in-the-Loop](#human-in-the-loop) |
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

### Cyclic Edges (Retry Loops)

Create loops that retry until a condition is met or a max iteration count is reached:

```python
chain.connect("research", "evaluate")
chain.connect(
    "evaluate", "research",
    max_iterations=3,                    # Max 3 retries
    exit_condition="quality >= 0.8",     # Exit loop when quality is high enough
)
chain.connect("evaluate", "respond", condition="quality >= 0.8")
```

**Cycle configuration:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iterations` | Upper bound on loop count | Required for cyclic edges |
| `exit_condition` | Expression to exit early | None (runs until max) |

When `max_iterations` is exceeded, a `ChainCycleError` is raised.

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

## Checkpointing

Chains automatically checkpoint state after each node. If execution fails, you can resume from the last successful checkpoint.

```python
from fastaiagent.chain.checkpoint import CheckpointStore

# Enable checkpointing (on by default)
chain = Chain("my-pipeline")
chain.add_node("step1", agent=agent1)
chain.add_node("step2", agent=agent2)
chain.add_node("step3", agent=agent3)
chain.connect("step1", "step2")
chain.connect("step2", "step3")

# First run — step2 might fail
try:
    result = chain.execute({"input": "data"})
except Exception as e:
    print(f"Failed at: {e}")
    print(f"Execution ID: {result.execution_id}")  # Save this
```

### Resuming from Checkpoint

```python
# Resume from where it failed
result = await chain.resume(
    execution_id="<saved-execution-id>",
    modified_state={"retry_count": 1},  # Optional: modify state before resuming
)
```

### Custom Checkpoint Store

```python
store = CheckpointStore(db_path="/path/to/checkpoints.db")
chain = Chain("my-pipeline", checkpoint_store=store)
```

### Inspecting Checkpoints

```python
store = CheckpointStore()
checkpoints = store.load(execution_id="<id>")
for cp in checkpoints:
    print(f"Node: {cp.node_id}, State: {cp.state_snapshot}")

latest = store.get_latest(execution_id="<id>")
print(f"Last completed: {latest.node_id}")
```

### Disabling Checkpointing

For lightweight chains that don't need persistence:

```python
chain = Chain("quick-pipeline", checkpoint_enabled=False)
```

## Human-in-the-Loop

HITL nodes pause execution for human approval:

```python
chain = Chain("approval-pipeline")
chain.add_node("draft", agent=drafter_agent)
chain.add_node("review", type=NodeType.hitl)
chain.add_node("send", agent=sender_agent)
chain.connect("draft", "review")
chain.connect("review", "send")

# With a custom approval handler
def approval_handler(node, context, state):
    draft = context["node_results"]["draft"]
    print(f"Review this draft: {draft}")
    return input("Approve? (y/n): ").lower() == "y"

result = chain.execute({"message": "Write a response"}, hitl_handler=approval_handler)
```

If no handler is provided, HITL nodes auto-approve (useful for testing).

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

## ChainResult

Every chain execution returns a `ChainResult`:

| Field | Type | Description |
|-------|------|-------------|
| `output` | `Any` | Final node's output |
| `final_state` | `dict` | Chain state after all nodes complete |
| `execution_id` | `str` | UUID for checkpointing and resume |
| `node_results` | `dict` | Map of node_id → output for each executed node |

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

This is the format used when pushing to the platform with `fa.push(chain)`. Chains pushed to the platform appear in the visual editor.

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
