# Agent Replay

Agent Replay lets you load any past execution trace, step through it, inspect each step's input/output/attributes, fork at any point, modify the prompt or input, and rerun from that point. This is the SDK's unique debugging feature — no other framework offers fork-and-rerun.

## Why Replay?

When an agent fails in production — hallucinates, calls the wrong tool, gives a bad answer — you need to understand **why** and test a fix **without re-running the entire pipeline**. Replay lets you:

1. Load the exact trace of the failure
2. Step through to find the problematic step
3. Fork at that step
4. Change the prompt, input, or state
5. Rerun from that point only
6. Compare the original vs fixed result

## Loading a Replay

### From Local Trace Storage

```python
from fastaiagent.trace.replay import Replay

# Load by trace ID (from fastaiagent traces list)
replay = Replay.load("b6acf1ef2c2779bbc2fcf80802ae0534")
```

### From a TraceData Object

```python
from fastaiagent.trace.storage import TraceStore

store = TraceStore()
trace_data = store.get_trace("b6acf1ef2c2779bbc2fcf80802ae0534")
replay = Replay(trace_data)
```

## Viewing the Summary

```python
print(replay.summary())
```

Output:
```
Trace: b6acf1ef2c2779bbc2fcf80802ae0534
Name: agent.support-bot
Status: OK
Spans: 4
Duration: 2025-01-15T10:30:00Z → 2025-01-15T10:30:02.5Z

Steps:
  [0] agent.run
  [1] llm.chat_completion
  [2] tool.search_docs
  [3] llm.chat_completion
```

## Stepping Through

### All Steps

```python
steps = replay.step_through()
for step in steps:
    print(f"Step {step.step}: {step.span_name}")
    if step.attributes:
        print(f"  Attributes: {step.attributes}")
```

### Specific Step

```python
steps = replay.steps()
print(f"Total steps: {len(steps)}")

# Inspect a specific step
step = replay.inspect(2)
print(f"Name: {step.span_name}")
print(f"Span ID: {step.span_id}")
print(f"Timestamp: {step.timestamp}")
print(f"Attributes: {step.attributes}")
```

## ReplayStep

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Step index (0-based) |
| `span_name` | `str` | What happened (e.g., "llm.chat_completion", "tool.search") |
| `span_id` | `str` | Unique span identifier |
| `input` | `dict` | Input to this step |
| `output` | `dict` | Output from this step |
| `attributes` | `dict` | OTel attributes (model, tokens, tool name, etc.) |
| `timestamp` | `str` | When this step executed |

## Forking and Rerunning

The core debugging workflow — fork at the problem step, fix, rerun:

```python
# Step 3 is where the LLM hallucinated
forked = replay.fork_at(step=3)

# Modify what you want to fix
forked.modify_prompt("Always cite the exact policy section number. Never guess.")
forked.modify_input({"query": "refund policy section 4.2"})

# Rerun from that point
result = forked.rerun()
print(f"New output: {result.new_output}")
print(f"Steps executed: {result.steps_executed}")
```

### Available Modifications

| Method | What it changes |
|--------|----------------|
| `modify_prompt(new_prompt)` | System prompt for the LLM call at the fork point |
| `modify_input(new_input)` | Input data passed to the step |
| `modify_config(**kwargs)` | Agent/LLM configuration (temperature, max_tokens, etc.) |
| `modify_state(new_state)` | Chain state (for chain replays) |

Methods return `self` for chaining:

```python
result = (
    replay.fork_at(step=3)
    .modify_prompt("Be more precise.")
    .modify_input({"context": "additional context"})
    .modify_config(temperature=0.2)
    .rerun()
)
```

## Comparing Results

After rerunning, compare the original and fixed execution:

```python
forked = replay.fork_at(step=3)
forked.modify_prompt("New instructions")
result = forked.rerun()

comparison = forked.compare(result)
print(f"Original steps: {len(comparison.original_steps)}")
print(f"Diverged at step: {comparison.diverged_at}")
```

### ComparisonResult

| Field | Type | Description |
|-------|------|-------------|
| `original_steps` | `list[ReplayStep]` | Steps from the original trace |
| `new_steps` | `list[ReplayStep]` | Steps from the rerun (if available) |
| `diverged_at` | `int \| None` | Step index where results diverged |

### ReplayResult

| Field | Type | Description |
|-------|------|-------------|
| `original_output` | `Any` | Output from the original execution |
| `new_output` | `Any` | Output from the rerun |
| `steps_executed` | `int` | Number of steps executed in the rerun |
| `trace_id` | `str \| None` | Trace ID of the original execution |

## CLI Commands

```bash
# Show replay steps for a trace
fastaiagent replay show <trace_id>

# Inspect a specific step
fastaiagent replay inspect <trace_id> 3
```

Example:
```bash
$ fastaiagent replay show b6acf1ef2c27

Trace: b6acf1ef2c2779bbc2fcf80802ae0534
Name: agent.support-bot
Status: OK
Spans: 4

Steps:
  [0] agent.run
  [1] llm.chat_completion
  [2] tool.search_docs
  [3] llm.chat_completion

$ fastaiagent replay inspect b6acf1ef2c27 2

Step 2: tool.search_docs
  Timestamp: 2025-01-15T10:30:01.300Z
  Attributes: {'fastai.tool.name': 'search_docs'}
```

## Debugging Workflow

A typical debugging session:

```python
from fastaiagent.trace import TraceStore
from fastaiagent.trace.replay import Replay

# 1. Find the failing trace
store = TraceStore()
traces = store.search("support-bot")
# Or use: fastaiagent traces list

# 2. Load and inspect
replay = Replay.load(traces[0].trace_id)
print(replay.summary())

# 3. Step through to find the problem
for step in replay.step_through():
    print(f"[{step.step}] {step.span_name}: {step.attributes}")
# Step 3: LLM hallucinated the refund policy ← found it

# 4. Fork and fix
forked = replay.fork_at(step=3)
forked.modify_prompt("Always cite exact policy section numbers from the docs.")

# 5. Rerun and verify
result = forked.rerun()
print(f"Fixed output: {result.new_output}")

# 6. Compare
comparison = forked.compare(result)
print(f"Diverged at step {comparison.diverged_at}")
```

## Error Handling

```python
from fastaiagent._internal.errors import ReplayError

try:
    step = replay.inspect(99)
except ReplayError as e:
    print(f"Error: {e}")  # "Step 99 out of range (0-3)"

try:
    forked = replay.fork_at(-1)
except ReplayError as e:
    print(f"Error: {e}")  # "Step -1 out of range (0-3)"
```
