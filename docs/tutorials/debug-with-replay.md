# Debug a Production Failure with Agent Replay

Your agent failed in production. Here's how to find and fix the bug in 60 seconds.

## Pull the Failing Trace

```python
from fastaiagent import FastAI
from fastaiagent.trace import Replay

fa = FastAI(api_key="sk-...", project="customer-support")

# Pull the production trace
trace = fa.traces.pull("trace_abc123")
replay = Replay(trace)

print(replay.summary())
# Trace: trace_abc123 (FAILED at step 4)
# Step 1: classify        - OK   320ms
# Step 2: search-orders   - OK   200ms
# Step 3: analyze         - OK   500ms
# Step 4: generate-refund - FAIL "Refund amount exceeds limit"
```

## Inspect the Failing Step

```python
step = replay.inspect(step=4)
print(step.input)   # {"order_total": 500.00, "refund_pct": 1.5}  <- bug! 150% refund
print(step.prompt)   # "Calculate refund: {{order_total}} x {{refund_pct}}"
```

## Fork, Fix, and Rerun

```python
forked = replay.fork_at(step=3)
forked.modify_state({"refund_pct": 0.9})  # Fix: 90% refund, not 150%
result = forked.rerun()

print(result.output)  # "Refund of $450.00 processed successfully"

# Compare with original
diff = replay.compare(result)
print(diff.summary())
# Step 4: generate-refund - FIXED (was ERROR, now SUCCESS)
# Output changed: error -> "Refund of $450.00 processed"
```

## Push the Fix

```python
# Fix the agent's prompt to clamp refund percentage
updated_agent = fa.pull_agent("refund-processor")
updated_agent.system_prompt += "\nRefund percentage must be between 0 and 1.0."
fa.push(updated_agent)  # Updated on platform
```

**That's fork-and-rerun debugging. No other SDK has this.**

## Using Replay from the CLI

```bash
# List recent traces
fastaiagent traces list

# Start interactive replay
fastaiagent replay trace_abc123

# Step through
fastaiagent replay trace_abc123 --step-through
```

## Next Steps

- [Agent Replay Reference](../replay/index.md) for the complete API
- [Tracing Guide](../tracing/index.md) for setting up tracing
- [Evaluation Guide](../evaluation/index.md) to prevent regressions
