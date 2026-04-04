# CLI Reference

The `fastaiagent` CLI provides commands for managing traces, replays, evaluations, prompts, and knowledge bases.

## Installation

The CLI is installed automatically with the SDK:

```bash
pip install fastaiagent
fastaiagent --help
```

## Commands

### `fastaiagent version`

Show the SDK version.

```bash
fastaiagent version
# fastaiagent 0.1.0a1
```

### `fastaiagent traces`

Manage locally stored traces.

```bash
# List recent traces
fastaiagent traces list
fastaiagent traces list --last 24h

# Export a trace as JSON
fastaiagent traces export <trace_id>
fastaiagent traces export <trace_id> --output trace.json

# Search traces
fastaiagent traces search "agent-name"
```

### `fastaiagent replay`

Interactive Agent Replay — step through, inspect, and fork traces.

```bash
# Start replay for a trace
fastaiagent replay <trace_id>

# Step through execution
fastaiagent replay <trace_id> --step-through

# Fork at a specific step
fastaiagent replay <trace_id> --fork-at 3
```

### `fastaiagent eval`

Run evaluations from the command line.

```bash
# Run evaluation
fastaiagent eval run --dataset test_cases.jsonl --scorer exact_match

# List available scorers
fastaiagent eval scorers
```

### `fastaiagent prompts`

Manage the local prompt registry.

```bash
# List all prompts
fastaiagent prompts list

# Show a specific prompt
fastaiagent prompts show <name>
fastaiagent prompts show <name> --version 2

# Set an alias
fastaiagent prompts alias <name> <version> <alias>
```

### `fastaiagent kb`

Manage local knowledge bases.

```bash
# Create/add to a KB
fastaiagent kb add <name> <file_or_directory>

# Search a KB
fastaiagent kb search <name> "query text"

# Show KB status
fastaiagent kb status <name>
```

## Environment Variables

| Variable | Default | Used By |
|----------|---------|---------|
| `FASTAIAGENT_API_KEY` | — | `fa.connect()` |
| `FASTAIAGENT_TARGET` | `https://app.fastaiagent.net` | `fa.connect()` |
| `OPENAI_API_KEY` | — | LLM calls (OpenAI provider) |
| `ANTHROPIC_API_KEY` | — | LLM calls (Anthropic provider) |

## Next Steps

- [Tracing Guide](../tracing/index.md) for understanding traces
- [Agent Replay Guide](../replay/index.md) for debugging workflows
- [Evaluation Guide](../evaluation/index.md) for running evals
