# Concepts & Mental Model

This page is the mental model for agents — *why* they exist, *when* to reach
for one (versus a Chain, Swarm, or Supervisor), *how* the run loop actually
executes, how the composable layers stack inside that loop, and how a run is
traced and debugged. Read it first, then use the feature pages ([Tools](tools.md),
[Memory](memory.md), [Middleware](middleware.md), [Durability](durability.md),
[Multi-Agent Teams](teams.md), [Swarm](swarm.md)) for depth.

## Why agents exist

An LLM on its own can only produce text. To *do* something — look up an order,
call an API, check a policy, then answer — it needs a loop: call the model, let
it request an action, run the action, feed the result back, and repeat until
it's ready to answer.

An **Agent** is that loop, packaged. It wraps an LLM with **tools** (actions it
can take), **guardrails** (validation on the way in and out), **memory**
(context across turns and sessions), **middleware** (cross-cutting hooks), and
**dynamic instructions** (a system prompt that adapts per request) — and runs
the whole thing as one traced, optionally durable unit.

The defining trait: an agent decides its own path at runtime. You give it a
goal and the tools; the model chooses which tools to call and when to stop.
That is exactly what you want for open-ended tasks — and exactly what you *don't*
want when the path must be fixed and reproducible (that's a [Chain](../chains/concepts.md)).

## When to use an Agent

The SDK has four ways to run work. The dividing line: an **Agent** lets the
model decide the path; a **Chain** fixes the path in a graph you draw; **Swarm**
and **Supervisor** compose *multiple* agents with different control shapes.

| Use | When | Control |
|-----|------|---------|
| **Agent** | One goal, open-ended path — let the tool-calling loop figure out the steps. | Model decides at runtime |
| **Chain** | The steps are known and must be deterministic — routing, retry loops, HITL gates, one unified trace. | You draw the graph |
| **Swarm** | Several specialists that hand off to each other; the *active* agent decides who goes next. No coordinator. | Peer-to-peer mesh |
| **Supervisor** | A central agent delegates to worker agents and synthesizes their outputs. | Hub-and-spoke |

Rule of thumb: start with a single **Agent**. Reach for a **Chain** when you
need deterministic structure, a **Swarm** when routing belongs to the
specialists, and a **Supervisor** when one LLM should orchestrate and combine
workers.

## The run loop

Calling `agent.run(...)` / `await agent.arun(...)` executes this sequence
(verified against `fastaiagent/agent/agent.py` and `agent/executor.py`):

1. **Open the `agent.<name>` span** — the root of the trace for this run.
2. **Resolve instructions** — if `system_prompt` is a callable
   ([dynamic instructions](dynamic-instructions.md)), call it with the
   `RunContext` to get this request's system prompt.
3. **Build the message list** — system prompt, then **memory context** (a
   `memory.read` span pulls prior turns / retrieved blocks), then the user input.
4. **Input guardrails** — run every `GuardrailPosition.input` guardrail on the
   input before the model ever sees it. A blocking guardrail stops the run here.
5. **Enter the tool-calling loop** — `for iteration in range(max_iterations)`
   (default `max_iterations=10`). Each iteration:
   - *(if durable)* write a **turn-boundary checkpoint**.
   - **`before_model`** middleware runs (may `StopAgent`).
   - **Call the LLM** — one `llm.<provider>.<model>` span.
   - **`after_model`** middleware runs.
   - **No tool calls? Return** — this is the normal exit: the model produced a
     final answer.
   - **Tool calls?** For each one: governance gate → `wrap_tool` middleware →
     the tool runs inside a **`tool.<name>` span**, with a
     `GuardrailPosition.tool_call` guardrail on the arguments *before* and a
     `GuardrailPosition.tool_result` guardrail on the output *after*. Results are
     appended to the messages and the loop continues.
6. **Output guardrails** — run every `GuardrailPosition.output` guardrail on the
   final answer.
7. **Write to memory** — a `memory.write` span records the user message and the
   assistant reply for future turns.

!!! info "Verified against a live run"
    Running an agent with two tools and input+output guardrails, the observed
    order was: **input guardrail → both tools execute in a single turn →
    output guardrail**. Multiple tool calls the model requests in one turn run
    within that same iteration (and can run concurrently — see
    [parallel tools](tools.md)); the loop advances to the next iteration only
    when the model needs another round. The run stops as soon as the model
    replies with no tool calls, or when `max_iterations` is hit
    (`MaxIterationsError`).

```
agent.arun(input)
  │
  ├─ resolve system prompt (dynamic instructions)
  ├─ memory.read → build messages
  ├─ input guardrails
  │
  ├─ LOOP (max_iterations):
  │    before_model ─▶ LLM call ─▶ after_model
  │        │
  │        ├─ no tool calls ──────────────▶ break (final answer)
  │        └─ tool calls: [tool_call GR ─▶ tool ─▶ tool_result GR] ×N ─▶ next iteration
  │
  ├─ output guardrails
  └─ memory.write ─▶ AgentResult(output, tool_calls, tokens, cost, trace_id)
```

## The composable layers

The power of the model is that each concern is a layer that snaps onto the same
loop without you rewriting it. Where each one acts:

| Layer | Where it acts in the loop | Page |
|-------|---------------------------|------|
| **Dynamic instructions** | Step 2 — computes the system prompt per request | [dynamic-instructions.md](dynamic-instructions.md) |
| **Memory** | Step 3 (read) and step 7 (write) | [memory.md](memory.md) |
| **Guardrails** | Steps 4, 5 (tool_call/tool_result), and 6 — four positions | [../guardrails/index.md](../guardrails/index.md) |
| **Tools** | Inside the loop, each in a `tool.<name>` span | [tools.md](tools.md) |
| **Middleware** | `before_model` / `after_model` each iteration, `wrap_tool` around each tool | [middleware.md](middleware.md) |
| **Durability** | Turn and pre-tool checkpoints; `interrupt()` suspends the loop | [durability.md](durability.md) |

Guardrails have **four positions** — `input`, `tool_call`, `tool_result`,
`output` — so you can validate at every boundary the loop crosses, not just the
final answer.

## Composing multiple agents

A single agent is one loop. When one loop isn't enough, agents compose — and
the composition itself is just an agent wrapping other agents:

| Shape | How it works | Go deeper |
|-------|--------------|-----------|
| **Single agent** | One tool-calling loop. | This page |
| **Supervisor / Worker** | A central agent treats each worker as a callable; it delegates, collects outputs, and synthesizes one answer. Hub-and-spoke. | [teams.md](teams.md) |
| **Swarm** | A mesh of peers; the active agent is given `handoff_to_<peer>` tools and transfers control itself. No coordinator. | [swarm.md](swarm.md) |

Choosing between them is the routing question: **should a central LLM decide who
does what (Supervisor), or should each specialist decide when to hand off
(Swarm)?** If instead the flow should be *fixed and deterministic* — explicit
routing, retry loops, human-approval gates — that's a [Chain](../chains/concepts.md),
not a multi-agent topology.

## How an agent is traced & debugged

Every run is one OpenTelemetry trace rooted at the `agent.<name>` span
(`runner.type` defaults to `agent`). LLM calls, tools, and memory reads/writes
nest underneath it as child spans, so a full run reads as a single tree:

```
agent.weather-probe               ← root span
├── memory.read
├── llm.openai.gpt-4.1            ← turn 1: model asks for tools
├── tool.get_weather             ← Paris
├── tool.get_weather             ← Tokyo
├── llm.openai.gpt-4.1            ← turn 2: model writes the answer
└── memory.write
```

The `AgentResult` also carries run-level signals for debugging without opening
the trace: `output`, `tool_calls` (each with its `iteration`, name, and args),
`tokens_used`, `cost`, `latency_ms`, `trace_id`, and — when a checkpointer is
attached — `execution_id` and `status` (`"completed"` or `"paused"`).

- **Traces** are stored in `local.db` and shown in the Local UI, one card per
  run, expandable into the span tree. See [Tracing](../tracing/index.md).
- **Guardrail events** (which fired, blocked or passed) surface as their own UI
  view — see [Guardrails](../guardrails/index.md).
- **Durable runs** checkpoint every turn and every tool call, so you can inspect
  or resume a run step by step. See [Durability](durability.md).
- **Replay** re-runs a standalone agent trace to reproduce a run — see [Replay](../replay/index.md).

## A guided learning path

Work through these runnable examples in order — each adds one capability:

1. [`examples/01_simple_agent.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/01_simple_agent.py) — the bare loop: an agent that answers.
2. [`examples/41_agent_tools.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/41_agent_tools.py) — give it tools and watch the tool-calling loop.
3. [`examples/03_guardrails.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/03_guardrails.py) — validate input and output.
4. [`examples/30_memory_blocks.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/30_memory_blocks.py) — add memory across turns.
5. [`examples/18_supervisor_worker.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/18_supervisor_worker.py) and [`examples/31_swarm_research_team.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/31_swarm_research_team.py) — compose multiple agents.

## Next steps

- [Agents](index.md) — the how-to reference for constructing and running agents
- [Tools](tools.md) — attach actions and control the tool-calling loop
- [Memory](memory.md) — context within a conversation and across sessions
- [Guardrails](../guardrails/index.md) — validate at all four positions
- [Middleware](middleware.md) — cross-cutting hooks around the loop
- [Durability](durability.md) — checkpoint, resume, and interrupt a run
- [Multi-Agent Teams](teams.md) and [Swarm](swarm.md) — compose agents
- [Chains](../chains/concepts.md) — when you need a deterministic graph instead
