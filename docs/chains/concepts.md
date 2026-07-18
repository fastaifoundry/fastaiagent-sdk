# Concepts & Mental Model

This page is the mental model for chains — *why* they exist, *when* to reach
for one, *how* the executor actually runs your graph, and *how* to trace and
debug a run. Read it first, then use the feature pages ([Execution
Spec](spec.md), [Cyclic Workflows](cyclic-workflows.md),
[Checkpointing](checkpointing.md), [Human-in-the-Loop](hitl.md)) for depth.

## Why chains exist

A single [Agent](../agents/index.md) is one tool-calling loop: you give it a
goal and it decides, step by step, which tools to call until it's done. That is
exactly what you want when the path is open-ended and you trust the model to
find it.

Real workflows often need the opposite — *explicit, deterministic structure*
that you control:

- **Routing** — send billing questions to the billing agent and technical ones
  to the tech agent, deterministically, not by hoping the model picks right.
- **Retry-until-quality loops** — research, evaluate, and re-research until an
  output crosses a quality bar.
- **Human approval gates** — pause before sending an email or charging a card,
  wait minutes or days for a human, then resume.
- **Durable, resumable runs** — survive a crash and continue from the last
  completed step instead of restarting.
- **One observable trace** — see the whole multi-agent run as a single tree,
  not a pile of disconnected agent traces.

A **Chain** is the orchestration layer that provides all of this. You draw a
directed graph — nodes do work, edges define the flow — and the executor runs
it with typed shared state, checkpointing, and unified tracing.

## When to use a Chain

The SDK has four ways to run more than one step. The dividing line: with a
Chain **you draw the graph**; with the others, **flow is decided at runtime**.

| Use | When |
|-----|------|
| **Agent** | One goal; let the tool-calling loop figure out the steps. No fixed structure. |
| **Chain** | You want an explicit, deterministic graph — routing, cycles, HITL gates, checkpointing, and a single unified trace. |
| **Swarm** | Peer agents hand off to each other dynamically; the path emerges at runtime. |
| **Supervisor** | A manager agent delegates to worker agents in a hierarchy. |

Reach for a Chain when the *shape* of the work is known and you want it to be
reproducible and observable — even if each node is itself an autonomous agent.

## The working model

A chain run is a loop over your graph. The executor:

1. **Orders the nodes** with a topological sort over the non-cyclic edges, so a
   node runs only after the nodes feeding it have run.
2. For each node in turn:
   - **Reads** the current chain state.
   - **Executes** the node by its type — an `agent`, a `tool`, a `condition`
     branch, a `parallel` fan-out, a `hitl` pause, or a `transformer` template.
   - **Merges** the node's output into state.
   - **Validates** the new state (if a `state_schema` is set).
   - **Routes** — picks the outgoing edge(s) to activate (see below).
   - **Checkpoints** the state snapshot and node output.

The thing that ties it all together is a single, shared, typed **`ChainState`**
that flows through every node. Each node reads it and writes back into it, so
later nodes and edge conditions see what earlier nodes produced.

!!! info "Two output-storage rules worth knowing"
    Agent-node output is stored under `_{node_id}_output`, so it persists
    across nodes. A **tool** node's return value is wrapped as
    `{"output": ..., "error": ...}` and each tool node overwrites
    `state.output` with its own result — to thread a value across several tool
    nodes, put it on top-level state via `initial_state`, not as a tool return.
    See [Tool Node State Behavior](index.md#tool-node-state-behavior).

!!! info "The authoritative contract"
    This section is the intuition. For the exact, test-backed rules —
    routing precedence, parallel failure modes, cycle accounting, the resume
    contract, and validation — see the [Execution Spec](spec.md).

### How routing actually decides

After a node runs, the executor rebuilds a **context** — `{input, state,
node_results}` — reflecting what that node just wrote, then walks the node's
outgoing edges *in declaration order*. Each edge's `condition` is a small
templated expression: placeholders like `{{state.category}}` or
`{{node_results.classify.output}}` are resolved against that context, then
compared with an operator (`==`, `!=`, `<`, `contains`, `startswith`, …). The
**first** conditional edge that evaluates true wins; a single unconditional
sibling acts as the default fallback. If every edge is unconditional, they *all*
fire (fan-out). This is why the graph can branch on live data without you
writing any dispatch code — the condition strings *are* the router, evaluated
against state the previous node produced.

## Topologies at a glance

A chain is a general directed graph (with cycles), so the same primitives —
`add_node` and `connect` — compose into every common shape:

| Topology | How | Go deeper |
|----------|-----|-----------|
| **Sequential** | `connect(a, b)` — linear hand-off | [index.md](index.md#connecting-nodes) |
| **Fan-out** | Multiple unconditional edges from one node — all fire | [spec.md](spec.md#routing) |
| **Conditional routing** | Edges carry `condition=`; first match wins, an unconditional sibling is the default | [index.md](index.md#conditional-edges) |
| **Condition-node handles** | A `NodeType.condition` returns a `handle`; the edge whose `label` matches fires | [index.md](index.md#conditional-branching) |
| **Parallel** | A `NodeType.parallel` node runs several agents concurrently via `asyncio.gather` | [index.md](index.md#parallel-execution) |
| **Cyclic loop** | A `connect(..., max_iterations=N, exit_condition="...")` edge repeats until the condition holds or the cap is hit | [cyclic-workflows.md](cyclic-workflows.md) |
| **DAG** | Any acyclic combination of the above; the topological sort handles converging/diverging paths | [spec.md](spec.md) |

A support pipeline uses several of these at once — conditional routing, a
retry loop, and a human gate:

```
             ┌─────────────────────────────┐
             │        (retry loop)          │
             ▼                              │
  classify ──▶ research ──▶ evaluate ──────┘  exit_condition: quality >= 0.8
     │                          │
     │ (conditional routing)    ▼ (once quality passes)
     ├──▶ billing_agent       draft ──▶ review (HITL) ──▶ send
     └──▶ tech_agent
```

## How a chain is traced & debugged

### One trace, not N

When you run with `trace=True` (the default), the chain opens a single
`chain.<name>` OpenTelemetry root span and every child agent, tool, and LLM
span nests underneath it. The root span carries
`fastaiagent.runner.type="chain"`, so a three-agent chain shows up as **one**
trace with a Gantt-style tree — not three orphan agent traces:

```
chain.support-pipeline           ← root span, runner.type="chain"
├── agent.classify
│   └── llm.openai.gpt-4.1
├── agent.research
│   ├── llm.openai.gpt-4.1
│   └── tool.search_docs
└── agent.send
    └── llm.anthropic.claude-sonnet-4-6
```

Pass `trace=False` to skip the `chain.<name>` root span when the chain runs as
a sub-step of another already-traced workflow — the child agent/tool spans
still emit, they just don't nest under a chain parent.

### Checkpoints are the step-inspection surface

There is no verbose mode; the way you inspect a run step by step is through
checkpoints. Every node writes a checkpoint containing the `state_snapshot`,
the `node_output`, and cycle counters, so you can replay the *state* of the run
without re-executing it:

```python
from fastaiagent.checkpointers import SQLiteCheckpointer

cp = SQLiteCheckpointer()
for c in cp.list(result.execution_id):
    print(c.node_id, c.status, c.state_snapshot)
```

See [Checkpointing](checkpointing.md) for the full inspection API and custom
backends.

### In the Local UI

Traces are stored in `local.db` and read over REST — the **Traces** tab shows
each chain as one card (chain name, span count, duration) and expands into the
span tree. There is also a read-only **workflow topology** view that renders
the graph (nodes + edges) from the chain definition. The Local UI is for
inspection, not editing.

!!! warning "Chain traces don't replay"
    The trace/replay surface reconstructs and re-runs a standalone **agent**
    trace — it does not replay chains. To re-run a chain from a saved step with
    changed inputs, use the checkpoint primitives
    `Chain.aresume(execution_id, modified_state=...)` (continue the same run)
    or `Chain.afork(execution_id, checkpoint_id=..., modified_state=...)`
    (branch a new run, leaving the original intact) — not the Replay engine.

### When something goes wrong

Chain failures surface as a small, catchable hierarchy (all subclass
`ChainError`):

| Error | Raised when |
|-------|-------------|
| `ChainCycleError` | A cyclic edge exceeds `max_iterations` |
| `ChainRoutingError` | `strict_routing=True` and no edge matched |
| `ChainStateValidationError` | State fails the `state_schema` |
| `ChainCheckpointError` | A checkpoint save/load fails |
| `ChainResumeError` | A resume is invalid (e.g. an interrupted run resumed without a `Resume(...)`); subclasses `ChainCheckpointError` |

## A guided learning path

Work through these runnable examples in order — each adds one capability:

1. [`examples/36_chain_workflow.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/36_chain_workflow.py) — your first chain: two agents, one unified trace.
2. [`examples/02_chain_with_cycles.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/02_chain_with_cycles.py) — add a retry loop with `max_iterations` and an `exit_condition`.
3. [`examples/47_workflow_topology.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/47_workflow_topology.py) — conditional routing plus a HITL gate, rendered in the topology view.
4. [`examples/42_durability_hitl.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/42_durability_hitl.py) — suspend/resume across a process restart with idempotent side effects.

## Next steps

- [Chains](index.md) — the how-to reference for every node type, edge, and option
- [Execution Spec](spec.md) — the authoritative, test-backed contract
- [Code-first nodes](typed-nodes.md) — write nodes as typed Python functions with `@node`
- [Cyclic Workflows](cyclic-workflows.md) — retry loops in depth
- [Checkpointing](checkpointing.md) — save, resume, and fork
- [Human-in-the-Loop](hitl.md) — suspending and blocking approval gates
- [Idempotency](idempotency.md) — make side effects safe across resume
