# Agent Durability

Pass a `checkpointer` to `Agent` and the tool-calling loop becomes durable:
each turn and each tool call write a checkpoint, and any tool can suspend
the agent for human approval via [`interrupt()`](../chains/hitl.md).

```python
from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, interrupt

def approve_refund(amount: int):
    if amount > 10_000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": amount},
        )
        return {"approved": decision.approved, "approver": decision.metadata.get("approver")}
    return {"approved": True, "auto": True}

agent = Agent(
    name="refund-agent",
    system_prompt="...",
    llm=...,
    tools=[FunctionTool(name="approve_refund", fn=approve_refund)],
    checkpointer=SQLiteCheckpointer(),
)

result = agent.run("Refund $50000", execution_id="refund-abc")
# result.status == "paused", result.pending_interrupt is populated.

# Sometime later (different process is fine):
from fastaiagent import Resume
result = await agent.aresume(
    "refund-abc",
    resume_value=Resume(approved=True, metadata={"approver": "alice"}),
)
# result.status == "completed", result.output is the final LLM response.
```

## What gets checkpointed

For each iteration of the tool loop the agent writes:

| Checkpoint | When | `node_id` | Resume semantics |
|---|---|---|---|
| **Turn boundary** | Before each LLM call | `turn:N` | Re-issue the LLM call at turn N. |
| **Pre-tool boundary** | Before each tool runs | `turn:N/tool:X` | Re-invoke the saved tool with saved args; **no LLM re-call**. |
| **Interrupted** | A tool called `interrupt()` | `turn:N/tool:X` (status=`interrupted`) | Re-invoke the tool with `_resume_value` in scope. |

`agent_path` on every checkpoint is `agent:<name>` (or
`agent:<name>/tool:<tool>` for tool-scoped checkpoints), so multi-agent
topologies in later phases can extend the prefix to
`supervisor:<s>/worker:<w>/agent:<a>/...`.

## Resume shapes

### Interrupted run — `interrupt()` was called

```python
result = agent.resume(execution_id, resume_value=Resume(approved=True))
```

The pending row is **atomically claimed** before the tool re-runs.
Concurrent resumers (or a double-clicked Approve button) see
[`AlreadyResumed`](../chains/hitl.md#atomic-resume-claim). The suspended
tool runs once more, `interrupt()` returns the resume value, and the loop
continues.

### Tool-boundary crash — process died mid-tool

If the process is killed (SIGKILL, OOM, container restart) while a tool is
running, the latest checkpoint is the pre-tool boundary. A plain
`agent.resume(execution_id)` re-invokes the saved tool with the saved
args, then continues the loop. The LLM is **not** re-called — the
assistant's tool_calls are already in the saved message history.

### Turn-boundary crash — process died during the LLM call

If the process is killed while waiting on the LLM, the latest checkpoint is
the turn boundary. `agent.resume(execution_id)` re-enters the loop at that
turn and re-issues the LLM call.

> **Cost note:** turn-boundary resume re-issues the LLM call. Same
> trade-off as Chain nodes. Wrap side-effectful tool functions with
> [`@idempotent`](../chains/idempotency.md) so a re-run does not re-fire
> the side effect.

## Constraint: tools must match between run and resume

The resume path looks up the suspended tool by name on `self.tools`. If
you build a fresh Agent for resume, it must register the same tools you
used at run time — otherwise `agent.resume(...)` raises
`ChainCheckpointError` with a message naming the missing tool.

## Swarms

Pass a `checkpointer` to `Swarm` and every handoff becomes a checkpoint
boundary. The swarm writes a `handoff:N` row before each agent runs that
captures the active agent, the inbound message, and the shared
blackboard. The active agent's own turn / tool checkpoints land under a
nested `agent_path` like `swarm:<s>/agent:<a>/tool:<t>` so the
`/approvals` UI can show "Tool X in Agent Y inside Swarm Z is awaiting
approval" without ambiguity.

```python
from fastaiagent import Agent, Swarm, SQLiteCheckpointer

swarm = Swarm(
    name="content_team",
    agents=[researcher, analyst, reporter],
    entrypoint="researcher",
    handoffs={"researcher": ["analyst"], "analyst": ["reporter"], "reporter": []},
    checkpointer=SQLiteCheckpointer(),
)

result = swarm.run("Investigate the topic.", execution_id="swarm-abc")
# If a tool inside any agent calls interrupt(), result.status == "paused"
# and result.pending_interrupt is populated.

# Resume from a separate process — same API as Agent / Chain.
result = await swarm.aresume("swarm-abc", resume_value=Resume(approved=True))
```

`swarm.resume(execution_id)` (no `resume_value`) is the crash-recovery
path: the swarm parses the active agent from the latest checkpoint's
`agent_path`, recovers `SwarmState` from the most recent `handoff:N`
boundary, re-runs that agent with the saved input, and continues the
loop through any remaining handoffs.

## Supervisor / Worker

Pass a `checkpointer` to `Supervisor` and every delegated worker becomes a
checkpoint subtree. The supervisor's own checkpoints sit at
`supervisor:<name>` and `supervisor:<name>/tool:delegate_to_<role>`. Each
worker's checkpoints sit at `supervisor:<name>/worker:<role>/...` — the
worker's `agent_path_label` is overridden to `"worker:<role>"` so the
hierarchy is unambiguous in the `/approvals` UI.

```python
from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent.agent.team import Supervisor, Worker

planner = Supervisor(
    name="planner",
    llm=...,
    workers=[
        Worker(agent=Agent(name="researcher", ...), role="researcher", ...),
        Worker(agent=Agent(name="auditor",    ...), role="auditor",    ...),
    ],
    checkpointer=SQLiteCheckpointer(),
)

result = planner.run("Investigate and report.", execution_id="job-42")
# If a worker tool calls interrupt(), result.status == "paused" and
# result.pending_interrupt.agent_path is e.g.
# "supervisor:planner/worker:auditor/tool:approve_refund".

# Resume from a separate process:
result = await planner.aresume("job-42", resume_value=Resume(approved=True))
```

`supervisor.resume(execution_id)` (no `resume_value`) is the
crash-recovery path: the supervisor recovers its original input from the
earliest supervisor checkpoint, then re-runs `supervisor.arun(...)` with
the same `execution_id`. The supervisor's LLM is re-issued; each
`delegate_to_<role>` tool detects existing worker state and resumes that
worker (LLM-free re-invocation of any saved pre-tool boundary), or runs
the worker fresh if no state exists yet. Subsequent workers run normally.

> **Cost note:** the supervisor's LLM is re-issued on every resume, just
> like Chain and Swarm. The deterministic-mock pattern in
> `tests/e2e/test_gate_supervisor_durability.py` shows how to make
> production prompts equivalently deterministic; in production, wrap any
> side-effectful tool with [`@idempotent`](../chains/idempotency.md).

## See also

- [Suspending HITL — `interrupt()`](../chains/hitl.md) — the primitive
  every paused checkpoint relies on.
- [`@idempotent`](../chains/idempotency.md) — protects side effects from
  resume re-runs.
- [Checkpointing](../chains/checkpointing.md) — the underlying storage
  contract shared with Chain.
