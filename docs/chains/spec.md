# Chain Execution Spec

This page is the authoritative contract for how a `Chain` runs. Each rule
here is backed by a test in
[`tests/test_chain_routing.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/tests/test_chain_routing.py)
or [`tests/test_chain_resume.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/tests/test_chain_resume.py).
If you encounter behavior that disagrees with this page, that's a bug —
please file an issue.

## Routing

Routing decides which outgoing edges activate after a node completes.

### Edge selection rules

For a non-condition source node with outgoing edges:

| Outgoing shape | Behavior |
|---|---|
| All edges unconditional | Every edge fires (fan-out). Backwards-compatible with pre-routing chains. |
| Any edge has `condition=` | First conditional edge whose expression evaluates true (declaration order) wins. |
| No conditional matches, ≥1 unconditional sibling | The first unconditional edge fires (default fallback). |
| No conditional matches, no unconditional sibling | See **Strict routing** below. |

For a `NodeType.condition` source:

| Result | Behavior |
|---|---|
| `result["matched"] == handle` | The edge whose `label == handle` wins. |
| No matching label | An edge labeled `"default"` is taken if present. |
| No `"default"` label | An unlabeled edge is taken if present. |
| None of the above | See **Strict routing** below. |

### Condition expressions

The router supports the operators documented in
[`docs/chains/index.md`](index.md) — `==`, `!=`, `<`, `<=`, `>`, `>=`,
`contains`, `startswith` — over `{{state.x}}` / `{{node_results.x.y}}` /
`{{input.y}}` templates resolved against the post-update context.

### Strict routing

```python
chain = Chain("my-flow", strict_routing=True)
```

When `strict_routing=True`, the "no edge matched" fall-through raises
`ChainRoutingError` instead of silently terminating the branch. The error
message names the node and the available labels so the misconfiguration
is obvious. Defaults to `False` for backwards compatibility — existing
chains continue to silently prune.

### State semantics

`ChainState.data` always returns a **copy**, not a live reference.
Mutating the returned dict never affects chain state. This is a stable
guarantee — the router rebuilds its context dictionary post-node so
condition expressions read the state the just-executed node wrote.

For checkpoint snapshotting, use `ChainState.snapshot()` which returns a
deep, JSON-safe copy (with `Image` / `PDF` instances serialized via their
`to_dict` helpers).

## Parallel

`NodeType.parallel` nodes fan child agents out via `asyncio.gather`. The
`NodeConfig.parallel_failure_mode` field (default `"continue"`) selects
how exceptions are surfaced:

| Mode | Behavior |
|---|---|
| `"continue"` *(default)* | Collect every result. Exceptions become `{"error": str(e)}` entries in `outputs`. Backwards-compatible. |
| `"fail_fast"` | First child exception cancels siblings and re-raises as `ChainError`. |
| `"any_success"` | Collect every result; filter out errors. If **every** child failed, raise `ChainError`. |

```python
node = NodeConfig(
    id="fan",
    type=NodeType.parallel,
    config={"agents": [a, b, c]},
    parallel_failure_mode="any_success",
)
```

## Cycles and interrupts

When a node inside a cycle calls `interrupt()`:

1. The executor catches `InterruptSignal`, persists an
   `"interrupted"` checkpoint, and returns `status="paused"` up to the
   caller — the cycle's iteration counter is **not** reset.
2. On `Chain.resume(execution_id, resume_value=Resume(...))`, the
   interrupted node re-executes with the resume value in scope. Cycle
   accounting continues from where it paused.
3. If the resumed node interrupts again, the cycle pauses again — the
   re-resume cycle is supported.

If a cyclic edge fires while the chain is being recursed for an earlier
cycle, only the outermost cycle's exit condition is consulted. This is
deliberate: nested cycles are evaluated outside-in.

## Resume contract

`Chain.resume(execution_id, *, resume_value=None, modified_state=None)`:

| Latest checkpoint status | `resume_value` provided? | Result |
|---|---|---|
| `"interrupted"` | Yes | Re-execute the interrupted node with `_resume_value` in scope. |
| `"interrupted"` | No | Raise `ChainResumeError` — interrupted runs need a `Resume(...)`. |
| any (no pending interrupt) | Yes | Raise `AlreadyResumed` — either the chain was never interrupted, or a prior resume already claimed the row. UI/CLI translate this to HTTP 409. |
| `"failed"` | No | Re-execute starting at the node *after* the failed one, optionally with `modified_state` patched in. |
| Unknown execution_id | (either) | Raise `ChainCheckpointError`. |

`ChainResumeError` subclasses `ChainCheckpointError` for backward
compatibility — existing `except ChainCheckpointError:` handlers in
v1.13.x and earlier continue to catch the new resume errors.

## Validation

`Chain.validate()` runs structural checks before execution:

| Rule | Severity |
|---|---|
| Chain has at least one node | error |
| All edge `source` and `target` IDs reference existing nodes | error |
| Cyclic edges have `max_iterations >= 1` | error |
| Condition nodes have an edge for every declared handle, plus a default | error |
| Non-condition sources have at most one unconditional edge alongside conditionals | error |
| Every node is reachable via some edge | error |

A node can opt out of the reachability error by passing
`reachable=False` in its config — for diagnostic nodes that are
intentionally disconnected:

```python
chain.add_node("diag", agent=diagnostic_agent, reachable=False)
```

The validator's coverage mirrors the router's selection logic: any
misconfiguration the router can't disambiguate fails validation at
`Chain.validate()` time, *before* the run starts.
