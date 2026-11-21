# Cyclic Workflows

A chain can loop: an edge that goes back to an earlier node, repeated until a condition holds or
a maximum number of passes is reached. Use it to check an agent's output and retry when it falls
short.

## Creating a retry loop

```python
from fastaiagent import Agent, Chain, FunctionTool, LLMClient
from fastaiagent.chain import NodeType


def check_length(draft: str) -> str:
    words = len(draft.split())
    if words <= 15:
        return f"PASS ({words} words)"
    return f"FAIL: the draft is {words} words; rewrite it in at most 15 words."


chain = Chain("draft-with-check")
chain.add_node("draft", agent=drafter)      # an agent writes
chain.add_node(                             # a function judges the draft
    "check",
    tool=FunctionTool(name="check_length", fn=check_length),
    input_mapping={"draft": "{{node_results.draft.output}}"},
)
chain.add_node("publish", type=NodeType.transformer, template="{{node_results.draft.output}}")

chain.connect("draft", "check")
chain.connect("check", "publish")
chain.connect(
    "check", "draft",                       # the edge that goes back
    max_iterations=3,                       # at most 3 passes through the loop
    exit_condition="{{node_results.check.output}} startswith PASS",
)
```

1. `draft` writes, `check` judges.
2. If the check does not start with `PASS`, the chain goes back to `draft`.
3. When it does, the loop ends and the chain continues to `publish`, which runs **once**.
4. If three passes go by without a `PASS`, the chain raises `ChainCycleError`.

The runnable version is
[`examples/02_chain_with_cycles.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/02_chain_with_cycles.py).
A real run of it:

```text
Passes: 2
  check 1: FAIL: the draft is 244 words; rewrite it in at most 15 words.
  check 2: PASS (13 words)
Published: Framework structuring AI agents to enable modular, scalable, coordinated, and faster team development.
```

## How data moves in a loop

- An **agent** node is given the chain's input by default: the state at the start of the pass. On
  a second pass that state holds what the loop's source node last wrote (above, the check's
  `FAIL: ...`), which is how `draft` learns what to fix.
- Give an agent node `input=` to hand it another node's output instead
  ([What an agent node is given](index.md#what-an-agent-node-is-given)). It is re-rendered on every
  pass, so it always reads the latest output.
- A **tool** or **transformer** node reads earlier results with `input_mapping=` or `template=`.

### An agent as the judge

The check above is a function. When judging needs a model, make the judge an agent and hand it the
draft with `input=`, and hand the drafter the last review the same way:

```python
chain.add_node(
    "draft",
    agent=drafter,
    input="Topic: {{input.message}}\nLast review: {{node_results.review.output}}",
)
chain.add_node(
    "review",
    agent=reviewer,               # answers "PASS" or "FAIL: <what to fix>"
    input="Review this draft:\n{{node_results.draft.output}}",
)
chain.connect("draft", "review")
chain.connect(
    "review", "draft",
    max_iterations=3,
    exit_condition="{{node_results.review.output}} startswith PASS",
)
```

On the first pass `Last review:` is empty, because the reviewer has not run yet. The template only
stops the run when the whole input renders empty.

## Loop settings

| Parameter | What it does | Default |
|---|---|---|
| `max_iterations` | The most passes the loop body runs before `ChainCycleError` | `10` when only `exit_condition` is given |
| `exit_condition` | Leave the loop when this is true | None: the loop runs until `max_iterations` and then raises |

Passing either one makes the edge a loop edge. A pass counts each time the loop's source node
(`check` above) finishes, so `max_iterations=3` means three drafts and three checks.

## Exit conditions

An exit condition is a template over the run, compared with one operator:

- Values come from `{{state.<key>}}`, `{{node_results.<node>.output}}` or `{{input.<key>}}`.
- Operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`, `startswith`.

```python
exit_condition="{{state.quality}} >= 0.8"
exit_condition="{{node_results.check.output}} startswith PASS"
exit_condition="{{state.done}} == true"
```

!!! warning "A bare word is text, not a lookup"
    `exit_condition="quality >= 0.8"` compares the literal word `quality` with `0.8`, which is
    never true, so the loop runs to `max_iterations` and raises. Write `{{state.quality}} >= 0.8`.
    Earlier versions of this page showed the bare form.

The condition is checked against what the loop's source node just wrote, so a node that sets
`state.quality` can end the loop in the same pass.

## What runs inside a loop

- The chain's middleware applies on every pass, with one state for the whole run, so a budget or
  counter keeps counting across passes.
- Nodes that ran before the loop stay visible to templates inside it
  (`{{node_results.<earlier node>.output}}`).
- Nodes after the loop run once, after the last pass.

## Validation

```python
errors = chain.validate()
```

`validate()` reports a loop edge whose `max_iterations` is below 1, and a loop drawn with
ordinary edges:

```text
Nodes ['check', 'draft'] form a loop through ordinary edges, which is never followed.
Mark the edge that goes back with connect(..., max_iterations=N) (and exit_condition= to leave early).
```

An edge that goes back without `max_iterations` or `exit_condition` is not a loop: the executor
does not follow it. Call `validate()` before running.

## Error handling

```python
from fastaiagent._internal.errors import ChainCycleError

try:
    result = chain.execute({"input": "data"})
except ChainCycleError as e:
    print(f"Loop limit hit: {e}")
    # "Cycle 'check' -> 'draft' exceeded max_iterations (3). ..."
```

## Changed in 1.105.0

Four things in this page were not true before 1.105.0, in every earlier version:

- **`max_iterations` above 1 was not enforced.** A loop whose exit condition never held ran until
  Python's recursion limit, about 980 passes, then failed with `RecursionError`. If you have a
  loop that relied on running more than its `max_iterations` (or more than the default 10), it
  now raises `ChainCycleError` at the cap: raise `max_iterations` to what you need.
- **Nodes after the loop ran once per pass**, not once.
- **Chain middleware did not apply inside a loop pass**, and a pass could not see the results of
  nodes that ran before the loop.
- **An `exit_condition` without `max_iterations` was dropped**, and the edge was not a loop.

A resumed run starts its loop count again from zero.

---

## Next steps

- [Chains](index.md) — Core chain documentation
- [Checkpointing](checkpointing.md) — Save and resume chain execution
- [Human-in-the-Loop](hitl.md) — Pause chains for human approval
