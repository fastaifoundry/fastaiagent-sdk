# The simple `Memory` API

One object for agent memory — tiered, multi-user safe, and observable.

```python
from fastaiagent import Agent, LLMClient, Memory

agent = Agent(name="support", llm=llm, memory=Memory(
    user_id=lambda ctx: ctx.state.user_id,   # one agent, many users (per-run id)
    learn=llm,                               # extract + persist durable user facts
))
```

This example runs **one** agent for **two** users (Alice and Bob) plus a global
fact, and shows that:

- durable facts are isolated per user (`user:alice` vs `user:bob`), learned +
  persisted during the run with a source trace;
- the **live conversation window is also isolated** — Alice never sees Bob's
  messages and vice versa (ask Alice her pet's name → "Rex", not Bob's "Mia");
- a global fact (`tier="global"`) is shared across everyone.

## Run

```sh
zsh -lc 'python companion.py'      # needs OPENAI_API_KEY
pip install playwright && python -m playwright install chromium
zsh -lc 'python snapshot.py'       # captures the UI to screenshots/
```

## What to look at
- **Trace** — `memory.read` / `memory.write` spans with per-block children. The
  direct global write happens before the conversation, so its `memory.persist`
  span is a trace of its own.
- **Memory page** (Build → Knowledge → Memory, or `/memory`) — the global fact
  under `agent:assistant` plus the `user:alice` and `user:bob` facts, each with
  its source (`trace` link or `manual`) and confidence.

## Notes
- `Memory(user_id=<resolver>)` keeps a per-user working window **in-process** —
  ideal for dev / single-node. Durable facts can live in Postgres or Redis
  (`examples/memory_backends/`); persist a user's window yourself with
  `mem.for_user("alice").save(path)`.
- A run whose user can't be resolved (no `context=`, a `None` id, or a resolver
  that raises) gets no conversation memory — global facts only — so anonymous
  callers never share a window. For a dict state, resolve with
  `lambda ctx: ctx.state["user_id"]`.
- The composable blocks (`ComposableMemory`, `VectorBlock`, …) still exist for
  advanced/custom behaviours; `Memory` is the recommended default.
