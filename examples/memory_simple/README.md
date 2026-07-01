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
- **Trace** — `memory.read` / `memory.write` spans with per-block children, plus
  a `memory.persist` span for the direct global write.
- **Memory page** (sidebar → Knowledge → Memory) — `global` + `user:alice` +
  `user:bob` facts, each with a source `trace` link and confidence.

## Notes
- `Memory(user_id=<resolver>)` keeps a per-user working window **in-process** —
  ideal for dev / single-node. Large-scale multi-user wants an external session
  store (Phase 2). Safe-by-default: a missing/unresolved user id yields no
  personal facts.
- The composable blocks (`ComposableMemory`, `VectorBlock`, …) still exist for
  advanced/custom behaviours; `Memory` is the recommended default.
