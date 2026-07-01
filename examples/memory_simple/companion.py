"""The simple `Memory` API — one object, multi-user, learns across sessions.

Shows the recommended front-door memory API:
- `Memory(user_id=<resolver>, learn=llm)` — one agent serves many users, safely;
- a global fact via `Memory.persist(tier="global")`;
- durable per-user facts learned + persisted during the run (with a source trace);
- complete isolation between users (durable facts *and* the live window).

Writes `last_run.json` (a trace id + db path) for `snapshot.py`.

Run:
    zsh -lc 'python companion.py'      # needs OPENAI_API_KEY
    zsh -lc 'python snapshot.py'       # capture the UI
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "memory_simple.db"
LAST_RUN = HERE / "last_run.json"


@dataclass
class Session:
    user_id: str


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — export it (it's in ~/.zshrc) and re-run via zsh -lc.")
        return 1

    os.environ["FASTAIAGENT_LOCAL_DB"] = str(DB_PATH)
    if DB_PATH.exists():
        DB_PATH.unlink()

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace.otel import reset as reset_tracer

    reset_config()
    reset_tracer()

    from fastaiagent import Agent, LLMClient, Memory
    from fastaiagent.agent.context import RunContext
    from fastaiagent.learn import MemoryStore

    llm = LLMClient(provider="openai", model="gpt-4.1")

    # ONE Memory, ONE agent — serves every user via the per-run resolver.
    mem = Memory(
        location=MemoryStore(db_path=str(DB_PATH)),
        agent_id="assistant",  # global tier
        user_id=lambda ctx: ctx.state.user_id,  # user tier, resolved per run
        learn=llm,  # extract + persist user facts
        window=20,
    )
    mem.persist("This assistant answers in a friendly, concise tone.", tier="global")

    agent = Agent(
        name="assistant",
        system_prompt="You are a concise assistant. Use what you remember about the user.",
        llm=llm,
        memory=mem,
    )

    alice = RunContext(state=Session(user_id="alice"))
    bob = RunContext(state=Session(user_id="bob"))

    last_trace_id = None
    for ctx, text in [
        (alice, "I'm Alice. I have a dog named Rex and I love hiking."),
        (bob, "I'm Bob. I have a cat named Mia and I'm allergic to dogs."),
        (alice, "What's my pet's name?"),
    ]:
        result = agent.run(text, context=ctx)
        last_trace_id = getattr(result, "trace_id", None) or last_trace_id
        print(f"[{ctx.state.user_id}] {text}\n    -> {result.output}\n")

    LAST_RUN.write_text(json.dumps({"trace_id": last_trace_id, "db_path": str(DB_PATH)}, indent=2))
    print(f"Wrote {LAST_RUN.name}: trace_id={last_trace_id}")
    print("Now run:  zsh -lc 'python snapshot.py'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
