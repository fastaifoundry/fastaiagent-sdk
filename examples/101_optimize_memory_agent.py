"""Example 101: optimize an agent built on a per-user ``Memory`` (1.82.0).

``optimize()`` can tune which of an agent's facts to inject (the memory lever).
With a per-user ``Memory`` two things have to hold, and both used to break:

  * **The lever reads where the agent reads.** Its facts live in the
    ``Memory``'s own store and project — here a separate SQLite file under
    ``project_id="acme"``, not the default ``local.db``. The lever used to look
    only in ``local.db``, find nothing, and skip.
  * **The optimized agent keeps users apart.** ``report.apply_to()`` returns an
    agent whose memory is still a per-user ``Memory`` — it used to return one
    window shared by every user.

Two global facts are seeded: one helps the eval (answer with the city only),
one hurts it (add a historical aside). The lever tries subsets of them.

Usage:
    zsh -lc 'python examples/101_optimize_memory_agent.py'   # needs OPENAI_API_KEY

Expected output (shape; scores and wording vary):
    [baseline] skipped=False accepted=True dev=0.00 baseline     (both facts: low)
    [memory] skipped=False accepted=False dev=0.00 facts k=2
    [memory] skipped=False accepted=True dev=1.00 facts k=1      (the helpful one alone)
    optimized memory is a Memory: True
    bob -> UNKNOWN          (alice's codeword never reaches bob)
    alice -> PELICAN-7
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass


@dataclass
class Session:
    user_id: str


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1
    workdir = tempfile.mkdtemp(prefix="fa-ex101-")
    os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(workdir, "local.db")
    warnings.simplefilter("ignore")  # "only 8 cases" — fine for a demo

    import fastaiagent as fa
    from fastaiagent.learn import MemoryStore

    mem = fa.Memory(
        location=MemoryStore(db_path=os.path.join(workdir, "agent-facts.db")),
        agent_id="capitals",
        project_id="acme",
        user_id=lambda ctx: ctx.state.user_id,
    )
    # The lever ranks facts by confidence, then recency, and ablates from the
    # bottom: with equal confidence, the newest fact is the one a smaller subset keeps.
    mem.persist("Always add a sentence of historical background.", tier="global")
    mem.persist("Answer with the city name only, no punctuation.", tier="global")

    llm = fa.LLMClient(provider="openai", model="gpt-4o-mini")
    agent = fa.Agent(name="capitals", system_prompt="You answer questions.", llm=llm, memory=mem)
    cases = [
        {"input": f"What is the capital of {country}?", "expected_output": city}
        for country, city in [
            ("France", "Paris"),
            ("Japan", "Tokyo"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("Germany", "Berlin"),
            ("Canada", "Ottawa"),
            ("Egypt", "Cairo"),
            ("Kenya", "Nairobi"),
        ]
    ]
    report = fa.optimize(
        agent,
        cases,
        scorers=["exact_match"],
        proposer_llm=llm,
        config=fa.OptimizeConfig(levers=("memory",), max_iterations=1),
    )
    for p in report.trajectory:
        status = f"skipped={p.skipped} accepted={p.accepted} dev={p.dev_score:.2f}"
        print(f"[{p.lever}] {status} {p.rationale}")

    better = report.apply_to(agent)
    print("optimized memory is a Memory:", isinstance(better.memory, fa.Memory))

    def say(user: str, text: str) -> str:
        return better.run(text, context=fa.RunContext(state=Session(user))).output

    say("alice", "My codeword is PELICAN-7. Reply with just: ok")
    print("bob ->", say("bob", "What is my codeword? Reply UNKNOWN if you don't know."))
    print("alice ->", say("alice", "What is my codeword? Reply with the codeword only."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
