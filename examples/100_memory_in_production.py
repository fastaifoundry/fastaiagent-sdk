"""Example 100: a per-user ``Memory`` in a long-running server (1.82.0).

One agent, many users, one process that stays up for weeks. Three things matter
there that a demo never shows:

  * **Windows are bounded.** Every user's conversation window lives in the
    process. ``max_users`` keeps the most recently active ones and drops the
    least recently used — here ``max_users=2`` so you can watch it happen.
  * **A dropped user keeps their facts.** Durable facts live in the store, so a
    returning user starts a fresh conversation but is still recognised.
  * **The user id stays out of the prompt.** A user's facts reach the model
    under ``Learned facts (user):`` — an email id is never sent to the provider.

Usage:
    zsh -lc 'python examples/100_memory_in_production.py'   # needs OPENAI_API_KEY

Expected output (shape; the model's wording varies):
    alice: noted — …
    bob:   noted — …
    carol: noted — …            (the third user: alice's window is dropped)
    alice's window after the drop: 0 messages; her facts: ['Alice is allergic to peanuts.']
    alice again -> … peanuts … no codeword  (her facts, not the dropped conversation)
    user id in the prompt? False
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass


@dataclass
class Session:
    user_id: str


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1
    workdir = tempfile.mkdtemp(prefix="fa-ex100-")
    os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(workdir, "local.db")

    import fastaiagent as fa
    from fastaiagent.learn import Fact, MemoryStore

    store = MemoryStore(db_path=os.path.join(workdir, "facts.db"))
    alice, bob, carol = "alice@example.com", "bob@example.com", "carol@example.com"
    # A fact about Alice, as your app (or `learn=`) would have stored it.
    store.add(Fact(scope="user", scope_id=alice, fact="Alice is allergic to peanuts."))

    llm = fa.LLMClient(provider="openai", model="gpt-4o-mini")
    mem = fa.Memory(
        location=store,
        user_id=lambda ctx: ctx.state.user_id,  # one window per user
        max_users=2,  # production default is 10_000; 2 shows the drop
    )
    agent = fa.Agent(
        name="concierge",
        system_prompt="You are a restaurant concierge. Answer in one short sentence.",
        llm=llm,
        memory=mem,
    )

    def say(user: str, text: str) -> str:
        return agent.run(text, context=fa.RunContext(state=Session(user))).output

    print("alice:", say(alice, "Book me a table for Friday, codeword TULIP."))
    print("bob:  ", say(bob, "Any vegan places nearby?"))
    print("carol:", say(carol, "Is the rooftop bar open?"))  # a third user: alice is dropped

    print(
        "alice's window after the drop:",
        len(mem.for_user(alice).messages),
        "messages; her facts:",
        [f.fact for f in mem.retrieve(tier="user", id=alice)],
    )
    question = "Anything on the menu I should avoid? And do you know my codeword? Say no if not."
    print("alice again ->", say(alice, question))

    # What the model was sent: her facts, under a heading without her email.
    context = mem.for_user(alice).get_context("menu")
    sent = " ".join(m.content or "" for m in context)
    print("user id in the prompt?", alice in sent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
