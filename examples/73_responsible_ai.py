"""Responsible AI — the Trust Layer.

A coherent set of runtime guardrails (plus the ``Reflect`` middleware) that make
an agent's I/O truthful, safe, and on-policy:

* ``no_secrets()``         — block leaked API keys / tokens / private keys
* ``grounded()``           — block hallucinations vs your source text
* ``toxicity_check()``     — keyword by default, ``mode="llm"`` for a 0–1 score
* ``banned_topics()`` / ``allowed_topics()`` — keep the agent on-mission
* ``Reflect``              — self-critique the final answer against facts
* ``responsible_ai()``     — compose the bundle in one call

The zero-dependency checks need no API key; the LLM-backed checks reuse your
``LLMClient`` and only run here when ``OPENAI_API_KEY`` is set.

Run:
    zsh -lc 'python examples/73_responsible_ai.py'
"""

from __future__ import annotations

import os

import fastaiagent as fa
from fastaiagent import Agent, LLMClient, responsible_ai


def demo_zero_dependency() -> None:
    """These run anywhere — no API key, no network."""
    print("== Secrets ==")
    g = fa.no_secrets()
    leak = 'config: api_key = "sk_live_0123456789abcdef0123"'
    res = g.execute(leak)
    print(f"  blocked={not res.passed}  ->  {res.message}")  # raw secret never shown

    print("== Toxicity (keyword default) ==")
    tox = fa.toxicity_check()
    print(f"  'have a nice day'  passed={tox.execute('have a nice day').passed}")
    print(f"  'I will attack you' passed={tox.execute('I will attack you').passed}")

    print("== Topic controls (keyword mode) ==")
    banned = fa.banned_topics(["politics"], mode="keyword")
    print(f"  'the weather'  passed={banned.execute('the weather').passed}")
    print(f"  'politics now' passed={banned.execute('politics now').passed}")

    print("== Default bundle (LLM-free) ==")
    print("  ", [r.name for r in responsible_ai()])


def demo_with_llm() -> None:
    """Groundedness, reflection, LLM toxicity, semantic topics — needs a key."""
    llm = LLMClient(provider="openai", model="gpt-4o-mini")

    print("\n== Groundedness ==")
    reference = "Paris is the capital of France. The Eiffel Tower is in Paris."
    grounded = fa.grounded(reference, llm=llm, threshold=0.7)
    ok = grounded.execute("The capital of France is Paris.")
    bad = grounded.execute("The capital of France is Berlin.")
    print(f"  supported -> passed={ok.passed} (score={ok.score})")
    print(f"  wrong     -> passed={bad.passed} (score={bad.score})")

    print("== Reflection (self-critique vs a fact) ==")
    from fastaiagent import MiddlewareContext, Reflect
    from fastaiagent._internal.async_utils import run_sync
    from fastaiagent.llm.client import LLMResponse

    reflect = Reflect(facts=["The capital of France is Paris."], llm=llm)
    draft = LLMResponse(content="The capital of France is Berlin.")
    revised = run_sync(reflect.after_model(MiddlewareContext(), draft))
    print(f"  draft 'Berlin' revised to: {revised.content!r}")

    print("== A guarded agent (bundle) ==")
    agent = Agent(
        name="safe-support",
        system_prompt="You are a concise support agent.",
        llm=llm,
        guardrails=responsible_ai(toxicity=True, banned=["politics"], llm=llm),
    )
    print(f"  {agent.run('How do I reset my password?').output}")


def main() -> None:
    demo_zero_dependency()
    if os.environ.get("OPENAI_API_KEY"):
        demo_with_llm()
    else:
        print("\n(Set OPENAI_API_KEY to also run the groundedness / reflection / LLM demos.)")


if __name__ == "__main__":
    main()
