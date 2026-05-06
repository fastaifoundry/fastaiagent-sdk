"""
Trace Learning Loop — minimal standalone demo.

The narrative:

  1. Run a tiny ``Agent`` a few times to seed traces in ``local.db``.
  2. Run the trace-learning extractor over those traces.
  3. Show the facts it learned and re-inject them into a new agent via
     ``PersistentFactBlock``.

This isolates the learning loop from any specific agent template so
you can read it top-to-bottom and understand the moving parts:

    Agent runs ──→ traces in local.db
                       │
                       ▼
              run_extraction(LLM, store)
                       │
                       ▼
                 learned_memory rows
                       │
                       ▼
        PersistentFactBlock(scope=…) reads them
                       │
                       ▼
              Next Agent run picks them up

For a more realistic, end-to-end story, see
``examples/self-improving-research``.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent.learn import MemoryStore, run_extraction

SCOPE = "agent"
SCOPE_ID = "learning-loop-demo"


def _llm() -> fa.LLMClient:
    return fa.LLMClient(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )


def _extractor_llm() -> fa.LLMClient:
    return fa.LLMClient(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL_EXTRACTOR", "gpt-4o-mini"),
    )


async def _seed_traces(prompts: list[str]) -> None:
    """Run a tiny Agent on each prompt to populate traces in local.db."""
    agent = fa.Agent(
        name=SCOPE_ID,
        system_prompt=(
            "You are a concise assistant. Always introduce yourself as "
            "the Learning Loop demo agent on the first turn, and answer "
            "the user's question in 1-2 sentences."
        ),
        llm=_llm(),
    )
    for i, prompt in enumerate(prompts, start=1):
        print(f"  [seed {i}/{len(prompts)}] {prompt[:60]}…")
        await agent.arun(prompt)


def _print_facts(store: MemoryStore) -> None:
    facts = store.list_active(scope=SCOPE, scope_id=SCOPE_ID)
    if not facts:
        print("  (no active facts)")
        return
    for f in facts:
        print(f"  • [{f.scope}/{f.scope_id}] {f.fact}")


async def run() -> None:
    print("=" * 60)
    print("Step 1 — seeding traces by running a small agent a few times")
    print("=" * 60)
    seed_prompts = [
        "What is the capital of France?",
        "Who wrote 'The Brothers Karamazov'?",
        "What does the abbreviation HTTP stand for?",
    ]
    await _seed_traces(seed_prompts)

    print()
    print("=" * 60)
    print("Step 2 — extracting durable facts from those traces")
    print("=" * 60)
    store = MemoryStore()
    results = run_extraction(
        llm=_extractor_llm(),
        store=store,
        scope=SCOPE,
        scope_id=SCOPE_ID,
        last_hours=1,  # we just wrote them
        max_facts_per_trace=5,
    )
    written = sum(len(r.written_ids) for r in results)
    candidates = sum(len(r.candidates) for r in results)
    print(f"  scanned {len(results)} traces — {candidates} candidates, {written} written")

    print()
    print("=" * 60)
    print("Step 3 — what's in learned_memory now")
    print("=" * 60)
    _print_facts(store)

    print()
    print("=" * 60)
    print("Step 4 — wire PersistentFactBlock into a new agent and ask")
    print("=" * 60)
    memory = fa.ComposableMemory(
        primary=fa.AgentMemory(),
        blocks=[fa.PersistentFactBlock(scope=SCOPE, scope_id=SCOPE_ID, max_facts=20)],
    )
    new_agent = fa.Agent(
        name=f"{SCOPE_ID}:replay",
        system_prompt="You are the same demo assistant.",
        llm=_llm(),
        memory=memory,
    )
    print("  > Asking: 'Tell me what you remember about yourself.'")
    result = await new_agent.arun("Tell me what you remember about yourself.")
    print()
    print(result.output)
    print()
    print(
        "↑ The system prompt was augmented with the learned facts above. "
        "Inspect the trace in the local UI to see them injected as a "
        "SystemMessage by PersistentFactBlock."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Learning Loop demo")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run a tiny smoke flow that doesn't call any LLM. Used by the "
            "test gate to verify the imports + DB wiring without burning "
            "tokens."
        ),
    )
    args = parser.parse_args()

    if args.self_test:
        # Just verify imports + that MemoryStore can be constructed.
        store = MemoryStore()
        # Empty list is fine — we just want to confirm the table exists.
        assert isinstance(store.list_active(scope=SCOPE, scope_id="never"), list)
        print("self-test: ok")
        return

    asyncio.run(run())


if __name__ == "__main__":
    main()
