"""
Self-improving research agent — closed loop demo.

Reuses the ``deep-research-agent`` template from PR A and wraps it with
the trace learning loop introduced in PR B. The end-to-end story:

    1. Run deep_research on N seed topics.            (Phase 1)
    2. Run the offline learn loop over those traces.  (Phase 2)
    3. Run deep_research on a follow-up topic.        (Phase 3)
       PersistentFactBlock automatically injects the
       facts learned in Phase 2 into the scope/write
       agents — observable in the trace.

The point isn't that the report is "better" by some objective metric
(that's a multi-trace eval problem); it's that the same template, with
no code changes, picks up durable knowledge from previous runs.

Usage:
    python agent.py --topic "MCP server adoption update"
    python agent.py --self-test           # offline-friendly smoke
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent.learn import MemoryStore, run_extraction


# Reuse deep-research-agent. Both example folders ship an ``agent.py`` so a
# plain ``import agent`` would collide with this file. Load the sibling
# template under a unique module name via importlib instead.
_DEEP_RESEARCH = Path(__file__).resolve().parent.parent / "deep-research-agent"
if str(_DEEP_RESEARCH) not in sys.path:
    # memory_setup imports `import fastaiagent`, so its directory still needs
    # to be on sys.path; we just don't want ``import agent`` to resolve here.
    sys.path.insert(0, str(_DEEP_RESEARCH))


def _load_deep_research_module():
    """Load deep-research-agent/agent.py under a unique module name."""
    target = _DEEP_RESEARCH / "agent.py"
    spec = importlib.util.spec_from_file_location(
        "deep_research_agent_main", str(target)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deep_research_agent_main"] = module
    spec.loader.exec_module(module)
    return module


_dr = _load_deep_research_module()
run_deep_research = _dr.run_deep_research

# Now memory_setup is safe to import — it doesn't shadow this file.
from memory_setup import SCOPE, SCOPE_ID  # noqa: E402

SEED_TOPICS = [
    "Retrieval-augmented generation",
    "Self-attention in the original Transformer architecture",
    "Model Context Protocol — what it standardizes",
]


def _extractor_llm() -> fa.LLMClient:
    return fa.LLMClient(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL_EXTRACTOR", "gpt-4o-mini"),
    )


async def phase_1_seed() -> None:
    """Run the deep-research pipeline on a handful of seed topics."""
    for i, topic in enumerate(SEED_TOPICS, start=1):
        print(f"\n[seed {i}/{len(SEED_TOPICS)}] topic = {topic!r}")
        report = await run_deep_research(topic)
        print(f"  → report: {len(report)} chars")


def phase_2_learn() -> int:
    """Run the offline trace-learning extractor."""
    store = MemoryStore()
    results = run_extraction(
        llm=_extractor_llm(),
        store=store,
        scope=SCOPE,
        scope_id=SCOPE_ID,
        last_hours=2,  # within this script's execution window
        max_facts_per_trace=10,
    )
    written = sum(len(r.written_ids) for r in results)
    print(f"\n[learn] processed {len(results)} traces, persisted {written} facts")
    if written:
        print("[learn] sample facts:")
        active = store.list_active(scope=SCOPE, scope_id=SCOPE_ID, limit=5)
        for f in active:
            print(f"        • {f.fact}")
    return written


async def phase_3_replay(follow_up_topic: str) -> str:
    """Run a follow-up research query — facts now flow in via memory_setup."""
    print(f"\n[replay] follow-up topic = {follow_up_topic!r}")
    print(
        "[replay] PersistentFactBlock will inject facts learned in phase 2 "
        "into the scope and writer prompts. Inspect the trace in the local "
        "UI (`fastaiagent ui`) to see them."
    )
    return await run_deep_research(follow_up_topic)


async def run_closed_loop(follow_up: str, skip_seed: bool = False) -> None:
    if not skip_seed:
        print("=" * 60)
        print("Phase 1 — seed traces by running deep_research on N topics")
        print("=" * 60)
        await phase_1_seed()

    print()
    print("=" * 60)
    print("Phase 2 — extract durable facts (offline learn loop)")
    print("=" * 60)
    phase_2_learn()

    print()
    print("=" * 60)
    print("Phase 3 — replay deep_research with facts now flowing")
    print("=" * 60)
    report = await phase_3_replay(follow_up)
    print()
    print("─" * 60)
    print("Final report:")
    print("─" * 60)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-improving deep-research demo")
    parser.add_argument(
        "--topic",
        default=(
            "How does Self-RAG differ from vanilla RAG, and where does MCP fit in?"
        ),
        help="Follow-up topic that benefits from the seed runs.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Reuse traces already in local.db; skip phase 1.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Offline smoke: imports + MemoryStore wiring only, no LLM calls.",
    )
    args = parser.parse_args()

    if args.self_test:
        store = MemoryStore()
        # Confirm the scope/scope_id constants reach the store layer cleanly.
        _ = store.list_active(scope=SCOPE, scope_id=SCOPE_ID)
        print("self-test: ok")
        return

    asyncio.run(run_closed_loop(args.topic, skip_seed=args.skip_seed))


if __name__ == "__main__":
    main()
