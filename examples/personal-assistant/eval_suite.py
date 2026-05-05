"""
Evaluation Suite — multi-turn session quality testing.

Run: python eval_suite.py
     python eval_suite.py --publish     # publish to FastAIAgent Platform

The hard thing to eval here isn't a single-turn answer — it's whether
memory layers do their job over a session. Three custom Python scorers
exercise the four block types in concert:

  * **fact_extraction_recall** — after a 4-turn session that mentioned
    several durable facts, the FactExtractionBlock must have captured
    a curated must-extract subset.

  * **vector_recall_relevance** — given a query about something
    discussed two turns ago, the VectorBlock must surface that prior
    exchange in its top-k. Asserts on the rendered SystemMessage.

  * **summary_block_active** — after enough turns to hit
    ``summarize_every`` threshold, the SummaryBlock must have populated
    a non-empty summary string.

A single canonical session is run end-to-end against a real LLM; the
scorers inspect the live ``ComposableMemory`` state afterwards. Persisted
to local.db so the run shows up at /evals.
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent.agent.memory import ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock,
    SummaryBlock,
    VectorBlock,
)
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import ScorerResult

from agent import _load_system_prompt
from memory_setup import build_memory
from tools import add_note, list_facts, make_deps, search_notes, today


# ─── Canonical session ───────────────────────────────────────────────────────
#
# Six turns chosen so:
#   - turns 1 + 2 + 4 mention durable facts FactExtractionBlock should catch
#   - turn 6 asks about turn 2's content — exercises VectorBlock recall
#   - turn count > summarize_every (4) so SummaryBlock fires

CANONICAL_TURNS: list[str] = [
    "Hi! My name is Riley Chen and I work as a staff engineer at Strato Labs. I'm based in Lisbon.",
    "I'm currently rebuilding our deployment pipeline; the painful part is that we use Argo CD with weird custom hooks.",
    "What's a good morning routine for someone with two kids?",
    "By the way, my partner's name is Jamie and our kids are Mira (4) and Theo (1).",
    "Can you give me three concrete tips on Argo CD hook design?",
    "Earlier I mentioned a deployment-pipeline pain point. Can you remind me what it was?",
]


# Substrings — every one must appear in *some* extracted fact.
MUST_EXTRACT_FACTS = [
    "Riley",
    "Strato Labs",
    "Lisbon",
    "Argo",
    "Jamie",  # partner
    "Mira",   # kid
]


# Substring — recall of turn 2's content when asked in turn 6.
RECALL_QUERY_SUBSTRING = "argo"


# ─── Scorers ────────────────────────────────────────────────────────────────


def _score_fact_extraction(memory: ComposableMemory) -> tuple[float, str]:
    fact_block: FactExtractionBlock | None = next(
        (b for b in memory.blocks if isinstance(b, FactExtractionBlock)), None
    )
    if fact_block is None:
        return 0.0, "no FactExtractionBlock attached"
    facts_text = " ".join(fact_block._facts).lower()
    hits = [needle for needle in MUST_EXTRACT_FACTS if needle.lower() in facts_text]
    ratio = len(hits) / len(MUST_EXTRACT_FACTS)
    return round(ratio, 4), (
        f"matched {len(hits)}/{len(MUST_EXTRACT_FACTS)} required facts; "
        f"total facts captured: {len(fact_block._facts)}"
    )


def _score_vector_recall(memory: ComposableMemory) -> tuple[float, str]:
    """Render the VectorBlock against the recall-test query and check that
    a relevant prior message surfaces in the top-k."""
    vector_block: VectorBlock | None = next(
        (b for b in memory.blocks if isinstance(b, VectorBlock)), None
    )
    if vector_block is None:
        return 0.0, "no VectorBlock attached"
    fragments = vector_block.render(RECALL_QUERY_SUBSTRING)
    if not fragments:
        return 0.0, "VectorBlock returned no fragments for the recall query"
    body = " ".join(str(f.content or "") for f in fragments).lower()
    if RECALL_QUERY_SUBSTRING in body:
        return 1.0, f"recalled prior turn containing {RECALL_QUERY_SUBSTRING!r}"
    return 0.0, f"top-k did not include {RECALL_QUERY_SUBSTRING!r}; got: {body[:140]}"


def _score_summary_active(memory: ComposableMemory) -> tuple[float, str]:
    summary_block: SummaryBlock | None = next(
        (b for b in memory.blocks if isinstance(b, SummaryBlock)), None
    )
    if summary_block is None:
        return 0.0, "no SummaryBlock attached"
    summary_text = (summary_block._summary or "").strip()
    seen = summary_block._messages_seen
    if not summary_text:
        return 0.0, f"SummaryBlock produced no summary after {seen} messages seen"
    return 1.0, f"summary length {len(summary_text)} chars after {seen} messages"


# ─── Runner ─────────────────────────────────────────────────────────────────


async def run_eval(publish: bool = False) -> None:
    print("\nPhase 1 — running canonical 6-turn session...\n")

    # Fresh memory for the eval (don't pick up the developer's REPL state).
    memory = build_memory(memory_dir=None)
    deps = make_deps(memory=memory)
    ctx = fa.RunContext(state=deps)

    agent = fa.Agent(
        name="personal-assistant-eval",
        # Same registry-backed prompt path the REPL agent uses.
        system_prompt=lambda _ctx=None: _load_system_prompt(),
        llm=fa.LLMClient(provider="openai", model="gpt-4o"),
        tools=[add_note, search_notes, list_facts, today],
        memory=memory,
    )

    for i, turn in enumerate(CANONICAL_TURNS, 1):
        result = await agent.arun(turn, context=ctx)
        print(f"  turn {i}: {result.tokens_used} tokens, {result.latency_ms} ms")

    print("\nPhase 2 — scoring memory blocks...\n")
    fr_score, fr_reason = _score_fact_extraction(memory)
    vr_score, vr_reason = _score_vector_recall(memory)
    sa_score, sa_reason = _score_summary_active(memory)

    print(f"  fact_extraction_recall   = {fr_score:.2f}  ({fr_reason})")
    print(f"  vector_recall_relevance  = {vr_score:.2f}  ({vr_reason})")
    print(f"  summary_block_active     = {sa_score:.2f}  ({sa_reason})")

    results = EvalResults()
    results.add("fact_extraction_recall", ScorerResult(score=fr_score, passed=fr_score >= 0.5, reason=fr_reason))
    results.add("vector_recall_relevance", ScorerResult(score=vr_score, passed=vr_score >= 1.0, reason=vr_reason))
    results.add("summary_block_active", ScorerResult(score=sa_score, passed=sa_score >= 1.0, reason=sa_reason))
    results.add_case(
        EvalCaseRecord(
            input="canonical 6-turn session",
            actual_output="(memory state inspected post-session)",
            per_scorer={
                "fact_extraction_recall": {"score": fr_score, "passed": fr_score >= 0.5, "reason": fr_reason},
                "vector_recall_relevance": {"score": vr_score, "passed": vr_score >= 1.0, "reason": vr_reason},
                "summary_block_active": {"score": sa_score, "passed": sa_score >= 1.0, "reason": sa_reason},
            },
        )
    )
    print()
    print(results.summary())

    try:
        run_id = results.persist_local(
            run_name="personal-assistant eval",
            dataset_name="canonical-session",
            agent_name="personal-assistant",
        )
        print(f"\n  ✓ persisted to local.db (run_id={run_id[:12]}...)")
    except Exception as e:
        print(f"\n  Could not persist eval run locally: {e}")

    if publish:
        try:
            results.publish(run_name="personal-assistant eval")
            print("  ✓ published to FastAIAgent Platform")
        except Exception as e:
            print(f"  Platform publish failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="Publish to platform")
    args = parser.parse_args()
    asyncio.run(run_eval(publish=args.publish))


if __name__ == "__main__":
    main()
