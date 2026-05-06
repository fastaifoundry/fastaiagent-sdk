"""
Eval suite — golden-question regression for the deep-research pipeline.

Each case is a topic the pipeline should produce a citation-rich Markdown
report for. We score with two custom scorers:

  * ``HasMinCitations``  — final report contains ≥ N numbered citations.
  * ``CitationsInTrail`` — every cited URL appears in the search ``trail``
    populated by the ``web_search`` tool, i.e. the writer didn't invent
    URLs the researchers never saw.

This intentionally avoids LLM-judge scorers because they're slow,
non-deterministic, and expensive. A LLM-judge ``Faithfulness`` scorer is
easy to add via ``fastaiagent.eval.scorers.builtin`` if desired.

Run:
    pytest -x examples/deep-research-agent/eval_suite.py        # not a pytest file by default
    python eval_suite.py                                        # treat as a script
"""

from __future__ import annotations

import asyncio
import re

from tools import make_deps

from fastaiagent.eval import Dataset, Scorer

CASES = [
    {
        "id": "rag-basics",
        "input": "Retrieval-augmented generation: what it is and why it works",
    },
    {
        "id": "transformer-attention",
        "input": "Self-attention in the original Transformer architecture",
    },
]


# ─── Scorers ─────────────────────────────────────────────────────────────────


class HasMinCitations(Scorer):
    """Score = 1.0 if the report has at least ``min_count`` numbered citations."""

    name = "has_min_citations"

    def __init__(self, min_count: int = 2):
        self.min_count = min_count

    async def ascore(
        self,
        *,
        input: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list | None = None,
    ) -> float:
        cites = len(re.findall(r"\[\d+\]", actual_output or ""))
        return 1.0 if cites >= self.min_count else 0.0


class CitationsInTrail(Scorer):
    """Score = 1.0 if every URL in the Sources section was actually retrieved."""

    name = "citations_in_trail"

    async def ascore(
        self,
        *,
        input: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list | None = None,
    ) -> float:
        # Pull URLs from the Sources block at the bottom of the report.
        urls_in_report = set(re.findall(r"https?://\S+", actual_output or ""))
        if not urls_in_report:
            return 0.0
        retrieved_urls = {entry.get("url") for entry in (context or [])}
        ok = sum(1 for u in urls_in_report if u in retrieved_urls)
        return ok / len(urls_in_report)


# ─── Runner ──────────────────────────────────────────────────────────────────


async def run_one(case: dict) -> tuple[str, list[dict]]:
    """Return (report, retrieval_trail)."""
    deps = make_deps()
    # We can't easily pass `deps` through `run_deep_research` and read back
    # the trail without re-constructing the call. Re-instantiate the inner
    # phases so the trail is observable.
    from agent import _run_research_phase, _run_scope, _run_write

    import fastaiagent as fa

    ctx = fa.RunContext(state=deps)
    brief = await _run_scope(case["input"], ctx)
    findings = await _run_research_phase(brief, ctx)
    report = await _run_write(brief, findings, ctx)
    return report, deps.trail


async def main() -> None:
    has_cites = HasMinCitations(min_count=2)
    cites_in_trail = CitationsInTrail()

    for case in CASES:
        print(f"\n=== {case['id']} ===")
        report, trail = await run_one(case)
        score_a = await has_cites.ascore(input=case["input"], actual_output=report)
        score_b = await cites_in_trail.ascore(
            input=case["input"], actual_output=report, context=trail
        )
        print(f"  has_min_citations:  {score_a:.2f}")
        print(f"  citations_in_trail: {score_b:.2f}")


if __name__ == "__main__":
    # Discovered Dataset usage is left for a follow-up; this script form
    # keeps the example runnable without a full eval-runner setup.
    _ = Dataset  # keeps the import meaningful for readers
    asyncio.run(main())
