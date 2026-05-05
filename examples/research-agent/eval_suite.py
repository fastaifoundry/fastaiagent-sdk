"""
Evaluation Suite — quality testing for the Research Agent.

Run: python eval_suite.py
     python eval_suite.py --publish     # publish to FastAIAgent Platform

Three scoring dimensions:

  * **AnswerRelevancy** (built-in v1.0)         — does the report address
    the topic? No context required.

  * **Faithfulness** (built-in v1.0)            — is every claim grounded
    in the retrieved sources? We pass the researcher's snippets as context.

  * **RequiredSources** (custom CodeScorer)     — does the report cite at
    least one URL from the curated must-include list for this topic? This
    is what guards against the verifier loop merely producing a *cited*
    report that nonetheless missed the canonical source.

The report+sources for each case are produced by running the supervisor
once per case in ``_collect_run_artifacts``; the per-case context and
expected URLs are then forwarded to the scorers.
"""

from __future__ import annotations

import argparse
import asyncio
import re

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent.eval import AnswerRelevancy, Faithfulness
from fastaiagent.eval.scorer import Scorer, ScorerResult

from tools import make_deps
from topology import build_supervisor


# ─── Eval dataset ────────────────────────────────────────────────────────────

EVAL_CASES: list[dict] = [
    {
        "input": "Transformer architecture",
        "expected": (
            "Should mention attention, multi-head attention, positional "
            "encoding, and cite the 2017 'Attention Is All You Need' paper."
        ),
        "required_sources": ["https://arxiv.org/abs/1706.03762"],
    },
    {
        "input": "Retrieval-augmented generation",
        "expected": (
            "Should describe combining a retriever (e.g., DPR) with a "
            "seq2seq generator and cite the 2020 RAG paper."
        ),
        "required_sources": ["https://arxiv.org/abs/2005.11401"],
    },
    {
        "input": "Constitutional AI",
        "expected": (
            "Should describe self-critique against a written set of "
            "principles and cite Bai et al. 2022."
        ),
        "required_sources": ["https://arxiv.org/abs/2212.08073"],
    },
    {
        "input": "Agent eval",
        "expected": (
            "Should mention multi-environment benchmarks like AgentBench "
            "or tau-bench."
        ),
        "required_sources": ["https://arxiv.org/abs/2308.03688"],
    },
]


# ─── Custom scorer: required-sources coverage ────────────────────────────────


class RequiredSourcesScorer(Scorer):
    """Pass iff every URL in ``required_sources`` appears in the report.

    Score = (URLs found) / (URLs required). ``required_sources`` is
    forwarded per-case via ``Dataset.from_list(...)`` items — but the
    built-in ``evaluate()`` only forwards top-level kwargs to scorers, so
    this scorer reads the field from a closure populated alongside the
    per-case agent_fn (see ``_run_case`` below).
    """

    name = "required_sources"

    def __init__(self, required_for_case: dict[str, list[str]], threshold: float = 1.0):
        self.required_for_case = required_for_case
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw
    ) -> ScorerResult:
        required = self.required_for_case.get(input, [])
        if not required:
            return ScorerResult(
                score=1.0, passed=True, reason="no required sources for this case"
            )
        found = [url for url in required if url in output]
        ratio = len(found) / len(required)
        return ScorerResult(
            score=round(ratio, 4),
            passed=ratio >= self.threshold,
            reason=f"matched {len(found)}/{len(required)} required URLs",
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────


_URL_RE = re.compile(r"https?://\S+")


def _extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,)") for u in _URL_RE.findall(text)]


# ─── Runner ──────────────────────────────────────────────────────────────────


async def run_eval(publish: bool = False) -> None:
    """Run each case through the supervisor, then score.

    Faithfulness needs *the retrieved snippets as context* — those live in
    ``deps.trail`` after the supervisor finishes. We accumulate per-case
    context strings and feed them in via a closure-wrapping agent_fn that
    pre-populates the scorer's kwargs through ``contexts``.

    The same closure populates the ``required_sources`` map the custom
    scorer reads.
    """
    supervisor = build_supervisor()

    # We collect per-case eval inputs (output, retrieved_context) by running
    # the supervisor once up front — separately from evaluate() — because the
    # built-in evaluate() forwards top-level kwargs uniformly to every case
    # and we need per-case context.
    print("\nPhase 1 — running supervisor on every case to capture artifacts...\n")
    artifacts: dict[str, dict] = {}
    for case in EVAL_CASES:
        topic = case["input"]
        deps = make_deps()  # fresh trail per case
        ctx = fa.RunContext(state=deps)
        result = await supervisor.arun(topic, context=ctx)
        retrieved_context = "\n\n".join(
            f"{r['title']} — {r['url']}\n{r['snippet']}" for r in deps.trail
        )
        artifacts[topic] = {
            "output": result.output,
            "context": retrieved_context,
            "trail_urls": [r["url"] for r in deps.trail],
        }
        print(f"  • {topic:<35} retrieved={len(deps.trail)} urls in report={len(_extract_urls(result.output))}")

    # Per-case ``required_sources`` lookup the custom scorer reads.
    required_for_case = {c["input"]: c["required_sources"] for c in EVAL_CASES}

    answer_relevancy = AnswerRelevancy()
    required_sources = RequiredSourcesScorer(required_for_case=required_for_case)

    # ── Score each case manually so we can pass per-case context. ──
    # ``evaluate()`` is great when context is uniform; here it isn't, so we
    # call the scorers directly and roll our own averages — same shape as
    # what ``EvalResults.summary()`` would print.
    print("\nPhase 2 — scoring...\n")
    rows: list[dict] = []
    for case in EVAL_CASES:
        topic = case["input"]
        art = artifacts[topic]
        output = art["output"]
        context = art["context"]
        expected = case.get("expected")

        ar = await answer_relevancy.ascore(input=topic, output=output, expected=expected)
        # Faithfulness wants ``context`` as a single string; use its async path
        # for cheaper batching.
        faith = await Faithfulness().ascore(
            input=topic, output=output, expected=expected, context=context
        )
        rs = required_sources.score(input=topic, output=output, expected=expected)

        rows.append(
            {
                "topic": topic,
                "answer_relevancy": ar.score,
                "faithfulness": faith.score,
                "required_sources": rs.score,
            }
        )
        print(
            f"  {topic:<35} "
            f"ar={ar.score:.2f}  faith={faith.score:.2f}  req={rs.score:.2f}"
        )

    print("\nEvaluation Results")
    print("=" * 50)
    for metric in ("answer_relevancy", "faithfulness", "required_sources"):
        avg = sum(r[metric] for r in rows) / len(rows)
        passed = sum(1 for r in rows if r[metric] >= 0.5)
        print(f"  {metric:<18}  avg={avg:.2f}  pass_rate={passed}/{len(rows)}")

    if publish:
        # Best-effort: rebuild as an EvalResults and publish.
        from fastaiagent.eval.results import EvalCaseRecord, EvalResults

        results = EvalResults()
        for case, row in zip(EVAL_CASES, rows):
            results.add(
                "answer_relevancy",
                ScorerResult(score=row["answer_relevancy"], passed=row["answer_relevancy"] >= 0.7),
            )
            results.add(
                "faithfulness",
                ScorerResult(score=row["faithfulness"], passed=row["faithfulness"] >= 0.7),
            )
            results.add(
                "required_sources",
                ScorerResult(score=row["required_sources"], passed=row["required_sources"] >= 1.0),
            )
            results.add_case(
                EvalCaseRecord(
                    input=case["input"],
                    expected_output=case.get("expected"),
                    actual_output=artifacts[case["input"]]["output"],
                    per_scorer={
                        "answer_relevancy": {"score": row["answer_relevancy"], "passed": row["answer_relevancy"] >= 0.7},
                        "faithfulness": {"score": row["faithfulness"], "passed": row["faithfulness"] >= 0.7},
                        "required_sources": {"score": row["required_sources"], "passed": row["required_sources"] >= 1.0},
                    },
                )
            )
        try:
            results.publish()
            print("\nResults published to FastAIAgent Platform")
        except Exception as e:
            print(f"\nPublish failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="Publish to platform")
    args = parser.parse_args()
    asyncio.run(run_eval(publish=args.publish))


if __name__ == "__main__":
    main()
