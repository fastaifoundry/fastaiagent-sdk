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
    # We score each case manually so we can pass per-case context, but we
    # still build an ``EvalResults`` and persist it to the local DB. That's
    # what makes the run show up in the Local UI's /evals page alongside
    # ``customer-support-agent``'s eval runs.
    from fastaiagent.eval.results import EvalCaseRecord, EvalResults

    print("\nPhase 2 — scoring...\n")
    results = EvalResults()
    rows: list[dict] = []
    for case in EVAL_CASES:
        topic = case["input"]
        art = artifacts[topic]
        output = art["output"]
        context = art["context"]
        expected = case.get("expected")

        ar = await answer_relevancy.ascore(input=topic, output=output, expected=expected)
        # Faithfulness wants ``context`` as a single string; use its async
        # path for cheaper batching.
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
        results.add("answer_relevancy", ar)
        results.add("faithfulness", faith)
        results.add("required_sources", rs)
        results.add_case(
            EvalCaseRecord(
                input=topic,
                expected_output=expected,
                actual_output=output,
                per_scorer={
                    "answer_relevancy": {"score": ar.score, "passed": ar.passed, "reason": ar.reason},
                    "faithfulness": {"score": faith.score, "passed": faith.passed, "reason": faith.reason},
                    "required_sources": {"score": rs.score, "passed": rs.passed, "reason": rs.reason},
                },
            )
        )
        print(f"  {topic:<35} ar={ar.score:.2f}  faith={faith.score:.2f}  req={rs.score:.2f}")

    print()
    print(results.summary())

    # ── Persist to the unified local.db so the run shows up at /evals ──
    try:
        run_id = results.persist_local(
            run_name="research-agent eval",
            dataset_name="research-topics-golden",
            agent_name="research-team",
        )
        print(f"\n  ✓ persisted to local.db (run_id={run_id[:12]}...)")
        print("  ✓ open `fastaiagent ui` and visit /evals/" + run_id[:12] + "...")
    except Exception as e:
        print(f"\n  Could not persist eval run locally: {e}")

    if publish:
        try:
            results.publish(run_name="research-agent eval")
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
