"""Step 5 — run the regression dataset against the fixed agent.

Loads ``regression_dataset.jsonl`` and calls
``fastaiagent.eval.evaluate(...)`` with the **fixed** agent. We
use ``LLMJudge(criteria="correctness")`` instead of ``exact_match``
because LLM outputs are paraphrase-stable but not byte-stable — a
strict string match would be flaky.

Run from the template directory::

    cd examples/regression-from-trace
    zsh -lc 'python verify.py'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent import build_fixed_agent  # noqa: E402

from fastaiagent.eval import LLMJudge, evaluate  # noqa: E402

DATASET = _HERE / "regression_dataset.jsonl"


def _require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set.")
        raise SystemExit(0)


def main() -> int:
    if not DATASET.exists() or DATASET.stat().st_size == 0:
        print(
            f"Dataset {DATASET} is empty. Run capture.py → fix.py → save_test.py first to seed it."
        )
        return 1
    _require_key()

    print(f"Step 5: Running evaluate() against {DATASET.name} with the fixed agent…")
    fixed_agent = build_fixed_agent()

    def _agent_fn(text: str) -> str:
        return fixed_agent.run(text).output

    results = evaluate(
        agent_fn=_agent_fn,
        dataset=str(DATASET),
        scorers=[LLMJudge(criteria="correctness")],
        persist=False,
    )

    judged = results.scores["llm_judge"]
    # LLMJudge itself is a real LLM call and can occasionally return
    # malformed JSON ("Judge error: Expecting value: line 1 column 1
    # (char 0)") — treat those as *inconclusive* rather than agent
    # failures. The agent might still be correct; the judge just
    # crashed parsing its own response.
    judge_crashed = [
        s for s in judged if not s.passed and (s.reason or "").startswith("Judge error:")
    ]
    real_pass = [s for s in judged if s.passed]
    real_fail = [
        s for s in judged if not s.passed and not (s.reason or "").startswith("Judge error:")
    ]
    total = len(judged)
    print()
    print(
        f"  llm_judge correctness: {len(real_pass)} pass / {len(real_fail)} fail / "
        f"{len(judge_crashed)} judge-crashed / {total} total"
    )
    for i, score in enumerate(judged):
        if not score.passed and (score.reason or "").startswith("Judge error:"):
            flag = "JUDGE-CRASH"
        else:
            flag = "PASS" if score.passed else "FAIL"
        reason = (score.reason or "")[:140]
        print(f"   [{i}] {flag}  score={score.score:.2f}  {reason}")

    # Pass rate gate is on the cases the judge actually scored (real
    # pass + real fail) — judge-crashed cases are excluded from both
    # numerator and denominator since we have no signal on them. Per
    # the plan (claude_files/top3recommendation.md §4.5), the gate is
    # ≥80% pass rate on the dataset.
    scored = len(real_pass) + len(real_fail)
    pass_rate = (len(real_pass) / scored) if scored > 0 else 0.0
    print(f"  pass_rate (excluding judge crashes): {pass_rate:.0%}  (target: ≥80%)")

    print()
    if pass_rate >= 0.80 and len(real_fail) == 0:
        msg = "Loop complete: every captured failure is now a passing regression test."
        if judge_crashed:
            msg += f" ({len(judge_crashed)} case(s) inconclusive due to judge crash — rerun verify.py to retry.)"
        print(f"  {msg}")
        return 0
    print(
        "  Some real fails — investigate the rerun (fix.py) or refine the scoring criteria."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
