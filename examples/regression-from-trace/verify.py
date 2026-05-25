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
    passed = sum(1 for s in judged if s.passed)
    total = len(judged)
    print()
    print(f"  llm_judge correctness: {passed}/{total} passed")
    for i, score in enumerate(judged):
        flag = "PASS" if score.passed else "FAIL"
        reason = (score.reason or "")[:140]
        print(f"   [{i}] {flag}  score={score.score:.2f}  {reason}")

    print()
    if passed == total:
        print("  Loop complete: every captured failure is now a passing regression test.")
        return 0
    print(
        "  Some cases still fail — investigate the rerun (fix.py) or refine the scoring criteria."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
