"""Example 78: Evaluate a REAL vision agent over a multimodal dataset.

Demonstrates (real Agent + real vision LLM — needs OPENAI_API_KEY):
- Dataset.from_jsonl resolving typed multimodal `input` parts (text + image)
  into real Image objects at load time.
- evaluate() driving a real vision agent over that dataset and scoring its
  real output with `contains`.

The dataset item's `input` is a list of parts; `evaluate()` passes that list
straight to `agent.run`, so no transformation is needed.

Run:
    zsh -lc 'python examples/78_multimodal_eval.py'
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import Dataset, evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "multimodal"


def _ensure_fixture() -> Path:
    """Generate the 'CAT' test image on a fresh checkout if needed."""
    cat = FIXTURES / "cat.jpg"
    if not cat.exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()
    return cat


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    cat = _ensure_fixture()

    with tempfile.TemporaryDirectory() as d:
        jsonl = Path(d) / "multimodal_cases.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "input": [
                        {
                            "type": "text",
                            "text": "What letters appear? Reply with just the letters.",
                        },
                        {"type": "image", "path": str(cat)},
                    ],
                    "expected": "CAT",
                }
            )
            + "\n"
        )

        ds = Dataset.from_jsonl(jsonl)
        print(
            f"Loaded {len(ds)} multimodal case(s); input parts: "
            f"{[type(p).__name__ for p in ds[0]['input']]}"
        )

        agent = Agent(
            name="vision-eval",
            system_prompt="You are a vision assistant. If you see text, quote it exactly.",
            llm=LLMClient(provider="openai", model="gpt-4o"),
        )

        results = evaluate(
            agent_fn=agent.run,
            dataset=ds,
            scorers=["contains"],
            persist=False,
        )
        print(results.summary())
        for name, scores in results.scores.items():
            for s in scores:
                status = "PASS" if s.passed else "FAIL"
                print(f"  [{status}] {name}: score={s.score:.2f}")


if __name__ == "__main__":
    main()
