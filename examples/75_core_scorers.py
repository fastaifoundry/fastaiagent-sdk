"""Example 75: The seven built-in core scorers on a REAL agent run.

Demonstrates:
- contains        : expected substring appears in the output
- length_between  : output length within [min_len, max_len]
- latency         : real run latency under a threshold (from AgentResult.latency_ms)
- cost_under      : real run cost under a budget (from AgentResult.cost)
- json_valid      : output parses as JSON (agent uses output_type → valid JSON)
- regex_match     : output matches a regex pattern
- exact_match     : output equals expected exactly (a constrained classification label)
- Dataset.from_list / from_jsonl / from_csv loading (no API key needed)

`latency`/`cost_under` read their measurement from kwargs — we pass the real
values off the AgentResult. `regex_match`/`length_between` take constructor args.

Run:
    zsh -lc 'python examples/75_core_scorers.py'   # full demo (needs OPENAI_API_KEY)
    python examples/75_core_scorers.py             # dataset-loading section only
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import Dataset
from fastaiagent.eval.builtins import (
    Contains,
    CostUnder,
    ExactMatch,
    JSONValid,
    Latency,
    LengthBetween,
    RegexMatch,
)


class CityInfo(BaseModel):
    """Structured output — guarantees valid JSON for the json_valid demo."""

    city: str
    country: str


def _show(name: str, result) -> None:
    status = "PASS" if result.passed else "FAIL"
    reason = f" — {result.reason}" if result.reason else ""
    print(f"  [{status}] {name:<16} score={result.score:.2f}{reason}")


def demo_real_agent() -> None:
    """Score a real agent's real output and real run metrics."""
    print("== Core scorers on a real agent run ==")
    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    agent = Agent(
        name="core-scorers-demo",
        system_prompt="You are a concise assistant. Follow the user's format exactly.",
        llm=llm,
    )

    # 1) Free-form answer — substring, length, and real latency/cost.
    r1 = agent.run("In one short sentence, what is the capital of France?")
    print(f"  q1: {r1.output!r}  ({r1.latency_ms} ms, ${r1.cost:.5f})")
    _show("contains", Contains().score(input="", output=r1.output, expected="Paris"))
    _show("length_between", LengthBetween(min_len=5, max_len=200).score(input="", output=r1.output))
    _show(
        "latency", Latency(max_ms=20000).score(input="", output=r1.output, latency_ms=r1.latency_ms)
    )
    _show("cost_under", CostUnder(max_usd=0.01).score(input="", output=r1.output, cost=r1.cost))

    # 2) Structured JSON via output_type — guaranteed-valid JSON to score.
    json_agent = Agent(
        name="json-demo",
        system_prompt="Return the requested fields.",
        llm=llm,
        output_type=CityInfo,
    )
    r2 = json_agent.run("What is the capital of France? Give the city and its country.")
    print(f"  q2: {r2.output!r}")
    _show("json_valid", JSONValid().score(input="", output=r2.output))
    _show("regex_match", RegexMatch(pattern=r'"country"').score(input="", output=r2.output))

    # 3) Constrained classification label — the natural fit for exact_match.
    r3 = agent.run(
        "Classify the sentiment of this review. Reply with EXACTLY one lowercase word "
        "(positive, negative, or neutral) and nothing else: 'I absolutely love this!'"
    )
    print(f"  q3: {r3.output!r}")
    _show("exact_match", ExactMatch().score(input="", output=r3.output, expected="positive"))


def demo_dataset_loading() -> None:
    """Dataset.from_list / from_jsonl / from_csv all yield the same shape (no key)."""
    print("\n== Dataset loading ==")
    rows = [
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "Capital of France?", "expected": "Paris"},
    ]
    with tempfile.TemporaryDirectory() as d:
        jsonl_path = Path(d) / "cases.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in rows))
        csv_path = Path(d) / "cases.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "expected"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"  from_list:  {len(Dataset.from_list(rows))} cases")
        print(f"  from_jsonl: {len(Dataset.from_jsonl(jsonl_path))} cases")
        print(f"  from_csv:   {len(Dataset.from_csv(csv_path))} cases")


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        demo_real_agent()
    else:
        print("(Set OPENAI_API_KEY to score a real agent run; showing dataset loading only.)")
    demo_dataset_loading()
