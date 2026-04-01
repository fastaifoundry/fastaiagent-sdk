"""Main evaluate() function."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import Scorer


def evaluate(
    agent_fn: Callable,
    dataset: Dataset | str | list[dict],
    scorers: list[Scorer | str] | None = None,
    concurrency: int = 4,
    **kwargs: Any,
) -> EvalResults:
    """Evaluate an agent function against a dataset with scorers.

    Example:
        results = evaluate(
            agent_fn=my_agent.run,
            dataset="test_cases.jsonl",
            scorers=["exact_match", "contains"],
        )
        print(results.summary())
    """
    return run_sync(
        aevaluate(agent_fn, dataset, scorers, concurrency, **kwargs)
    )


async def aevaluate(
    agent_fn: Callable,
    dataset: Dataset | str | list[dict],
    scorers: list[Scorer | str] | None = None,
    concurrency: int = 4,
    **kwargs: Any,
) -> EvalResults:
    """Async evaluation."""
    # Resolve dataset
    if isinstance(dataset, str):
        p = Path(dataset)
        if p.suffix == ".jsonl":
            ds = Dataset.from_jsonl(p)
        elif p.suffix == ".csv":
            ds = Dataset.from_csv(p)
        else:
            ds = Dataset.from_jsonl(p)
    elif isinstance(dataset, list):
        ds = Dataset.from_list(dataset)
    else:
        ds = dataset

    # Resolve scorers
    resolved_scorers: list[Scorer] = []
    for s in (scorers or ["exact_match"]):
        if isinstance(s, str):
            cls = BUILTIN_SCORERS.get(s)
            if cls:
                resolved_scorers.append(cls())
            else:
                raise ValueError(f"Unknown scorer: {s}")
        else:
            resolved_scorers.append(s)

    results = EvalResults()

    # Run evaluation
    sem = asyncio.Semaphore(concurrency)

    async def eval_one(item: dict) -> None:
        async with sem:
            input_text = item.get("input", str(item))
            expected = item.get("expected_output", item.get("expected"))

            # Call agent
            try:
                output = agent_fn(input_text)
                if asyncio.iscoroutine(output):
                    output = await output
                if hasattr(output, "output"):
                    output_text = output.output
                else:
                    output_text = str(output)
            except Exception as e:
                output_text = f"Error: {e}"

            # Score
            for scorer in resolved_scorers:
                result = scorer.score(
                    input=input_text,
                    output=output_text,
                    expected=expected,
                    **kwargs,
                )
                results.add(scorer.name, result)

    tasks = [eval_one(item) for item in ds]
    await asyncio.gather(*tasks)

    return results
