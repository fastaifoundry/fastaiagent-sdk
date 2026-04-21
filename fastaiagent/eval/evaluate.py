"""Main evaluate() function."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.errors import EvalError
from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import Scorer


def evaluate(
    agent_fn: Callable[..., Any],
    dataset: Dataset | str | list[dict[str, Any]],
    scorers: list[Scorer | str] | None = None,
    concurrency: int = 4,
    persist: bool = True,
    run_name: str | None = None,
    dataset_name: str | None = None,
    agent_name: str | None = None,
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

    By default the run is persisted to ``./.fastaiagent/local.db`` so the
    Local UI can surface it. Pass ``persist=False`` for ephemeral runs.
    """
    return run_sync(
        aevaluate(
            agent_fn,
            dataset,
            scorers,
            concurrency,
            persist=persist,
            run_name=run_name,
            dataset_name=dataset_name,
            agent_name=agent_name,
            **kwargs,
        )
    )


async def aevaluate(
    agent_fn: Callable[..., Any],
    dataset: Dataset | str | list[dict[str, Any]],
    scorers: list[Scorer | str] | None = None,
    concurrency: int = 4,
    persist: bool = True,
    run_name: str | None = None,
    dataset_name: str | None = None,
    agent_name: str | None = None,
    **kwargs: Any,
) -> EvalResults:
    """Async evaluation."""
    # Resolve dataset
    resolved_dataset_name: str | None = dataset_name
    if isinstance(dataset, str):
        p = Path(dataset)
        if resolved_dataset_name is None:
            resolved_dataset_name = p.name
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
    for s in scorers or ["exact_match"]:
        if isinstance(s, str):
            cls = BUILTIN_SCORERS.get(s)
            if cls:
                resolved_scorers.append(cls())
            else:
                available = ", ".join(sorted(BUILTIN_SCORERS.keys()))
                raise EvalError(
                    f"Unknown scorer '{s}'. "
                    f"Available built-in scorers: {available}. "
                    f"Or pass a Scorer instance directly."
                )
        else:
            resolved_scorers.append(s)

    results = EvalResults()

    # Run evaluation
    sem = asyncio.Semaphore(concurrency)

    async def eval_one(item: dict[str, Any]) -> None:
        async with sem:
            input_text = item.get("input", str(item))
            expected = item.get("expected_output", item.get("expected"))

            trace_id: str | None = None
            # Call agent
            try:
                output = agent_fn(input_text)
                if asyncio.iscoroutine(output):
                    output = await output
                if hasattr(output, "output"):
                    output_text = output.output
                else:
                    output_text = str(output)
                trace_id = getattr(output, "trace_id", None)
            except Exception as e:
                output_text = f"Error: {e}"

            # Score
            per_scorer: dict[str, dict[str, Any]] = {}
            for scorer in resolved_scorers:
                result = scorer.score(
                    input=input_text,
                    output=output_text,
                    expected=expected,
                    **kwargs,
                )
                results.add(scorer.name, result)
                per_scorer[scorer.name] = {
                    "passed": bool(result.passed),
                    "score": float(result.score),
                    "reason": result.reason,
                }

            results.add_case(
                EvalCaseRecord(
                    input=input_text,
                    expected_output=expected,
                    actual_output=output_text,
                    trace_id=trace_id,
                    per_scorer=per_scorer,
                )
            )

    tasks = [eval_one(item) for item in ds]
    await asyncio.gather(*tasks)

    if persist:
        try:
            run_id = results.persist_local(
                run_name=run_name,
                dataset_name=resolved_dataset_name,
                agent_name=agent_name,
            )
            # Stash the run_id on the returned object so callers can deep-link
            # into the Local UI (e.g. /evals/<run_id> or /evals/compare?a=…).
            results.run_id = run_id
        except Exception:
            # Persistence is best-effort — never fail a run because the
            # local UI DB couldn't be written.
            pass

    return results
