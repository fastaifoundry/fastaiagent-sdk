"""Pytest plugin for fastaiagent's evaluation framework.

Registered via the ``[project.entry-points.pytest11]`` group in
``pyproject.toml`` so any project that has ``fastaiagent`` installed picks
it up automatically.

What you get:

- A :func:`case` decorator — turn a regular pytest function into a single
  evaluation case with ``input`` and ``expected``. The fixture
  ``evaluate_one`` exposes a one-call helper that runs the agent and
  scores against the expected output.

- A :func:`dataset` decorator — parametrise a test over every row of a
  JSONL or CSV dataset (uses :class:`fastaiagent.eval.Dataset`).

- ``evaluate_one`` fixture — runs a single eval inline, returns an
  :class:`fastaiagent.eval.results.EvalCaseRecord`, asserts pass on
  ``exact_match`` by default unless overridden.

- Auto-persist: each tagged test creates one ``eval_runs`` row in the
  local database, tagged ``run_name="pytest::<module>::<test>"``, so CI
  eval results show up in the local UI's ``/evals`` page.

Example:

    from fastaiagent.testing import TestModel
    from fastaiagent.agent import Agent
    from fastaiagent.eval import case

    @case(input="hello", expected="hi")
    def test_greet(evaluate_one):
        agent = Agent(name="g", llm=TestModel(response="hi"))
        evaluate_one(agent.run, scorers=["exact_match"])

The plugin is opt-in: tests that don't import any of these helpers are
unaffected.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import Scorer

logger = logging.getLogger(__name__)

# Marker name we attach via raw setattr so the plugin can find decorated
# tests at fixture time. We avoid pytest.mark to keep the surface explicit.
_CASE_ATTR = "_fastaiagent_case"


def case(
    *,
    input: Any,
    expected: Any | None = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a pytest function as a single fastaiagent eval case.

    The ``evaluate_one`` fixture inside the test reads the tag and feeds
    ``input`` / ``expected`` automatically.

    Args:
        input: Whatever you'd pass to your agent's ``run()``.
        expected: Reference answer used by scorers like ``exact_match``.
        name: Optional case name (defaults to the test function name).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            fn,
            _CASE_ATTR,
            {"input": input, "expected": expected, "name": name or fn.__name__},
        )
        return fn

    return decorator


def dataset(
    path: str | Path,
    *,
    ids_from: str = "input",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Parametrise a test over every row of a JSONL/CSV dataset.

    Each row becomes a separate pytest invocation. The test signature must
    accept an ``eval_case`` argument (a dict with ``input`` and ``expected``
    keys); use the ``evaluate_one`` fixture to score against it.

    Args:
        path: Path to a ``.jsonl`` or ``.csv`` file.
        ids_from: Field used to label parametrised cases in pytest output.
    """
    p = Path(path)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if p.suffix == ".csv":
            ds = Dataset.from_csv(p)
        else:
            ds = Dataset.from_jsonl(p)
        cases = list(ds)
        ids = [str(c.get(ids_from, i)) for i, c in enumerate(cases)]
        wrapped: Callable[..., Any] = pytest.mark.parametrize(
            "eval_case", cases, ids=ids
        )(fn)
        return wrapped

    return decorator


def _resolve_scorers(scorers: list[Scorer | str] | None) -> list[Scorer]:
    """Mirror ``aevaluate``'s scorer resolution but expose a clean error."""
    out: list[Scorer] = []
    for s in scorers or ["exact_match"]:
        if isinstance(s, str):
            cls = BUILTIN_SCORERS.get(s)
            if cls is None:
                pytest.fail(
                    f"Unknown scorer '{s}'. Available: "
                    f"{', '.join(sorted(BUILTIN_SCORERS))}."
                )
            out.append(cls())
        else:
            out.append(s)
    return out


def _format_failure(case_record: EvalCaseRecord) -> str:
    lines = [
        "fastaiagent eval case failed.",
        f"  input:    {case_record.input!r}",
        f"  expected: {case_record.expected_output!r}",
        f"  actual:   {case_record.actual_output!r}",
        "  scorers:",
    ]
    for name, info in (case_record.per_scorer or {}).items():
        passed = info.get("passed")
        score = info.get("score")
        reason = info.get("reason")
        lines.append(f"    - {name}: passed={passed} score={score} reason={reason}")
    return "\n".join(lines)


@pytest.fixture
def evaluate_one(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    """Run a single agent invocation and score it against ``expected``.

    Reads the ``@case(input=..., expected=...)`` tag from the calling test
    or falls back to a parametrised ``eval_case`` arg from ``@dataset(...)``.

    Persists one ``eval_runs`` row per test invocation, tagged
    ``run_name="pytest::<test-id>"``, so CI results appear in the local UI.

    Returns the helper as a callable so the test body retains control over
    timing, error handling, and any assertions on the result besides the
    eval pass/fail.
    """
    test_fn = request.function
    case_meta = getattr(test_fn, _CASE_ATTR, None)
    eval_case = (
        request.getfixturevalue("eval_case") if "eval_case" in request.fixturenames else None
    )

    def _run(
        agent_fn: Callable[..., Any],
        *,
        input: Any | None = None,
        expected: Any | None = None,
        scorers: list[Scorer | str] | None = None,
        assert_pass: bool = True,
        case_name: str | None = None,
        persist: bool = True,
    ) -> EvalCaseRecord:
        # Resolve input / expected: explicit args > @case tag > @dataset row.
        in_text = input
        exp = expected
        if in_text is None and case_meta:
            in_text = case_meta.get("input")
        if exp is None and case_meta:
            exp = case_meta.get("expected")
        if in_text is None and isinstance(eval_case, dict):
            in_text = eval_case.get("input")
        if exp is None and isinstance(eval_case, dict):
            exp = eval_case.get("expected_output", eval_case.get("expected"))
        if in_text is None:
            pytest.fail(
                "evaluate_one: no input. Use @case(input=..., expected=...) "
                "or @dataset(...) on the test, or pass input= explicitly."
            )

        sig = inspect.signature(agent_fn)
        try:
            output = agent_fn(in_text) if len(sig.parameters) >= 1 else agent_fn()
        except Exception as e:  # surface as a failed case rather than a hard error
            output = f"Error: {e}"

        if hasattr(output, "output"):
            output_text = output.output
        else:
            output_text = str(output)
        trace_id = getattr(output, "trace_id", None)

        scorer_objs = _resolve_scorers(scorers)
        results = EvalResults()
        per_scorer: dict[str, dict[str, Any]] = {}
        all_passed = True
        for scorer in scorer_objs:
            result = scorer.score(input=str(in_text), output=output_text, expected=exp)
            results.add(scorer.name, result)
            per_scorer[scorer.name] = {
                "passed": bool(result.passed),
                "score": float(result.score),
                "reason": result.reason,
            }
            all_passed = all_passed and bool(result.passed)

        record = EvalCaseRecord(
            input=in_text,
            expected_output=exp,
            actual_output=output_text,
            trace_id=trace_id,
            per_scorer=per_scorer,
        )
        results.add_case(record)

        if persist:
            try:
                run_name = f"pytest::{request.node.nodeid}"
                if case_name:
                    run_name = f"{run_name}::{case_name}"
                results.persist_local(
                    run_name=run_name,
                    dataset_name=None,
                    agent_name=None,
                )
            except Exception:  # pragma: no cover — non-fatal
                logger.warning(
                    "Failed to persist pytest eval run for %s",
                    request.node.nodeid,
                    exc_info=True,
                )

        if assert_pass and not all_passed:
            pytest.fail(_format_failure(record))
        return record

    return _run
