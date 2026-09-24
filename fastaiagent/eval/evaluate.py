"""Main evaluate() function."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.errors import EvalError
from fastaiagent._internal.pause import describe_pause
from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import Scorer

logger = logging.getLogger(__name__)


class _CasePaused(Exception):  # noqa: N818 — internal; recorded as the case's error
    """The agent paused instead of answering (see :func:`describe_pause`)."""


def _case_error(exc: BaseException) -> str:
    """The recorded error for a case that produced no scorable output. A bare
    ``InterruptSignal`` (an agent with no checkpointer pausing) is described as
    the pause it is, not by its reason string alone."""
    return (describe_pause(exc) or str(exc))[:500]


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    """Whether calling ``fn`` returns an awaitable without doing blocking work.

    Covers plain ``async def``, an object whose ``__call__`` is ``async def``,
    and ``functools.partial`` wrapping either.
    """
    import functools
    import inspect

    target: Any = fn
    while isinstance(target, functools.partial):
        target = target.func
    if inspect.iscoroutinefunction(target):
        return True
    call = getattr(target, "__call__", None)  # noqa: B004
    return call is not None and inspect.iscoroutinefunction(call)


def infer_agent_name(agent_fn: Callable[..., Any]) -> str | None:
    """Best-effort agent name for a callable like ``agent.run``.

    Load-bearing for connected mode: the plane resolves ``agent_name`` to a real
    agent so eval evidence attaches to a *specific* system. Without it every run
    lands unattributed and per-agent marking silently does nothing.

    Handles the common shapes — a bound method of an ``Agent`` / ``Supervisor`` /
    ``Swarm`` / ``Chain`` (``agent.run``), a ``functools.partial`` around one, and
    an object exposing ``.name`` directly. Returns None when there is nothing
    trustworthy to report; a wrong name is worse than none.
    """
    import functools

    target: Any = agent_fn
    while isinstance(target, functools.partial):
        target = target.func
    owner = getattr(target, "__self__", None)  # bound method -> its instance
    if owner is None and not callable(target):
        owner = target
    name = getattr(owner, "name", None) if owner is not None else None
    return name if isinstance(name, str) and name else None


async def _call_agent_fn(agent_fn: Callable[..., Any], input_text: Any) -> Any:
    """Invoke a user agent callable from inside the eval loop.

    A **synchronous** ``agent_fn`` is run on a worker thread rather than
    directly on the event loop. Two reasons, both load-bearing:

    * Calling it inline blocks the loop for the whole agent run, which silently
      defeated the ``concurrency`` semaphore — sync callables serialized no
      matter what concurrency was requested.
    * Frameworks increasingly *refuse* to run synchronously inside a live loop.
      CrewAI ≥1.15 raises "Agent execution was invoked synchronously from
      within a running event loop. Use ``kickoff_async()``" rather than doing
      it anyway, which turned every sync-driven eval case into an errored one.

    ``asyncio.to_thread`` propagates the current ``contextvars`` copy, so OTel
    span context still flows into the thread and ``trace_id`` capture works.
    """
    if _is_async_callable(agent_fn):
        return await agent_fn(input_text)
    output = await asyncio.to_thread(agent_fn, input_text)
    # A sync callable is still allowed to hand back a coroutine; awaiting it
    # here (back on the loop) preserves the previous contract.
    if asyncio.iscoroutine(output):
        output = await output
    return output


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
            # What the run cost and how long it took. Collected here because
            # ``evaluate`` is the only thing holding the result, and because two
            # documented budget gates were reading them out of ``**kwargs`` that
            # nobody ever filled: ``CostUnder`` and ``Latency`` were
            # structurally incapable of failing.
            run_facts: dict[str, Any] = {}
            call_started = time.monotonic()
            # Call agent
            try:
                output = await _call_agent_fn(agent_fn, input_text)
                # A paused run (approval policy / interrupt()) has no answer yet:
                # its ``output`` is "". Scoring that graded a run that never
                # finished as a wrong answer; it is errored (unscored) instead.
                paused = describe_pause(output)
                if paused:
                    raise _CasePaused(paused)
                if hasattr(output, "output"):
                    out_val = output.output
                else:
                    out_val = output
                # Coerce a missing output to "" so scorers don't crash on None;
                # preserve any real (possibly non-str) output unchanged.
                output_text = "" if out_val is None else out_val
                trace_id = getattr(output, "trace_id", None)
                # Latency is always knowable — measured here when the callable
                # returns something that isn't an AgentResult (a bare string,
                # a dict), so the gate never has to guess.
                measured_ms = int((time.monotonic() - call_started) * 1000)
                run_facts["latency_ms"] = int(getattr(output, "latency_ms", 0) or measured_ms)
                # Cost is NOT always knowable, and "unknown" must not read as
                # "free" — see ``AgentResult.cost_known``. A callable that
                # returns a bare string reports neither, so the gate is told the
                # cost is unknown rather than zero.
                if hasattr(output, "cost"):
                    run_facts["cost"] = float(getattr(output, "cost", 0.0) or 0.0)
                    run_facts["cost_known"] = bool(getattr(output, "cost_known", False))
                else:
                    run_facts["cost"] = 0.0
                    run_facts["cost_known"] = False
            except Exception as e:
                # Infrastructure failure DURING scoring (provider 500, timeout,
                # network/auth error) is NOT an agent-quality miss. Record the case
                # as errored (non-signal) and do not score it — so it can't count as
                # a failure the optimizer would try to "fix".
                results.add_case(
                    EvalCaseRecord(
                        input=input_text,
                        expected_output=expected,
                        actual_output=None,
                        trace_id=None,
                        per_scorer={},
                        error=_case_error(e),
                    )
                )
                return

            # Score
            per_scorer: dict[str, dict[str, Any]] = {}
            # Caller kwargs win: someone passing ``cost=`` to ``evaluate()``
            # is overriding on purpose, and always could.
            score_kwargs = {**run_facts, **kwargs}
            for scorer in resolved_scorers:
                result = scorer.score(
                    input=input_text,
                    output=output_text,
                    expected=expected,
                    **score_kwargs,
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
                # Fall back to the callable's own agent so connected runs attach
                # to a real system instead of landing unattributed.
                agent_name=agent_name or infer_agent_name(agent_fn),
            )
            # Stash the run_id on the returned object so callers can deep-link
            # into the Local UI (e.g. /evals/<run_id> or /evals/compare?a=…).
            results.run_id = run_id
        except Exception:
            # Persistence is best-effort — never fail a run because the
            # local UI DB couldn't be written.
            logger.warning("Failed to persist eval run to local DB", exc_info=True)

    return results
