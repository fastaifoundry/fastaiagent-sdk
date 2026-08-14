"""A synchronous ``agent_fn`` must not be executed on the eval event loop.

``evaluate()`` drives cases through an async loop. It used to call a sync
``agent_fn`` inline, so the user's agent ran *on the loop thread*. That had two
consequences:

* the ``concurrency`` semaphore was defeated — sync callables serialized;
* frameworks that refuse to run synchronously inside a live loop errored out.
  CrewAI ≥1.15 raises "Agent execution was invoked synchronously from within a
  running event loop. Use ``kickoff_async()``", which turned every eval case
  into an errored one.

These tests pin the contract with plain Python — no framework needed. The
CrewAI path itself is covered by the e2e harness.
"""

from __future__ import annotations

import asyncio
import threading
import time

from fastaiagent.eval.evaluate import evaluate


def test_sync_agent_fn_does_not_run_on_the_event_loop() -> None:
    """The exact condition CrewAI ≥1.15 refuses on."""
    seen: list[bool] = []

    def agent(text: str) -> str:
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return "Paris"

    res = evaluate(
        agent_fn=agent,
        dataset=[{"input": "capital of France?", "expected_output": "Paris"}],
        scorers=["exact_match"],
        persist=False,
    )
    assert seen == [False], "sync agent_fn ran inside a running event loop"
    assert res.cases[0].actual_output == "Paris"
    assert res.cases[0].error is None


def test_sync_agent_fn_that_refuses_to_run_in_a_loop_succeeds() -> None:
    """Simulates CrewAI's guard verbatim — this is what CI hit."""

    def crewai_like(text: str) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return "Paris"
        raise RuntimeError(
            "Agent execution was invoked synchronously from within a running "
            "event loop. Use `agent.kickoff_async()` / `crew.kickoff_async()`."
        )

    res = evaluate(
        agent_fn=crewai_like,
        dataset=[{"input": "capital of France?", "expected_output": "Paris"}],
        scorers=["exact_match"],
        persist=False,
    )
    case = res.cases[0]
    assert case.error is None, f"agent errored: {case.error}"
    assert case.actual_output == "Paris"


def test_sync_agent_fn_actually_runs_concurrently() -> None:
    """Inline execution silently serialized sync callables regardless of
    ``concurrency``; offloading to threads makes the semaphore mean something."""
    threads: set[int] = set()
    barrier_hits = []

    def slow(text: str) -> str:
        threads.add(threading.get_ident())
        barrier_hits.append(time.monotonic())
        time.sleep(0.3)
        return "Paris"

    cases = [{"input": f"q{i}", "expected_output": "Paris"} for i in range(4)]
    start = time.monotonic()
    res = evaluate(
        agent_fn=slow, dataset=cases, scorers=["exact_match"], persist=False, concurrency=4
    )
    elapsed = time.monotonic() - start

    assert len(res.cases) == 4
    assert all(c.actual_output == "Paris" for c in res.cases)
    # Serialized would be >= 1.2s; concurrent lands well under.
    assert elapsed < 0.9, f"4x0.3s cases took {elapsed:.2f}s — not running concurrently"
    assert len(threads) > 1, "all cases ran on one thread"


def test_async_agent_fn_still_awaited_directly() -> None:
    """The async path is unchanged — no thread hop, awaited on the loop."""
    on_loop: list[bool] = []

    async def agent(text: str) -> str:
        on_loop.append(asyncio.get_running_loop() is not None)
        return "Paris"

    res = evaluate(
        agent_fn=agent,
        dataset=[{"input": "q", "expected_output": "Paris"}],
        scorers=["exact_match"],
        persist=False,
    )
    assert on_loop == [True]
    assert res.cases[0].actual_output == "Paris"


def test_sync_agent_fn_returning_a_coroutine_is_still_awaited() -> None:
    """Back-compat: a sync callable handing back a coroutine kept working."""

    async def _inner() -> str:
        return "Paris"

    def agent(text: str):  # noqa: ANN202 - deliberately sync, returns a coroutine
        return _inner()

    res = evaluate(
        agent_fn=agent,
        dataset=[{"input": "q", "expected_output": "Paris"}],
        scorers=["exact_match"],
        persist=False,
    )
    assert res.cases[0].actual_output == "Paris"


def test_agent_fn_exception_is_still_recorded_as_an_errored_case() -> None:
    """Offloading must not swallow or reshape a real agent failure."""

    def boom(text: str) -> str:
        raise RuntimeError("provider 500")

    res = evaluate(
        agent_fn=boom,
        dataset=[{"input": "q", "expected_output": "a"}],
        scorers=["exact_match"],
        persist=False,
    )
    case = res.cases[0]
    assert case.actual_output is None
    assert "provider 500" in (case.error or "")
