"""Async utilities for running coroutines from sync code."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running
    (e.g., Jupyter, async test frameworks) by offloading to a thread.

    **The caller's context travels with it.** A plain
    ``ThreadPoolExecutor().submit(asyncio.run, coro)`` starts the worker with an
    *empty* :mod:`contextvars` context, so every ``ContextVar`` the caller set
    read back as its default inside the coroutine — while the same code in a
    plain script (the ``asyncio.run`` branch below) saw them fine. The bug was
    therefore invisible until someone ran from Jupyter, pytest-asyncio, or a
    framework proxy that already owns the loop.

    The one that made it concrete: ``fa.guardrail_context(context=docs)`` backs a
    ``groundedness`` rule, and losing it means the rule cannot find its context,
    raises, and — with the default ``on_error="block"`` — blocks the run it was
    supposed to score.

    ``ctx.run`` makes the copied context current for the worker, and
    ``asyncio.run``'s task inherits it at creation, so the coroutine sees exactly
    what the caller had.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(ctx.run, asyncio.run, coro).result()
    return asyncio.run(coro)
