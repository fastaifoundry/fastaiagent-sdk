"""Run limits, cancel and run info for an agent run (1.91.0, 2H + 2I).

One :class:`RunControl` per top-level run, registered by ``execution_id`` while
the run is in progress:

* ``Agent.cancel(execution_id)`` and ``timeout_seconds`` both end a run through
  it. ``arun`` runs its work as a child task, so either can stop it at once —
  inside a model or tool call; ``astream`` is a generator its consumer drives, so
  it checks at each event instead.
* :func:`check_before_model_call` holds ``max_total_tokens`` / ``max_cost_usd``
  on the run's spend before every model call, and keeps the turn and model that
  :class:`RunInfo` reports.

A run nested in another (a Swarm hop, a Supervisor worker) shares its parent's
``execution_id`` and is not registered: the run is the parent's to stop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastaiagent._internal.errors import AgentTimeoutError, RunLimitError, StopAgent
from fastaiagent.agent.context import RunUsage

T = TypeVar("T")

#: ``run_reason`` on a run-end checkpoint and ``stop_reason`` on a result.
CANCELLED = "cancelled"
TIMEOUT = "timeout"


@dataclass(frozen=True)
class RunInfo:
    """The run a tool is part of — read-only (1.91.0). See ``RunContext.run``."""

    execution_id: str
    agent_name: str
    #: The tool loop's turn, from 0.
    turn: int
    #: The model of the current turn's call (a middleware may swap it per call).
    model: str
    #: What the run has spent so far.
    usage: RunUsage = field(default_factory=RunUsage)


@dataclass
class RunControl:
    """What stops one run, and what it is doing."""

    execution_id: str
    agent_name: str
    model: str
    timeout_seconds: float | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    turn: int = 0
    #: Why the run is being stopped: ``CANCELLED`` or ``TIMEOUT``.
    reason: str | None = None
    started: float = field(default_factory=time.monotonic)
    task: asyncio.Task[Any] | None = None
    loop: asyncio.AbstractEventLoop | None = None
    #: Set by the run as it starts, for the result of a run stopped from outside
    #: it: the spend accumulator, the live message list and its prefix.
    usage_bucket: Any = None
    llm_messages: list[Any] | None = None
    prefix_ids: set[int] = field(default_factory=set)
    #: A cap that ended a re-ask (``max_total_tokens`` / ``max_cost_usd``).
    limit_reason: str | None = None
    #: The run's trace, for the result of a run stopped from outside (1.93.0).
    trace_id: str | None = None
    #: The run this one is nested in (a Swarm hop, a Supervisor worker, an agent
    #: called from a tool). Cancel and the timeout are the top-level run's; the
    #: caps are each run's own, on its own spend.
    parent: RunControl | None = None

    def root(self) -> RunControl:
        control = self
        while control.parent is not None:
            control = control.parent
        return control

    def stop(self, reason: str) -> None:
        """Stop the run: at once when it runs as a task, else at its next check."""
        if self.reason is None:
            self.reason = reason
        task, loop = self.task, self.loop
        if task is not None and loop is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)

    def overdue(self) -> bool:
        return (
            self.timeout_seconds is not None
            and time.monotonic() - self.started >= self.timeout_seconds
        )

    def timeout_error(self) -> AgentTimeoutError:
        return AgentTimeoutError(
            f"agent '{self.agent_name}' passed timeout_seconds={self.timeout_seconds:g} "
            f"(execution_id={self.execution_id})"
        )

    def stop_error(self) -> BaseException:
        """The exception the run's checkpoint records for its stop (the
        top-level run's: a nested run stops with it)."""
        top = self.root()
        return top.timeout_error() if top.reason == TIMEOUT else _RunCancelled(top)


class RunCancelled(BaseException):  # noqa: N818 - a signal, like CancelledError
    """Ends a cancelled run; never reaches the caller. Its name is what a
    cancelled run's ``run_error`` shows (``_RunCancelled`` before 1.93.0)."""

    def __init__(self, control: RunControl) -> None:
        super().__init__(f"run {control.execution_id} was cancelled")


_RunCancelled = RunCancelled


_active: ContextVar[RunControl | None] = ContextVar("fastaiagent_run_control", default=None)
_registry: dict[str, list[RunControl]] = {}
_lock = threading.Lock()


def active_control() -> RunControl | None:
    return _active.get()


def add(control: RunControl) -> None:
    """Make the run cancellable by its ``execution_id``."""
    with _lock:
        _registry.setdefault(control.execution_id, []).append(control)


def remove(control: RunControl) -> None:
    with _lock:
        runs = _registry.get(control.execution_id, [])
        if control in runs:
            runs.remove(control)
        if not runs:
            _registry.pop(control.execution_id, None)


def bind(control: RunControl) -> Any:
    """Make ``control`` the active run's. A generator binds around each step,
    never across a yield (it runs in its consumer's context)."""
    return _active.set(control)


def unbind(token: Any) -> None:
    _active.reset(token)


def cancel(execution_id: str) -> bool:
    """Stop every run of ``execution_id`` in progress here. ``False`` when none is."""
    with _lock:
        runs = list(_registry.get(execution_id, []))
    for control in runs:
        control.stop(CANCELLED)
    return bool(runs)


def stop_of(exc: BaseException, control: RunControl | None) -> str | None:
    """``CANCELLED`` / ``TIMEOUT`` when ``exc`` is the run being stopped on
    purpose, else ``None`` — for the run-end checkpoint's ``run_reason``."""
    if control is None:
        return None
    reason = control.root().reason
    if reason is not None and isinstance(
        exc, (asyncio.CancelledError, _RunCancelled, AgentTimeoutError)
    ):
        return reason
    return None


def arm_timeout(control: RunControl) -> asyncio.TimerHandle | None:
    if control.timeout_seconds is None or control.loop is None:
        return None
    remaining = max(control.timeout_seconds - (time.monotonic() - control.started), 0.0)
    return control.loop.call_later(remaining, control.stop, TIMEOUT)


def check_stopped() -> None:
    """Raise when the active run was cancelled or ran out of time (1.93.0).

    The tool loop calls this before each tool call and at the top of each turn.
    A cancel is delivered to ``arun``'s task as ``CancelledError`` at its next
    real await, and a sync tool blocks the loop: after it returned, the turn's
    remaining tools ran and the next turn's ``before_model`` hooks fired before
    the stop landed. A no-op outside a registered run.
    """
    control = _active.get()
    if control is None:
        return
    top = control.root()
    if top.reason is not None:
        raise top.stop_error()
    if top.overdue():
        top.reason = TIMEOUT
        raise top.timeout_error()


async def controlled(
    execution_id: str,
    name: str,
    work: Callable[[], Awaitable[T]],
    *,
    timeout_seconds: float | None = None,
) -> T:
    """Run ``work`` as a run that :func:`cancel` and ``timeout_seconds`` can stop
    (1.93.0) — what ``Agent._controlled`` does, for a Chain or a Swarm.

    A top-level run is registered by ``execution_id`` and runs as a task, so a
    stop lands at once, inside a node or a hop. A run inside another runs inline
    and stops with its parent. Raises ``_RunCancelled`` when cancelled and
    ``AgentTimeoutError`` when out of time; the caller's own cancellation stops
    the run and propagates.
    """
    parent = _active.get()
    control = RunControl(
        execution_id=execution_id,
        agent_name=name,
        model="",
        timeout_seconds=timeout_seconds,
        parent=parent,
    )
    token = bind(control)  # before the task, which copies it
    try:
        if parent is not None:
            return await work()
        add(control)
        control.loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(work())
        control.task = task
        timer = arm_timeout(control)
        try:
            return await task
        except (asyncio.CancelledError, _RunCancelled):
            if control.reason is None:
                # The caller's own task was cancelled: stop the run with it.
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 - unwinding already
                    pass
                raise
            if control.reason == TIMEOUT:
                raise control.timeout_error() from None
            raise _RunCancelled(control) from None
        finally:
            if timer is not None:
                timer.cancel()
    finally:
        unbind(token)
        remove(control)


async def stoppable(work: Any) -> Any:
    """Await ``work`` (a coroutine) so a cancel or timeout stops it mid-call (1.93.0).

    ``arun`` already runs as a task its stop cancels. A streamed run has no such
    task — its consumer drives it — so a stop waited for the step in flight: a
    slow tool, or a Supervisor worker running inside a delegate call, ran to the
    end. A streamed run's tool call now runs as the run's task while it lasts,
    with the timeout armed. Inline when there is no run or a task already
    carries the stop.
    """
    control = _active.get()
    top = control.root() if control is not None else None
    if top is None or top.task is not None:
        return await work
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(work)
    top.task, top.loop = task, loop
    timer = None
    if top.timeout_seconds is not None:
        remaining = max(top.timeout_seconds - (time.monotonic() - top.started), 0.0)
        timer = loop.call_later(remaining, top.stop, TIMEOUT)
    try:
        return await task
    except asyncio.CancelledError:
        if top.reason is None:
            raise  # the consumer's own cancellation
        raise top.stop_error() from None
    finally:
        top.task = None
        if timer is not None:
            timer.cancel()


def check_before_model_call(turn: int | None = None, model: str | None = None) -> None:
    """Hold the active run to its limits before a model call.

    Raises ``StopAgent`` (the cooperative end, decision B2-7) when the run has
    spent its ``max_total_tokens`` / ``max_cost_usd``, and ``RunLimitError`` when
    ``max_cost_usd`` is set but the spend cannot be priced. Records the turn and
    model for :class:`RunInfo`. A no-op outside a registered run.
    """
    control = _active.get()
    if control is None:
        return
    if turn is not None:
        control.turn = turn
    if model:
        control.model = model
    top = control.root()
    if top.reason is not None:
        # A stop that arrived between checks (a streamed run).
        raise top.stop_error()
    if top.overdue():
        top.reason = TIMEOUT
        raise top.timeout_error()
    if control is not top and control.overdue():
        # A nested agent's own timeout, checked before each of its calls.
        raise control.timeout_error()
    if control.max_total_tokens is None and control.max_cost_usd is None:
        return
    usage = _usage()
    if control.max_total_tokens is not None and usage.tokens >= control.max_total_tokens:
        raise StopAgent(
            f"Stopped: the run used {usage.tokens} tokens "
            f"(max_total_tokens={control.max_total_tokens}).",
            reason="max_total_tokens",
        )
    if control.max_cost_usd is not None:
        if not usage.cost_known:
            raise RunLimitError(
                f"max_cost_usd={control.max_cost_usd:g} cannot be checked: this run's "
                "spend cannot be priced (a model with no rate). Set a rate with "
                "set_rate_overrides() or the models.json 'pricing' block, or remove "
                "max_cost_usd."
            )
        if usage.cost >= control.max_cost_usd:
            raise StopAgent(
                f"Stopped: the run spent ${usage.cost:.6f} "
                f"(max_cost_usd={control.max_cost_usd:g}).",
                reason="max_cost_usd",
            )


def _usage() -> RunUsage:
    from fastaiagent.agent.context import current_run_usage_view

    return current_run_usage_view()


def current_run_info() -> RunInfo | None:
    """The active run's :class:`RunInfo`, or ``None`` outside a run."""
    control = _active.get()
    if control is None:
        return None
    return RunInfo(
        execution_id=control.execution_id,
        agent_name=control.agent_name,
        turn=control.turn,
        model=control.model,
        usage=_usage(),
    )
