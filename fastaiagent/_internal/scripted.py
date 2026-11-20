"""Which eval runs were answered by a scripted model (1.104.0).

``TestModel`` and ``FunctionModel`` return what a test scripted, so an eval run that
used one measures the script, not the agent. Such a run is fine locally — it proves
the wiring — but sent to a connected plane its verdict reads as evidence about the
agent. GitHub Copilot did exactly that in a skill test: with no model key it
scripted the answers, scored 100% and recorded a passing verdict on the plane.

``evaluate()`` and the pytest plugin's ``evaluate_one`` open a
:func:`scripted_model_recorder` around the calls they score; every scripted model
call made inside it marks it. The run then stays local (see
``fastaiagent.eval.results.record_gate_result``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class ScriptedCalls:
    """Set by any scripted model call made while this recorder is current."""

    __slots__ = ("seen",)

    def __init__(self) -> None:
        self.seen = False

    def __bool__(self) -> bool:
        return self.seen


_current: ContextVar[ScriptedCalls | None] = ContextVar(
    "fastaiagent_scripted_calls", default=None
)


def note_scripted_call() -> None:
    """Called by ``TestModel`` / ``FunctionModel`` on every completion, stream or
    decision. A no-op outside a recorder."""
    calls = _current.get()
    if calls is not None:
        calls.seen = True


@contextmanager
def scripted_model_recorder() -> Iterator[ScriptedCalls]:
    """Record whether any scripted model answers inside the block. Tasks and
    ``asyncio.to_thread`` workers started inside it share the same recorder,
    because they copy the current context."""
    calls = ScriptedCalls()
    token = _current.set(calls)
    try:
        yield calls
    finally:
        _current.reset(token)
