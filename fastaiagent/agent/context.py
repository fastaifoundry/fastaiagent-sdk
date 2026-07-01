"""RunContext — typed dependency injection container for agent execution."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generic, TypeVar

T = TypeVar("T")


class RunContext(Generic[T]):
    """Typed context passed to tools during agent execution.

    The ``state`` field carries the developer's runtime dependencies
    (DB connections, API clients, user sessions, etc.). It is never
    serialized and never crosses the push boundary to the platform.

    RunContext must always be constructed explicitly — raw objects
    are not auto-wrapped.

    Example:
        @dataclass
        class AppState:
            db: Database
            user_id: str

        ctx = RunContext(state=AppState(db=db, user_id="u-1"))
        result = agent.run("Hello", context=ctx)
    """

    def __init__(self, state: T) -> None:
        self._state = state

    @property
    def state(self) -> T:
        return self._state

    def __repr__(self) -> str:
        return f"RunContext(state={self._state!r})"


# The RunContext active during the current agent turn. Set by the Agent run
# path (run/arun/astream) and read by memory blocks that resolve a dynamic
# ``scope_id`` per run (e.g. ``PersistentFactBlock(scope_id=lambda ctx: ...)``).
# Default ``None`` — outside a run, callable scope_ids resolve to "" (safe: no
# personal facts), so nothing leaks when there's no context.
_active_run_context: ContextVar[RunContext | None] = ContextVar(
    "fastaiagent_active_run_context", default=None
)


def get_active_run_context() -> RunContext | None:
    """Return the RunContext for the current agent turn, or ``None``."""
    return _active_run_context.get()


def set_active_run_context(ctx: RunContext | None):
    """Set the active RunContext; returns a token for :func:`reset`."""
    return _active_run_context.set(ctx)


def reset_active_run_context(token) -> None:
    """Reset the active RunContext to its prior value."""
    _active_run_context.reset(token)
