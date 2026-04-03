"""RunContext — typed dependency injection container for agent execution."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

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
