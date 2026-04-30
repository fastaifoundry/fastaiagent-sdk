"""``@idempotent`` decorator — execution-scoped result cache for side effects.

Wrap a side-effectful function so a re-executed node (e.g. on resume from
``interrupt()``) does not re-run it. The cache key is
``(execution_id, function_qualname + sha256(args, kwargs))`` and rows live in
the ``idempotency_cache`` table managed by the active :class:`Checkpointer`.

Outside a chain run (no active execution_id) the wrapped function is called
straight through with no caching.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic_core import to_jsonable_python

from fastaiagent.chain.interrupt import _execution_id

if TYPE_CHECKING:
    from fastaiagent.checkpointers.protocol import Checkpointer

# Set by ``execute_chain`` (and by Agent / Swarm in later phases) so the
# decorator can resolve the active backend without the user threading it
# through every call site.
_current_checkpointer: ContextVar[Checkpointer | None] = ContextVar(
    "_current_checkpointer", default=None
)


class IdempotencyError(Exception):
    """Raised when an ``@idempotent`` function returns a non-serializable value."""


def _build_default_key(qualname: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Stable hash key from the function name + JSON-encoded args/kwargs."""
    payload = json.dumps(
        [qualname, list(args), kwargs],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def idempotent(
    fn: Callable[..., Any] | None = None,
    *,
    key_fn: Callable[..., str] | None = None,
) -> Callable[..., Any]:
    """Cache the wrapped function's result by ``(execution_id, key)``.

    Usage:

        @idempotent
        def charge(amount, customer_id): ...

        @idempotent(key_fn=lambda user, req: f"{user.id}:{req.id}")
        def process(user, req): ...

    Behavior:
        - First call inside an execution: runs the body, stores the result.
        - Subsequent calls in the same execution with the same key: returns
          the cached value, never re-runs the body.
        - Calls in a different execution_id: cache miss, runs again.
        - Calls outside any chain run: no caching, body runs every time.

    The cached value is the JSON-serializable form of the original return
    (Pydantic models / dataclasses go through ``pydantic_core.to_jsonable_python``).
    Returns are stored as JSON, so a cache hit yields the deserialized form
    (dicts / lists / primitives) — design idempotent functions to return
    plain data, or hydrate the cached dict back into a model at the call site.
    """

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(f)
        def inner(*args: Any, **kwargs: Any) -> Any:
            exec_id = _execution_id.get()
            cp = _current_checkpointer.get()
            if exec_id is None or cp is None:
                # Outside a chain run, or no checkpointer wired — run the
                # body every time. This also makes @idempotent functions
                # safe to call from unit tests.
                return f(*args, **kwargs)

            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = _build_default_key(f.__qualname__, args, kwargs)

            cached = cp.get_idempotent(exec_id, key)
            if cached is not None:
                return cached

            result = f(*args, **kwargs)

            try:
                serializable = to_jsonable_python(result)
                # Round-trip through json so we surface non-serializable
                # leaves (e.g. arbitrary objects nested inside dicts) here
                # rather than at write time.
                json.dumps(serializable)
            except (TypeError, ValueError) as e:
                raise IdempotencyError(
                    f"@idempotent function {f.__qualname__!r} returned a "
                    f"non-JSON-serializable value of type "
                    f"{type(result).__name__!r}: {e}. Wrap the return in a "
                    "Pydantic model, return plain data, or split this into a "
                    "non-cached node."
                ) from e

            cp.put_idempotent(exec_id, key, serializable)
            return result

        return inner

    return wrap if fn is None else wrap(fn)
