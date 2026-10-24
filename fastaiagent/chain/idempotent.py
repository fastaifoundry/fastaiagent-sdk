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
import inspect
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

#: The execution id the cache is keyed by — the run that bound the checkpointer.
#: A run nested in it (an Agent node of a Chain) has an execution id of its own,
#: but must share its parent's cache: the parent's resume re-runs it from the top.
_cache_execution_id: ContextVar[str | None] = ContextVar("_cache_execution_id", default=None)


def bind_cache(execution_id: str, checkpointer: Checkpointer | None) -> Any:
    """Bind the cache for a run — unless an enclosing run already bound one.

    Before 1.92.0 every run bound its own, so an Agent node of a checkpointed
    Chain replaced the Chain's cache with the Agent's (usually none) under a fresh
    execution id, and a charge before an ``interrupt()`` ran again on every Chain
    resume. Returns the tokens for :func:`unbind_cache` (``None`` when nested)."""
    if _current_checkpointer.get() is not None:
        return None
    return _current_checkpointer.set(checkpointer), _cache_execution_id.set(execution_id)


def unbind_cache(tokens: Any) -> None:
    if tokens is not None:
        cp_token, exec_token = tokens
        _cache_execution_id.reset(exec_token)
        _current_checkpointer.reset(cp_token)


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

    ``async def`` functions are supported (1.92.0): the body is awaited once
    and its result cached the same way. A ``None`` result is cached too.
    """

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        def scope(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, str, str] | None:
            """The checkpointer, execution id and key, or ``None`` outside a run
            (no execution_id or no checkpointer): then the body runs every time,
            which also keeps @idempotent functions callable from unit tests."""
            # The id of the run that bound the cache (see bind_cache); a bare
            # ``_execution_id`` when a caller bound the two variables directly.
            exec_id = _cache_execution_id.get() or _execution_id.get()
            cp = _current_checkpointer.get()
            if exec_id is None or cp is None:
                return None
            if key_fn is not None:
                return cp, exec_id, key_fn(*args, **kwargs)
            return cp, exec_id, _build_default_key(f.__qualname__, args, kwargs)

        def store(cp: Any, exec_id: str, key: str, result: Any) -> None:
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
            cp.put_idempotent(exec_id, key, {_ENVELOPE: serializable})

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def ainner(*args: Any, **kwargs: Any) -> Any:
                found = scope(args, kwargs)
                if found is None:
                    return await f(*args, **kwargs)
                cp, exec_id, key = found
                hit, cached = _unwrap(cp.get_idempotent(exec_id, key))
                if hit:
                    return cached
                result = await f(*args, **kwargs)
                store(cp, exec_id, key, result)
                return result

            return ainner

        @functools.wraps(f)
        def inner(*args: Any, **kwargs: Any) -> Any:
            found = scope(args, kwargs)
            if found is None:
                return f(*args, **kwargs)
            cp, exec_id, key = found
            hit, cached = _unwrap(cp.get_idempotent(exec_id, key))
            if hit:
                return cached
            result = f(*args, **kwargs)
            store(cp, exec_id, key, result)
            return result

        return inner

    return wrap if fn is None else wrap(fn)


#: Results are stored as ``{_ENVELOPE: value}`` (1.92.0) so a cached ``None``
#: is told apart from "not cached" — before, a function returning ``None`` ran
#: again on every call. Rows written by older versions hold the bare value.
_ENVELOPE = "__fa_idem__"


def _unwrap(stored: Any) -> tuple[bool, Any]:
    """``(hit, value)`` for what the checkpointer returned."""
    if isinstance(stored, dict) and stored.keys() == {_ENVELOPE}:
        return True, stored[_ENVELOPE]
    # A pre-1.92.0 row: a bare value, where ``None`` meant "not cached".
    return stored is not None, stored
