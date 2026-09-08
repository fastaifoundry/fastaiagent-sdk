"""A run-scoped slot for values a guardrail needs but the payload doesn't carry.

Most guardrails judge one thing: the text at the gate. ``groundedness`` is the
exception — it scores an *answer* against the *context* it was supposed to use,
and an output guardrail only ever receives the answer.

A locally-defined :func:`~fastaiagent.guardrail.builtins.grounded` guardrail
solves this with a callable (``grounded(lambda: latest_docs)``), but a callable
cannot be serialised into a policy rule, so a plane-authored ``groundedness``
rule has no way to reach the retrieval step. This slot is that convention: the
retrieval step puts the context in, the rule names the key it wants with
``config.context_key``, and neither has to know about the other.

    import fastaiagent as fa

    docs = retriever.search(question)
    with fa.guardrail_context(context=docs):
        reply = agent.run(question)

Backed by a :class:`~contextvars.ContextVar`, so it is per-task: concurrent runs
in the same process never see each other's context, and a value set inside an
``asyncio.Task`` does not leak out of it.

**A missing slot is not a pass.** A ``groundedness`` rule that cannot find its
context raises, and ``on_error`` decides (fail closed by default). An answer
scored against nothing blocks everything and scored against itself blocks
nothing; both are worse than reporting that the rule could not run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

_guardrail_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "fastaiagent_guardrail_context", default=None
)


def get_guardrail_context() -> dict[str, Any]:
    """Return the current run's guardrail context, or ``{}`` when unset."""
    return _guardrail_context.get() or {}


def set_guardrail_context(**values: Any) -> Token[dict[str, Any] | None]:
    """Merge ``values`` into the run-scoped guardrail context.

    Returns the :class:`~contextvars.Token` to hand back to
    :func:`clear_guardrail_context`. Prefer :func:`guardrail_context`, which
    does that for you.
    """
    merged = {**get_guardrail_context(), **values}
    return _guardrail_context.set(merged)


def clear_guardrail_context(token: Token[dict[str, Any] | None]) -> None:
    """Restore the guardrail context to what it was before ``token`` was issued."""
    _guardrail_context.reset(token)


@contextmanager
def guardrail_context(**values: Any) -> Iterator[dict[str, Any]]:
    """Set run-scoped guardrail context for the duration of the block.

        with fa.guardrail_context(context=retrieved_chunks):
            reply = agent.run(question)

    Yields the merged mapping, and restores the previous value on exit —
    including when the body raises.
    """
    token = set_guardrail_context(**values)
    try:
        yield get_guardrail_context()
    finally:
        clear_guardrail_context(token)
