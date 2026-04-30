"""``interrupt()`` primitive — suspending HITL for chains and agents.

Calling :func:`interrupt` inside a node raises :class:`InterruptSignal`, which
the chain executor catches: it writes a checkpoint with ``status="interrupted"``,
inserts a row in ``pending_interrupts``, and returns a paused
:class:`~fastaiagent.chain.chain.ChainResult`. A separate
:meth:`Chain.resume` call sets the ``_resume_value`` ``ContextVar`` and
re-executes the same node — this time ``interrupt()`` returns the
:class:`Resume` value instead of raising.

The ``context`` dict is **frozen** at suspend time: it is JSON-serialized into
the ``pending_interrupts`` / ``interrupt_context`` columns and is never
recomputed. The human approves a specific snapshot.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, Field

# Set by the chain executor at the start of execute_chain / Chain.resume so
# interrupt() and (Phase 3) @idempotent can read the active execution id.
_execution_id: ContextVar[str | None] = ContextVar("_execution_id", default=None)

# Set by Chain.resume right before re-executing the interrupted node, and
# cleared after that node completes. When non-None, interrupt() returns it
# instead of raising InterruptSignal.
_resume_value: ContextVar[Resume | None] = ContextVar("_resume_value", default=None)

# Set by Agent / Swarm / Supervisor (Phases 5-7) so checkpoints carry the
# nested path. None for plain Chain.
_agent_path: ContextVar[str | None] = ContextVar("_agent_path", default=None)


class Resume(BaseModel):
    """Value passed to :meth:`Chain.resume` and returned by :func:`interrupt`."""

    approved: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    # ``data`` is reserved for non-approval resume cases (future).


class InterruptSignal(Exception):  # noqa: N818  (public API name from v1 spec)
    """Raised by :func:`interrupt` when no resume value is in scope.

    The chain executor catches this internally; user code should not.
    """

    def __init__(self, reason: str, context: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.context = context


class AlreadyResumed(Exception):  # noqa: N818  (public API name from v1 spec)
    """Raised when ``Chain.resume`` is called twice for the same execution.

    Coordination is handled by atomically deleting the ``pending_interrupts``
    row inside :meth:`Chain.resume`; the loser of a race sees no row and
    raises this.
    """


def interrupt(reason: str, context: dict[str, Any]) -> Resume:
    """Suspend the workflow for human approval.

    First call (no resume value in scope): raises :class:`InterruptSignal`.
    The chain executor catches it, persists the suspension, and returns a
    paused result.

    Second call (after :meth:`Chain.resume`): returns the :class:`Resume`
    value the resumer passed in.
    """
    v = _resume_value.get()
    if v is not None:
        return v
    raise InterruptSignal(reason, context)
