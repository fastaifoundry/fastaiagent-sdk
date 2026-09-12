"""Checkpoint Pydantic model — shared across chain and checkpointer backends.

The concrete storage implementation lives in :mod:`fastaiagent.checkpointers`.
This module only defines the model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

#: ``step_type`` of the terminal row a run writes when it ends (durability
#: audit D5). Everything else is a step *within* a run; this one says the run
#: itself is over, and its ``status`` says how it went.
RUN_END = "run_end"


class Checkpoint(BaseModel):
    """A checkpoint snapshot of chain (or agent) execution at a node."""

    checkpoint_id: str = ""
    parent_checkpoint_id: str | None = None
    chain_name: str = ""
    execution_id: str = ""
    node_id: str = ""
    node_index: int = 0
    #: What kind of boundary this checkpoint sits on — ``llm_call``,
    #: ``tool_call``, ``hitl_pause``, ``node``, ``handoff``, ``fork_origin``, or
    #: :data:`RUN_END`. Optional and defaulted, so a checkpoint written by older
    #: code (or restored from a plane row that predates it) is still valid.
    #:
    #: The plane's wire schema has always had this key and caps it at 40
    #: characters; until D5 the SDK never emitted it, which is why a finished
    #: run and one that died right after its last step looked identical there.
    step_type: str | None = None
    status: str = "completed"
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    node_input: dict[str, Any] = Field(default_factory=dict)
    node_output: dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    iteration_counters: dict[str, int] = Field(default_factory=dict)
    interrupt_reason: str | None = None
    interrupt_context: dict[str, Any] = Field(default_factory=dict)
    agent_path: str | None = None
    created_at: str = ""


def write_run_end(
    checkpointer: Any,
    *,
    execution_id: str,
    chain_name: str,
    status: str,
    error: BaseException | None = None,
    state_snapshot: dict[str, Any] | None = None,
    agent_path: str | None = None,
) -> None:
    """Write the terminal checkpoint for a run. Best-effort, never raises.

    ``status`` is ``completed`` or ``failed``. This is the row that lets anyone
    downstream tell a finished run from one that died the instant after its last
    step — before it, both left a ``completed`` checkpoint as the newest row.

    ⚠ **Never call this on the paused path.** A run that returns
    ``status="paused"`` has not ended, and a row written after an
    ``interrupted`` one would hide it from the four ``latest.status ==
    "interrupted"`` guards that make a HITL resume demand a ``Resume`` value.

    ⚠ **It must never mask the exception that caused it.** On the failure path
    the caller is inside an ``except`` about to re-raise, so a checkpointer
    problem here has to stay a log line — the same discipline
    ``record_pause_event`` already uses.

    Derives what it can from the run's last checkpoint so callers stay thin:

    * ``node_index`` is one past the last step, so the row sorts **last** under
      both orderings the SDK uses — ``get_last`` reads ``created_at DESC,
      rowid DESC`` while ``list`` reads ``node_index ASC``. A run-end row that
      sorted first would be the row ``Supervisor._hydrate_input`` scanned first
      and the row a scoped ``aresume`` never reached.
    * ``state_snapshot`` defaults to the final state rather than a bare marker.
      That is deliberate: once this row is the newest, it is what the plane
      serves as ``/latest`` and what a console State tab renders, so a tombstone
      carrying nothing would *lose* the end state that used to be visible there.
      It costs one extra row's worth of snapshot per run, not a copy per step.
    """
    import logging
    import uuid as _uuid
    from datetime import datetime, timezone

    logger = logging.getLogger(__name__)
    try:
        last = checkpointer.get_last(execution_id)
    except Exception:  # pragma: no cover - defensive
        last = None
    if state_snapshot is None:
        state_snapshot = dict(last.state_snapshot) if last is not None else {}
    # ⚠ ``agent_path`` is NOT derived from the last checkpoint, deliberately.
    # This row describes the RUN, so it must carry the path of whoever owns the
    # run — not of whichever nested agent happened to write most recently. In a
    # Supervisor the last row is often a *worker's*, and inheriting
    # ``supervisor:X/worker:Y`` would make ``_has_worker_state`` report state for
    # a worker that never ran, flipping the next delegation from ``arun`` to
    # ``aresume``. Callers that own a path pass it; a plain Chain has none.

    snapshot: dict[str, Any] = dict(state_snapshot)
    snapshot["run_status"] = status
    if error is not None:
        snapshot["run_error"] = f"{type(error).__name__}: {error}"

    try:
        checkpointer.put(
            Checkpoint(
                checkpoint_id=str(_uuid.uuid4()),
                chain_name=chain_name,
                execution_id=execution_id,
                node_id=RUN_END,
                node_index=(last.node_index + 1) if last is not None else 0,
                step_type=RUN_END,
                status=status,
                state_snapshot=snapshot,
                agent_path=agent_path,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
    except Exception:
        logger.debug("run-end checkpoint write failed for %s", execution_id, exc_info=True)


def latest_resumable(
    checkpointer: Any,
    execution_id: str,
    *,
    latest: Checkpoint | None = None,
    runner: str = "Execution",
) -> Checkpoint | None:
    """The newest checkpoint a resume may re-enter at — or a refusal.

    The two terminal statuses go opposite ways, and that asymmetry is the whole
    point of the marker:

    * **completed** → raise :class:`AlreadyResumed`. Before the marker existed,
      resuming a run that had already finished successfully found its last
      ``turn:N`` / node checkpoint, saw ``status="completed"``, and cheerfully
      re-executed the agent — re-calling the model and re-firing every
      side-effecting tool that was not wrapped in ``@idempotent``. Nothing
      anywhere distinguished "crashed at turn 3" from "finished after turn 3".
    * **failed** → skip the marker and hand back the last real checkpoint.
      Crash recovery is the feature; a run that raised must stay resumable, and
      behaviour there is identical to before.

    Skipping matters beyond tidiness: a run-end row's ``node_id`` names no node
    the executor can restart at, and :meth:`Chain.resume` locates its restart
    point by matching that id against the topological order. An unmatched id
    leaves ``start_node`` as None, which ``execute_chain`` reads as "start at
    index 0" — silently replaying the entire chain.
    """
    from fastaiagent.chain.interrupt import AlreadyResumed

    if latest is None:
        got = checkpointer.get_last(execution_id)
        latest = got if isinstance(got, Checkpoint) else None
    if latest is None or not is_run_end(latest):
        return latest
    if latest.status != "failed":
        raise AlreadyResumed(
            f"{runner} '{execution_id}' already finished ({latest.status}) — there is "
            "nothing to resume. Use fork() to branch a new run from one of its "
            "checkpoints, or use a fresh execution_id to run again."
        )
    # A failed run: step back past the tombstone to the last real re-entry point.
    history: list[Checkpoint] = list(checkpointer.list(execution_id, limit=500))
    for candidate in reversed(history):
        if not is_run_end(candidate):
            return candidate
    return None


def is_run_end(checkpoint: Checkpoint | None) -> bool:
    """True when this row marks the END of a run, not a step inside one.

    Every resume path must ask this before treating the latest checkpoint as a
    re-entry point. A run-end row is a *tombstone*: its ``node_id`` names no
    node the executor can restart at, so feeding it to
    :meth:`Chain.resume` — which locates the restart point by matching
    ``node_id`` against the topological order — would match nothing, leave
    ``start_node`` as None, and silently replay the whole chain from the top.
    """
    return checkpoint is not None and checkpoint.step_type == RUN_END
