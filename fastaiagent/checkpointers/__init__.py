"""Checkpoint storage backends.

Public surface:
    - :class:`Checkpointer` — protocol shared by every backend
    - :class:`SQLiteCheckpointer` — default local backend (used by Chain/Agent)
    - :class:`PendingInterrupt` — model for workflows suspended on ``interrupt()``
    - :class:`PostgresCheckpointer` — Phase 8 stub; real impl ships then
"""

from __future__ import annotations

from fastaiagent.checkpointers.protocol import Checkpointer, PendingInterrupt
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

__all__ = [
    "Checkpointer",
    "PendingInterrupt",
    "SQLiteCheckpointer",
    "PostgresCheckpointer",
]


def __getattr__(name: str) -> object:
    if name == "PostgresCheckpointer":
        from fastaiagent.checkpointers.postgres import PostgresCheckpointer

        return PostgresCheckpointer
    raise AttributeError(f"module 'fastaiagent.checkpointers' has no attribute {name!r}")
