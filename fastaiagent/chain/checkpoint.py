"""Checkpoint Pydantic model — shared across chain and checkpointer backends.

The concrete storage implementation lives in :mod:`fastaiagent.checkpointers`.
This module only defines the dataclass.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    """A checkpoint snapshot of chain (or agent) execution at a node."""

    checkpoint_id: str = ""
    parent_checkpoint_id: str | None = None
    chain_name: str = ""
    execution_id: str = ""
    node_id: str = ""
    node_index: int = 0
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
