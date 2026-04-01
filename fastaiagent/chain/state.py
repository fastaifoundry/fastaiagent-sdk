"""Chain state management with optional Pydantic schema validation."""

from __future__ import annotations

import copy
from typing import Any

from fastaiagent._internal.errors import ChainStateValidationError
from fastaiagent.tool.schema import validate_schema


class ChainState:
    """State that flows through chain execution.

    Supports optional JSON Schema validation at each node.
    """

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        self._data.update(data)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot for checkpointing."""
        return copy.deepcopy(self._data)

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> ChainState:
        """Restore from a checkpoint snapshot."""
        return cls(initial=data)

    def validate(self, schema: dict) -> None:
        """Validate state against a JSON Schema. Raises on failure."""
        violations = validate_schema(schema, self._data)
        if violations:
            messages = [v.message for v in violations[:5]]
            raise ChainStateValidationError(
                f"State validation failed: {'; '.join(messages)}"
            )

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ChainState({self._data})"
