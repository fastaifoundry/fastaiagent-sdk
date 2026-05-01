"""Chain state management with optional Pydantic schema validation."""

from __future__ import annotations

import copy
from typing import Any

from fastaiagent._internal.errors import ChainStateValidationError
from fastaiagent.tool.schema import validate_schema


def _serialize_for_checkpoint(value: Any) -> Any:
    """Walk ``value`` and convert :class:`Image`/:class:`PDF` to JSON-safe dicts.

    Tuples become lists. Other non-collection values pass through unchanged.
    """
    from fastaiagent.multimodal.image import Image as MMImage
    from fastaiagent.multimodal.pdf import PDF as MMPDF

    if isinstance(value, MMImage) or isinstance(value, MMPDF):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _serialize_for_checkpoint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_checkpoint(v) for v in value]
    return value


def _hydrate_from_checkpoint(value: Any) -> Any:
    """Inverse of :func:`_serialize_for_checkpoint`. Rehydrates Image/PDF dicts."""
    from fastaiagent.multimodal.image import Image as MMImage
    from fastaiagent.multimodal.pdf import PDF as MMPDF

    if isinstance(value, dict):
        marker = value.get("type") if "type" in value else None
        if marker == "image" and "data_base64" in value:
            return MMImage.from_dict(value)
        if marker == "pdf" and "data_base64" in value:
            return MMPDF.from_dict(value)
        return {k: _hydrate_from_checkpoint(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_hydrate_from_checkpoint(v) for v in value]
    return value


class ChainState:
    """State that flows through chain execution.

    Supports optional JSON Schema validation at each node. Multimodal
    values (``Image``, ``PDF``) survive checkpoint round-trips via the
    ``to_dict``/``from_dict`` helpers on each type.
    """

    def __init__(self, initial: dict[str, Any] | None = None):
        # Hydrate any Image/PDF dict-markers produced by a previous
        # ``snapshot()`` so resumes restore real objects, not dict stubs.
        # User-supplied initial state with real ``Image``/``PDF`` instances
        # passes through unchanged because the walker only rebuilds dicts
        # whose ``type`` field is ``"image"`` / ``"pdf"`` *and* carries a
        # ``data_base64`` field.
        self._data: dict[str, Any] = (
            {} if initial is None else _hydrate_from_checkpoint(dict(initial))
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        self._data.update(data)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for checkpointing.

        ``Image`` and ``PDF`` instances anywhere inside the state are
        converted via their ``to_dict`` helpers; everything else is deep
        copied as before. Reverse with :py:meth:`from_snapshot`.
        """
        deep = copy.deepcopy(self._data)
        return {k: _serialize_for_checkpoint(v) for k, v in deep.items()}

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> ChainState:
        """Restore from a checkpoint snapshot, rehydrating any media values."""
        hydrated = {k: _hydrate_from_checkpoint(v) for k, v in data.items()}
        return cls(initial=hydrated)

    def validate(self, schema: dict[str, Any]) -> None:
        """Validate state against a JSON Schema. Raises on failure."""
        violations = validate_schema(schema, self._data)
        if violations:
            messages = [v.message for v in violations[:5]]
            raise ChainStateValidationError(f"State validation failed: {'; '.join(messages)}")

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
