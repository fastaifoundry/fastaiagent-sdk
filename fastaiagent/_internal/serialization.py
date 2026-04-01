"""Serialization helpers for to_dict / from_dict patterns."""

from __future__ import annotations

import enum
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible types."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return str(value)


def to_json(data: dict[str, Any], pretty: bool = False) -> str:
    """Convert a dict to JSON string."""
    kwargs: dict[str, Any] = {}
    if pretty:
        kwargs["indent"] = 2
    return json.dumps(data, default=str, **kwargs)


def from_json(text: str) -> dict[str, Any]:
    """Parse a JSON string to dict."""
    result: dict[str, Any] = json.loads(text)
    return result
