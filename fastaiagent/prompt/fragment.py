"""Prompt fragment for modular prompt composition."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Fragment(BaseModel):
    """A reusable prompt building block.

    Referenced in prompts via {{@fragment_name}}.
    """

    name: str
    content: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "content": self.content, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fragment:
        return cls(**data)
