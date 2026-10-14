"""Prompt fragment for modular prompt composition."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

#: A ``{{@name}}`` fragment reference. One pattern for finding and resolving.
FRAGMENT_RE = re.compile(r"\{\{@(\w+)\}\}")


def fragment_refs(template: str) -> list[str]:
    """The fragment names ``template`` references, in first-use order, once each."""
    return list(dict.fromkeys(FRAGMENT_RE.findall(template or "")))


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
