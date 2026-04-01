"""Prompt class with variable substitution."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """A prompt template with {{variable}} substitution."""

    name: str
    template: str
    variables: list[str] = Field(default_factory=list)
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        return list(set(re.findall(r"\{\{(\w+)\}\}", self.template)))

    def format(self, **kwargs: Any) -> str:
        """Substitute {{variable}} placeholders."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prompt:
        return cls(**data)
