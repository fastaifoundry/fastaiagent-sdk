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
    # Provenance (Gap 4) — set when this prompt came from the control-plane
    # registry so a run that uses it can attribute the llm_call span for Prompt
    # Analytics. Optional/defaulted → additive; local prompts leave them unset.
    slug: str | None = None
    source: str | None = None  # "platform" | "local"
    environment: str | None = None  # e.g. "production"

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

    def trace_attributes(self) -> dict[str, Any]:
        """The ``fastaiagent.prompt.*`` attributes that link a run to this prompt.

        For a root span opened by your own code, such as
        ``fa.trace_context(..., **prompt.trace_attributes())`` around a
        framework call. ``name`` and ``version`` for the Local UI's prompt
        lineage; ``slug`` and ``environment`` too for a control-plane prompt,
        which the plane links by slug and version. A native ``fa.Agent`` given
        the ``Prompt`` stamps the same keys itself. Unset values are left out.
        """
        attrs = {
            "fastaiagent.prompt.name": self.name,
            "fastaiagent.prompt.version": self.version,
            "fastaiagent.prompt.slug": self.slug,
            "fastaiagent.prompt.environment": self.environment,
        }
        return {k: v for k, v in attrs.items() if v is not None}

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
