"""Guardrail class and related types."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync


class GuardrailPosition(str, Enum):
    """Where in the pipeline this guardrail runs."""

    input = "input"
    output = "output"
    tool_call = "tool_call"
    tool_result = "tool_result"


class GuardrailType(str, Enum):
    """Implementation type of the guardrail."""

    code = "code"
    llm_judge = "llm_judge"
    regex = "regex"
    schema = "schema"
    classifier = "classifier"


class GuardrailResult(BaseModel):
    """Result of a guardrail execution."""

    passed: bool
    score: float | None = None
    message: str | None = None
    execution_time_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Guardrail:
    """A validation guardrail for agent input/output/tool calls.

    Supports 5 implementation types: code, llm_judge, regex, schema, classifier.
    """

    def __init__(
        self,
        name: str,
        guardrail_type: GuardrailType = GuardrailType.code,
        position: GuardrailPosition = GuardrailPosition.output,
        config: dict[str, Any] | None = None,
        blocking: bool = True,
        description: str = "",
        fn: Callable | None = None,
    ):
        self.name = name
        self.guardrail_type = guardrail_type
        self.position = position
        self.config = config or {}
        self.blocking = blocking
        self.description = description
        self.fn = fn  # for code guardrails with inline function

    def execute(self, data: str | dict) -> GuardrailResult:
        """Execute the guardrail synchronously."""
        return run_sync(self.aexecute(data))

    async def aexecute(self, data: str | dict) -> GuardrailResult:
        """Execute the guardrail asynchronously."""
        import time

        from fastaiagent.guardrail.implementations import run_guardrail

        start = time.monotonic()
        result = await run_guardrail(self, data)
        result.execution_time_ms = int((time.monotonic() - start) * 1000)
        return result

    def to_dict(self) -> dict:
        """Serialize to canonical format."""
        return {
            "name": self.name,
            "guardrail_type": self.guardrail_type.value,
            "position": self.position.value,
            "config": self.config,
            "blocking": self.blocking,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Guardrail:
        """Deserialize from canonical format."""
        return cls(
            name=data["name"],
            guardrail_type=GuardrailType(data.get("guardrail_type", "code")),
            position=GuardrailPosition(data.get("position", "output")),
            config=data.get("config", {}),
            blocking=data.get("blocking", True),
            description=data.get("description", ""),
        )
