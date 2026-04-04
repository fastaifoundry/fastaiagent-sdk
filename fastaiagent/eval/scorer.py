"""Scorer base class and ScorerResult."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ScorerResult(BaseModel):
    """Result of scoring a single evaluation case."""

    score: float = 0.0
    passed: bool = False
    reason: str | None = None


class Scorer:
    """Base class for evaluation scorers."""

    name: str = "base"

    def score(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        raise NotImplementedError

    @classmethod
    def from_platform(cls, name: str) -> Scorer:
        """Pull scorer config from platform (e.g., LLM judge config)."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )
        api = get_platform_api()
        data = api.get(f"/public/v1/eval/scorers/{name}")

        from fastaiagent.eval.llm_judge import LLMJudge

        return LLMJudge(
            criteria=data.get("criteria", "correctness"),
            prompt_template=data.get("prompt_template"),
            scale=data.get("scale", "binary"),
        )

    @staticmethod
    def code(name: str | None = None) -> Callable[..., Any]:
        """Decorator to create a custom code scorer.

        Example:
            @Scorer.code("length_check")
            def check_length(input, output, expected=None):
                return ScorerResult(score=1.0 if len(output) > 10 else 0.0, passed=len(output) > 10)
        """

        def decorator(fn: Callable[..., Any]) -> CodeScorer:
            return CodeScorer(name=name or fn.__name__, fn=fn)

        return decorator


class CodeScorer(Scorer):
    """A scorer backed by a Python function."""

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self._fn = fn

    def score(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        result = self._fn(input=input, output=output, expected=expected, **kwargs)
        if isinstance(result, ScorerResult):
            return result
        if isinstance(result, bool):
            return ScorerResult(score=1.0 if result else 0.0, passed=result)
        if isinstance(result, (int, float)):
            return ScorerResult(score=float(result), passed=float(result) >= 0.5)
        return ScorerResult(score=0.0, passed=False, reason=str(result))
