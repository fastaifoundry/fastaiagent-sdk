"""Scorer base class and ScorerResult."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, model_validator

#: What one scorer said about one case (1.89.0).
VerdictOutcome = Literal["passed", "failed", "errored", "skipped"]


class ScorerResult(BaseModel):
    """Result of scoring a single evaluation case.

    Three states besides a verdict exist since 1.89.0, so that a scorer that
    could not judge never reads as a judgement (the same contract as a
    guardrail's ``errored``):

    * ``errored``: the check could not run: the judge call failed, its verdict
      was unparseable after a retry, the model refused, or the scorer raised.
    * ``skipped``: the check does not apply to this case: no expected output,
      no expected trajectory, no retrieved context.
    * a verdict: ``passed`` with ``score``.

    An errored or skipped result is forced to ``passed=False, score=0.0``, so a
    reader that ignores the new fields fails closed. Rates exclude both: see
    :attr:`outcome`.
    """

    score: float = 0.0
    passed: bool = False
    reason: str | None = None
    errored: bool = False
    skipped: bool = False
    details: dict[str, Any] | None = None
    """Non-content extras: a judge's ``raw_score``/``scale``/``threshold``, a
    trajectory's tool names. Never the case's text."""

    @model_validator(mode="after")
    def _not_a_verdict_never_passes(self) -> ScorerResult:
        if self.errored or self.skipped:
            self.passed = False
            self.score = 0.0
        return self

    @property
    def outcome(self) -> VerdictOutcome:
        if self.errored:
            return "errored"
        if self.skipped:
            return "skipped"
        return "passed" if self.passed else "failed"

    @classmethod
    def error(cls, reason: str, **details: Any) -> ScorerResult:
        """The check could not run. ``reason`` says why; it is not a verdict."""
        return cls(errored=True, reason=reason, details=details or None)

    @classmethod
    def not_applicable(cls, reason: str, **details: Any) -> ScorerResult:
        """The check does not apply to this case (a missing input)."""
        return cls(skipped=True, reason=reason, details=details or None)

    def stored(self) -> dict[str, Any]:
        """The per-case record: ``{passed, score, reason}``, plus ``errored``,
        ``skipped`` and ``details`` only when they are not at their defaults,
        so a run with none of them stores exactly what 1.88.0 did."""
        out: dict[str, Any] = {"passed": self.passed, "score": self.score, "reason": self.reason}
        if self.errored:
            out["errored"] = True
        if self.skipped:
            out["skipped"] = True
        if self.details:
            out["details"] = self.details
        return out


def verdict_outcome(entry: Any) -> VerdictOutcome:
    """The outcome of a stored ``per_scorer`` entry (a dict) or a ``ScorerResult``."""
    if isinstance(entry, ScorerResult):
        return entry.outcome
    if isinstance(entry, dict):
        if entry.get("errored"):
            return "errored"
        if entry.get("skipped"):
            return "skipped"
        return "passed" if entry.get("passed") else "failed"
    return "passed" if entry else "failed"


class Scorer:
    """Base class for evaluation scorers."""

    name: str = "base"

    def score(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """How this scorer was configured, for the run record (1.89.0).

        Stays local (persisted under ``metadata["scorer_config"]``); never keys,
        headers or other credentials. Subclasses add their own fields.
        """
        return {"type": type(self).__name__}

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
    """A scorer backed by a Python function.

    Extra keyword arguments are filtered to what the wrapped function actually
    declares. ``evaluate()`` supplies run facts (``latency_ms``, ``cost``,
    ``cost_known``) to every scorer since 1.67.0, and a user function written as
    ``def check(input, output, expected)`` must keep working — it simply never
    asked for them. A function with ``**kwargs`` still receives everything.
    """

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self._fn = fn
        self._accepted = _accepted_kwargs(fn)

    def score(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        if self._accepted is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in self._accepted}
        result = self._fn(input=input, output=output, expected=expected, **kwargs)
        if isinstance(result, ScorerResult):
            return result
        if isinstance(result, bool):
            return ScorerResult(score=1.0 if result else 0.0, passed=result)
        if isinstance(result, (int, float)):
            return ScorerResult(score=float(result), passed=float(result) >= 0.5)
        return ScorerResult(score=0.0, passed=False, reason=str(result))


def _accepted_kwargs(fn: Callable[..., Any]) -> set[str] | None:
    """Keyword names ``fn`` will accept, or ``None`` when it takes ``**kwargs``.

    ``None`` means "pass everything through" — that is the signature that opted
    in. An unreadable signature (a C callable, an exotic wrapper) also returns
    ``None``: better to hand it everything and let it raise on its own terms
    than to silently drop an argument it wanted.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
