"""Scorers from plain data, for callers that cannot pass a Python object (1.89.0).

The Local UI's "Run eval" could only send scorer *names*, so it could run
``exact_match`` and nothing that needs a parameter (a regex pattern, a judge's
criteria and model). A spec is a dict:

* ``{"type": "builtin", "name": "regex_match", "params": {"pattern": "^\\d+$"}}``
* ``{"type": "builtin", "name": "task_completion", "model": {"provider": "openai",
  "model": "gpt-4o-mini"}}``: an LLM-backed built-in with its judge model;
* ``{"type": "llm_judge", "criteria": "...", "scale": "1-5", "threshold": 0.6,
  "model": {...}, "name": "tone"}``;
* ``{"type": "geval", "criteria": "...", "evaluation_steps": [...],
  "rubric": [[1, "wrong"], [5, "right"]], "model": {...}}``.

A plain string is the built-in of that name. :func:`build_scorer` validates
everything up front (unknown names, parameters a scorer does not take, a regex
that does not compile) and raises :class:`EvalError` with a readable message;
it never contacts a model. :data:`CATALOG` is what the UI offers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval.scorer import Scorer

ScorerFamily = Literal["deterministic", "llm", "trajectory", "rag"]


@dataclass(frozen=True)
class Param:
    name: str
    type: Literal["string", "number", "integer"]
    required: bool = False
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "description": self.description,
        }


@dataclass(frozen=True)
class CatalogEntry:
    """One scorer the UI can offer."""

    name: str
    family: ScorerFamily
    description: str
    params: tuple[Param, ...] = ()
    needs: tuple[str, ...] = ()
    """Row fields it reads: ``expected_output``, ``expected_trajectory``,
    ``expected_tool_calls``, ``expected_steps``, ``context``."""
    judge: bool = False
    """It calls a model, so it needs a judge model with a key."""
    needs_agent: bool = False
    """It reads what the agent did (tool calls), so an echo run cannot feed it."""
    factory: Callable[..., Scorer] | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "needs": list(self.needs),
            "judge": self.judge,
            "needs_agent": self.needs_agent,
        }


def _builtin(name: str) -> Callable[..., Scorer]:
    def make(**kw: Any) -> Scorer:
        from fastaiagent.eval.builtins import BUILTIN_SCORERS

        return BUILTIN_SCORERS[name](**kw)

    return make


def _trajectory(cls_name: str) -> Callable[..., Scorer]:
    def make(**kw: Any) -> Scorer:
        from fastaiagent.eval import trajectory

        return getattr(trajectory, cls_name)(**kw)  # type: ignore[no-any-return]

    return make


def _det(
    name: str,
    desc: str,
    needs: tuple[str, ...] = (),
    params: tuple[Param, ...] = (),
    needs_agent: bool = False,
) -> CatalogEntry:
    return CatalogEntry(
        name,
        "deterministic",
        desc,
        params=params,
        needs=needs,
        needs_agent=needs_agent,
        factory=_builtin(name),
    )


def _judged(
    name: str, desc: str, family: ScorerFamily = "llm", needs: tuple[str, ...] = ()
) -> CatalogEntry:
    return CatalogEntry(name, family, desc, needs=needs, judge=True, factory=_builtin(name))


def _traj(name: str, cls: str, desc: str, needs: tuple[str, ...] = ()) -> CatalogEntry:
    return CatalogEntry(
        name, "trajectory", desc, needs=needs, needs_agent=True, factory=_trajectory(cls)
    )


def _criteria_params(scale: str) -> tuple[Param, ...]:
    return (
        Param("criteria", "string", required=True),
        Param("scale", "string", default=scale, description="binary, 0-1, 1-5 or 1-10"),
        Param("threshold", "number", default=0.5, minimum=0, maximum=1),
    )


def _entries() -> tuple[CatalogEntry, ...]:
    exp = ("expected_output",)
    traj = ("expected_trajectory",)
    ctx = ("context",)
    return (
        # Deterministic
        _det("exact_match", "Output equals the expected output.", exp),
        _det("contains", "Output contains the expected output.", exp),
        _det("json_valid", "Output is valid JSON."),
        _det(
            "regex_match",
            "Output matches a regular expression.",
            params=(
                Param("pattern", "string", required=True, description="Python regex, ≤ 500 chars."),
            ),
        ),
        _det(
            "length_between",
            "Output length is within bounds.",
            params=(
                Param("min_len", "integer", default=0, minimum=0),
                Param("max_len", "integer", default=10000, minimum=0),
            ),
        ),
        # An echo (dataset check) takes no time and costs nothing: these two
        # measure an agent's run, so they need one.
        _det(
            "latency",
            "The run finished within a time budget.",
            params=(Param("max_ms", "integer", default=5000, minimum=1),),
            needs_agent=True,
        ),
        _det(
            "cost_under",
            "The run cost less than a budget (an unpriced model is errored, not free).",
            params=(Param("max_usd", "number", default=0.1, minimum=0),),
            needs_agent=True,
        ),
        _det("bleu", "BLEU overlap with the expected output.", exp),
        _det("rouge", "ROUGE overlap with the expected output.", exp),
        _det("levenshtein", "Edit-distance similarity to the expected output.", exp),
        _det("pii_leakage", "No email, phone, SSN or card number in the output."),
        _det("prompt_injection", "Heuristic prompt-injection check on the input."),
        # LLM-backed
        _judged("task_completion", "A judge rates whether the task was completed."),
        _judged("toxicity", "A judge rates the output's toxicity."),
        _judged("bias", "A judge rates the output's bias."),
        _judged("answer_relevancy", "A judge rates how relevant the answer is."),
        CatalogEntry(
            "llm_judge",
            "llm",
            "Your own criteria, judged by a model.",
            params=_criteria_params("binary"),
            judge=True,
        ),
        CatalogEntry(
            "geval",
            "llm",
            "G-Eval: criteria, evaluation steps and a rubric, judged by a model.",
            params=_criteria_params("1-5"),
            judge=True,
        ),
        # RAG: per-row context reaches the scorers since 1.89.0.
        _judged(
            "faithfulness", "Every output claim is supported by the row's context.", "rag", ctx
        ),
        _judged(
            "hallucination", "The output adds nothing the context does not support.", "rag", ctx
        ),
        _judged(
            "context_precision",
            "The row's context chunks are relevant to the question.",
            "rag",
            ctx,
        ),
        _judged(
            "context_recall",
            "The context covers the expected answer's claims.",
            "rag",
            (*ctx, *exp),
        ),
        # Trajectory: what the agent did, against each row's expectation.
        _traj(
            "tool_usage_accuracy", "ToolUsageAccuracy", "The agent used the expected tools.", traj
        ),
        _traj(
            "path_correctness",
            "PathCorrectness",
            "The agent called the tools in the expected order.",
            traj,
        ),
        _traj(
            "tool_call_correctness",
            "ToolCallCorrectness",
            "The agent's tool calls match by name and arguments.",
            ("expected_tool_calls",),
        ),
        _traj(
            "step_efficiency",
            "StepEfficiency",
            "The agent took no more steps than expected.",
            ("expected_steps",),
        ),
        _traj(
            "cycle_efficiency",
            "CycleEfficiency",
            "The agent did not repeat the same tool call back to back.",
        ),
    )


#: What the UI offers, by name. DecisionJudge and OpenAI moderation are left
#: out: they need a specific provider. Session scorers need conversation turns,
#: which a dataset row does not carry.
CATALOG: dict[str, CatalogEntry] = {e.name: e for e in _entries()}

_SCALES = ("binary", "0-1", "1-5", "1-10")


class JudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=200)


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuiltinSpec(_Spec):
    type: Literal["builtin"] = "builtin"
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    model: JudgeModel | None = None


class LLMJudgeSpec(_Spec):
    type: Literal["llm_judge"]
    name: str | None = Field(default=None, max_length=64)
    criteria: str = Field(min_length=1, max_length=4000)
    scale: str = "binary"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model: JudgeModel | None = None

    @field_validator("scale")
    @classmethod
    def _scale(cls, v: str) -> str:
        if v not in _SCALES:
            raise ValueError(f"scale must be one of {', '.join(_SCALES)}")
        return v


class GEvalSpec(LLMJudgeSpec):
    type: Literal["geval"]  # type: ignore[assignment]
    scale: str = "1-5"
    evaluation_steps: list[str] | None = Field(default=None, max_length=20)
    rubric: list[tuple[float, str]] | None = Field(default=None, max_length=20)


def _readable(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        where = ".".join(str(x) for x in e.get("loc", ()) if x not in ("body",))
        parts.append(f"{where}: {e.get('msg')}" if where else str(e.get("msg")))
    return "; ".join(parts)


def parse_spec(raw: str | dict[str, Any]) -> BuiltinSpec | LLMJudgeSpec | GEvalSpec:
    """A spec from a string or a dict; :class:`EvalError` with a readable reason."""
    if isinstance(raw, str):
        return BuiltinSpec(name=raw)
    if not isinstance(raw, dict):
        raise EvalError(f"A scorer must be a name or an object, not {type(raw).__name__}")
    kind = raw.get("type", "builtin")
    models: dict[str, type[BuiltinSpec] | type[LLMJudgeSpec]] = {
        "builtin": BuiltinSpec,
        "llm_judge": LLMJudgeSpec,
        "geval": GEvalSpec,
    }
    model = models.get(kind)
    if model is None:
        raise EvalError(f"Unknown scorer type {kind!r}: use builtin, llm_judge or geval")
    try:
        return model.model_validate({**raw, "type": kind})
    except ValidationError as e:
        raise EvalError(f"Scorer {raw.get('name') or kind}: {_readable(e)}") from e


def _check_params(entry: CatalogEntry, params: dict[str, Any]) -> dict[str, Any]:
    known = {p.name: p for p in entry.params}
    unknown = sorted(set(params) - set(known))
    if unknown:
        raise EvalError(f"Scorer '{entry.name}' takes no parameter {', '.join(unknown)}")
    out: dict[str, Any] = {}
    for p in entry.params:
        if p.name not in params or params[p.name] in (None, ""):
            if p.required:
                raise EvalError(f"Scorer '{entry.name}' needs {p.name}")
            continue
        value = params[p.name]
        try:
            if p.type == "integer":
                if isinstance(value, bool) or float(value) != int(float(value)):
                    raise ValueError
                value = int(float(value))
            elif p.type == "number":
                if isinstance(value, bool):
                    raise ValueError
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            raise EvalError(f"Scorer '{entry.name}': {p.name} must be a {p.type}") from None
        if p.minimum is not None and value < p.minimum:
            raise EvalError(f"Scorer '{entry.name}': {p.name} must be at least {p.minimum}")
        if p.maximum is not None and value > p.maximum:
            raise EvalError(f"Scorer '{entry.name}': {p.name} must be at most {p.maximum}")
        out[p.name] = value
    if entry.name == "regex_match":
        import re

        pattern = out["pattern"]
        if len(pattern) > 500:
            raise EvalError("Scorer 'regex_match': pattern is longer than 500 characters")
        try:
            re.compile(pattern)
        except re.error as e:
            raise EvalError(f"Scorer 'regex_match': the pattern does not compile: {e}") from e
    if entry.name == "length_between" and out.get("min_len", 0) > out.get("max_len", 10000):
        raise EvalError("Scorer 'length_between': min_len is greater than max_len")
    return out


def _client(model: JudgeModel | None) -> Any:
    from fastaiagent.llm import LLMClient

    if model is None:
        return LLMClient()
    return LLMClient(provider=model.provider, model=model.model)


def judge_provider(spec: BuiltinSpec | LLMJudgeSpec | GEvalSpec) -> str | None:
    """The provider a spec's judge would call, or ``None`` when it calls none."""
    if isinstance(spec, BuiltinSpec):
        entry = CATALOG.get(spec.name)
        if entry is None or not entry.judge:
            return None
    if spec.model is not None:
        return spec.model.provider
    from fastaiagent.llm import LLMClient

    return str(LLMClient().provider)


def build_scorer(raw: str | dict[str, Any]) -> Scorer:
    """One :class:`Scorer` from a name or a spec; :class:`EvalError` when invalid."""
    spec = parse_spec(raw)
    if isinstance(spec, BuiltinSpec):
        entry = CATALOG.get(spec.name)
        if entry is not None and entry.factory is None:
            raise EvalError(
                f"Scorer '{spec.name}' needs its criteria: send "
                f'{{"type": "{spec.name}", "criteria": "...", "model": {{...}}}}'
            )
        if entry is None or entry.factory is None:
            available = ", ".join(sorted(n for n, e in CATALOG.items() if e.factory))
            raise EvalError(f"Unknown scorer '{spec.name}'. Available: {available}")
        params = _check_params(entry, spec.params)
        if spec.model is not None and not entry.judge:
            raise EvalError(f"Scorer '{entry.name}' does not call a model; remove 'model'")
        if entry.judge:
            params["llm"] = _client(spec.model)
        return entry.factory(**params)
    from fastaiagent.eval.llm_judge import GEval, LLMJudge

    if isinstance(spec, GEvalSpec):
        return GEval(
            name=spec.name or "g_eval",
            criteria=spec.criteria,
            evaluation_steps=spec.evaluation_steps,
            rubric=[(r[0], r[1]) for r in spec.rubric] if spec.rubric else None,
            scale=spec.scale,
            threshold=spec.threshold,
            llm=_client(spec.model),
        )
    return LLMJudge(
        name=spec.name or "llm_judge",
        criteria=spec.criteria,
        scale=spec.scale,
        threshold=spec.threshold,
        llm=_client(spec.model),
    )


__all__ = [
    "CATALOG",
    "BuiltinSpec",
    "CatalogEntry",
    "GEvalSpec",
    "JudgeModel",
    "LLMJudgeSpec",
    "Param",
    "build_scorer",
    "judge_provider",
    "parse_spec",
]
