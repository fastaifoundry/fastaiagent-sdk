"""Scoring one case, shared by ``evaluate()`` and the pytest plugin (1.89.0).

Before, each had its own loop, and both let a raising scorer escape: in
``evaluate()`` one exception aborted the whole run and nothing was persisted; in
the pytest plugin the case was never added to the session run. A scorer that
raises is now recorded as **errored** for that case, like a judge outage.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval.scorer import Scorer, ScorerResult

logger = logging.getLogger(__name__)

#: Per-case dataset keys forwarded to scorers. A row's value overrides a
#: run-wide ``evaluate(**kwargs)`` value for these names only.
ROW_KEYS: tuple[str, ...] = (
    "expected_trajectory",
    "expected_tool_calls",
    "expected_steps",
    "context",
    "contexts",
)


def resolve_scorers(scorers: list[Scorer | str] | None) -> list[Scorer]:
    """Scorer instances for ``scorers`` (default ``["exact_match"]``).

    Raises :class:`EvalError` for an unknown name, for a built-in that needs
    constructor arguments (``regex_match`` needs a pattern), and for two scorers
    with the same name, whose verdicts would overwrite each other in the case
    record (1.89.0; they were silently merged).
    """
    from fastaiagent.eval.builtins import BUILTIN_SCORERS

    out: list[Scorer] = []
    for s in scorers or ["exact_match"]:
        if isinstance(s, str):
            cls = BUILTIN_SCORERS.get(s)
            if cls is None:
                available = ", ".join(sorted(BUILTIN_SCORERS))
                raise EvalError(
                    f"Unknown scorer '{s}'. Available built-in scorers: {available}. "
                    "Or pass a Scorer instance directly."
                )
            required = _required_init_args(cls)
            if required:
                raise EvalError(
                    f"Scorer '{s}' needs {', '.join(required)}; pass an instance, e.g. "
                    f"{cls.__name__}({', '.join(f'{r}=…' for r in required)})."
                )
            out.append(cls())
        else:
            out.append(s)
    seen: set[str] = set()
    for scorer in out:
        if scorer.name in seen:
            raise EvalError(
                f"Two scorers are named '{scorer.name}'; their verdicts would overwrite each "
                "other. Give one a distinct name (scorer.name = …)."
            )
        seen.add(scorer.name)
    return out


def _required_init_args(cls: type) -> list[str]:
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, p in params.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def trajectory_facts(output: Any) -> dict[str, Any]:
    """What the agent did, from an ``AgentResult``'s ``tool_calls`` (1.89.0).

    ``evaluate()`` never passed it, so every trajectory scorer read an empty
    record and passed. Returns ``{}`` when the output carries no tool-call
    record (a plain string): the trajectory scorers then report errored.
    """
    calls = getattr(output, "tool_calls", None)
    if not isinstance(calls, list):
        return {}
    names: list[str] = []
    structured: list[dict[str, Any]] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("tool_name") or c.get("name")
        if name is None:
            continue
        names.append(str(name))
        structured.append({"name": str(name), "arguments": c.get("arguments", {})})
    return {
        "actual_trajectory": names,
        "actual_tool_calls": structured,
        "actual_steps": len(names),
    }


def row_facts(item: Any) -> dict[str, Any]:
    """The per-case scorer inputs a dataset row carries (see :data:`ROW_KEYS`)."""
    if not isinstance(item, dict):
        return {}
    return {k: item[k] for k in ROW_KEYS if item.get(k) is not None}


@dataclass
class EvalSpanContext:
    """Who a case's spans belong to (1.89.0). Structural only, never content."""

    run_id: str | None = None
    run_name: str | None = None
    kind: str | None = None
    case_ordinal: int | None = None

    @property
    def source(self) -> str:
        """``optimize`` for an AutoLLM candidate run, ``eval`` otherwise."""
        return "optimize" if self.kind == "optimize" else "eval"

    def attributes(self, **extra: Any) -> dict[str, Any]:
        out = {
            "fastaiagent.eval.run_id": self.run_id,
            "fastaiagent.eval.run_name": self.run_name,
            "fastaiagent.eval.kind": self.kind,
            "fastaiagent.eval.case_ordinal": self.case_ordinal,
        }
        out.update({f"fastaiagent.eval.{k}": v for k, v in extra.items()})
        return {k: v for k, v in out.items() if v is not None}


def agent_scope(ctx: EvalSpanContext | None) -> Any:
    """The scope the agent call runs in: every span of the case's agent trace
    is stamped ``fastaiagent.source=eval`` (``optimize`` for AutoLLM), so the
    production views leave it out (D10, 1.89.0). It is still exported: the
    plane corroborates eval verdicts against these traces."""
    if ctx is None:
        return contextlib.nullcontext()
    from fastaiagent.trace.provenance import provenance_scope

    return provenance_scope(ctx.source, attributes=ctx.attributes())


@contextlib.contextmanager
def _scoring_span(ctx: EvalSpanContext | None, subject_trace_id: str | None) -> Iterator[Any]:
    """One root ``eval.score`` span per scored case; ``None`` when tracing is off.

    A root of its own (E3): the judges' ``llm.*`` calls used to land as loose
    root traces, and as a child of the agent trace they would inflate the
    agent's cost. Best-effort: a tracing failure never stops scoring.
    """
    if ctx is None:
        yield None
        return
    stack = contextlib.ExitStack()
    span = None
    try:
        from opentelemetry import context as otel_context

        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.provenance import provenance_scope

        stack.enter_context(
            provenance_scope(
                ctx.source, attributes=ctx.attributes(subject_trace_id=subject_trace_id)
            )
        )
        span = stack.enter_context(
            get_tracer("fastaiagent.eval").start_as_current_span(
                "eval.score", context=otel_context.Context()
            )
        )
    except Exception:  # pragma: no cover - tracing is bookkeeping
        logger.debug("Could not open the eval.score span", exc_info=True)
    with stack:
        yield span


def _scorer_span() -> Any:
    try:
        from fastaiagent.trace.otel import get_tracer

        return get_tracer("fastaiagent.eval").start_as_current_span("eval.scorer")
    except Exception:  # pragma: no cover
        return contextlib.nullcontext()


def _trace_id_of(span: Any) -> str | None:
    try:
        sc = span.get_span_context()
        return format(sc.trace_id, "032x") if sc.trace_id else None
    except Exception:
        return None


def score_case(
    scorers: list[Scorer],
    *,
    input: Any,
    output: Any,
    expected: Any,
    score_kwargs: dict[str, Any] | None = None,
    span_context: EvalSpanContext | None = None,
    subject_trace_id: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, ScorerResult]], str | None]:
    """Run every scorer on one case. Returns ``(per_scorer, scored, scoring_trace_id)``.

    A scorer that raises, or returns something that is not a ``ScorerResult``,
    is recorded as errored for this case; the others still run.

    With ``span_context`` the scoring runs inside one ``eval.score`` root span,
    with an ``eval.scorer`` child per scorer carrying only structural
    attributes (name, outcome, score; never the reason, which is payload). A
    judge's ``llm.*`` call nests beneath. ``scoring_trace_id`` is that trace.
    """
    per_scorer: dict[str, dict[str, Any]] = {}
    scored: list[tuple[str, ScorerResult]] = []
    scoring_trace_id: str | None = None
    with _scoring_span(span_context, subject_trace_id) as root:
        if root is not None:
            scoring_trace_id = _trace_id_of(root)
        for scorer in scorers:
            with _scorer_span() if root is not None else contextlib.nullcontext() as child:
                try:
                    result = scorer.score(
                        input=input, output=output, expected=expected, **(score_kwargs or {})
                    )
                    if not isinstance(result, ScorerResult):
                        result = ScorerResult.error(
                            f"scorer returned {type(result).__name__}, not a ScorerResult"
                        )
                except Exception as e:
                    result = ScorerResult.error(f"{type(e).__name__}: {e}"[:500])
                if child is not None:
                    _stamp_scorer(child, scorer.name, result)
            scored.append((scorer.name, result))
            per_scorer[scorer.name] = result.stored()
    return per_scorer, scored, scoring_trace_id


def _stamp_scorer(span: Any, name: str, result: ScorerResult) -> None:
    try:
        from fastaiagent.trace.span import set_fastaiagent_attributes

        set_fastaiagent_attributes(
            span,
            **{
                "eval.scorer": name,
                "eval.outcome": result.outcome,
                "eval.score": result.score,
                "eval.passed": result.passed,
            },
        )
        if result.outcome == "errored":
            from opentelemetry.trace import Status, StatusCode

            # The check could not run: an ERROR span, so the trace shows it.
            span.set_status(Status(StatusCode.ERROR, "scorer could not run"))
    except Exception:  # pragma: no cover
        logger.debug("Could not stamp the eval.scorer span", exc_info=True)


def scorer_config(scorers: list[Scorer]) -> dict[str, dict[str, Any]]:
    """``{name: scorer.describe()}`` for the run record. Never raises."""
    out: dict[str, dict[str, Any]] = {}
    for s in scorers:
        try:
            out[s.name] = s.describe()
        except Exception:
            out[s.name] = {"type": type(s).__name__}
    return out


def agent_fingerprint(agent_fn: Any) -> str | None:
    """A stable short fingerprint of the agent behind ``agent_fn`` (1.89.0).

    ``agent_version`` was always NULL. For a bound method of an ``Agent`` (or
    anything with ``to_dict()``) it is the SHA-256 of that definition, so two
    runs of the same agent share it and a changed prompt or tool does not.
    """
    import functools

    target: Any = agent_fn
    while isinstance(target, functools.partial):
        target = target.func
    owner = getattr(target, "__self__", None)
    to_dict = getattr(owner, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        blob = json.dumps(to_dict(), sort_keys=True, default=str)
    except Exception:
        return None
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
