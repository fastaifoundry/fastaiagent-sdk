"""Scoring one case, shared by ``evaluate()`` and the pytest plugin (1.89.0).

Before, each had its own loop, and both let a raising scorer escape: in
``evaluate()`` one exception aborted the whole run and nothing was persisted; in
the pytest plugin the case was never added to the session run. A scorer that
raises is now recorded as **errored** for that case, like a judge outage.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval.scorer import Scorer, ScorerResult

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


def score_case(
    scorers: list[Scorer],
    *,
    input: Any,
    output: Any,
    expected: Any,
    score_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, ScorerResult]]]:
    """Run every scorer on one case. Returns ``(per_scorer, scored)``.

    A scorer that raises, or returns something that is not a ``ScorerResult``,
    is recorded as errored for this case; the others still run.
    """
    per_scorer: dict[str, dict[str, Any]] = {}
    scored: list[tuple[str, ScorerResult]] = []
    for scorer in scorers:
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
        scored.append((scorer.name, result))
        per_scorer[scorer.name] = result.stored()
    return per_scorer, scored


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
