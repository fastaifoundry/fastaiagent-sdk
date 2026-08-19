"""Load and compare persisted eval runs (Agent CI baselines).

The Local UI has compared two runs since the EvalComparePage shipped, but the
logic lived inside the authed HTTP route — unreachable from Python or CI.
This module is the single home for that logic:

* :func:`load_run` — read one persisted run (+ cases) straight from
  ``local.db`` by ``run_id`` or ``run_name`` (latest wins).
* :func:`match_cases` / :func:`scorer_deltas` — the ordinal-then-input case
  alignment and per-scorer delta computation the UI route now delegates to.
* :func:`compare_runs` — regressed / improved / unchanged buckets plus the
  pass-rate delta, as a plain dataclass for pytest and the CLI to gate on.

Errored (infra-failed, unscored) cases are non-signal on either side: an
outage must not read as a regression or an improvement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastaiagent._internal.errors import EvalError

logger = logging.getLogger(__name__)

_JSON_CASE_KEYS = ("per_scorer", "input", "expected_output", "actual_output")
_JSON_RUN_KEYS = ("scorers", "metadata")


def _unpack(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out = dict(row)
    for key in keys:
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON for eval field %r", key, exc_info=True)
    return out


def case_outcome(case: dict[str, Any]) -> str:
    """'errored' for infra-failed cases, else 'passed'/'failed' by scorers."""
    if case.get("error"):
        return "errored"
    per = case.get("per_scorer") or {}
    if not isinstance(per, dict) or not per:
        return "passed"
    return (
        "passed" if all(isinstance(v, dict) and v.get("passed") for v in per.values()) else "failed"
    )


def scorer_deltas(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-scorer {passed_before, passed_after, changed} list for one case pair."""
    per_a = a.get("per_scorer") or {}
    per_b = b.get("per_scorer") or {}
    if not isinstance(per_a, dict) or not isinstance(per_b, dict):
        return []
    out = []
    for scorer in sorted(set(per_a) | set(per_b)):
        ra = per_a.get(scorer) or {}
        rb = per_b.get(scorer) or {}
        pa = bool(ra.get("passed")) if isinstance(ra, dict) else False
        pb = bool(rb.get("passed")) if isinstance(rb, dict) else False
        out.append(
            {
                "scorer": scorer,
                "passed_before": pa,
                "passed_after": pb,
                "changed": pa != pb,
            }
        )
    return out


def match_cases(
    cases_a: list[dict[str, Any]], cases_b: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Align cases across two runs: by ``ordinal`` first, by ``input`` equality
    as a fallback so reordered datasets still match. Unmatched cases drop."""
    index_b: dict[Any, dict[str, Any]] = {}
    for c in cases_b:
        key = c.get("ordinal")
        if key is not None:
            index_b[key] = c
    by_input: dict[str, dict[str, Any]] = {}
    for c in cases_b:
        try:
            by_input[json.dumps(c.get("input"), sort_keys=True)] = c
        except (TypeError, ValueError):
            logger.debug("Failed to serialize eval case input for comparison index", exc_info=True)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ca in cases_a:
        cb = index_b.get(ca.get("ordinal"))
        if cb is None:
            try:
                cb = by_input.get(json.dumps(ca.get("input"), sort_keys=True))
            except (TypeError, ValueError):
                logger.debug(
                    "Failed to serialize eval case input for comparison lookup", exc_info=True
                )
                cb = None
        if cb is not None:
            pairs.append((ca, cb))
    return pairs


@dataclass
class RunComparison:
    """Case-matched comparison of run ``a`` (baseline) vs run ``b`` (current)."""

    run_a: dict[str, Any]
    run_b: dict[str, Any]
    regressed: list[dict[str, Any]] = field(default_factory=list)
    improved: list[dict[str, Any]] = field(default_factory=list)
    unchanged_pass: int = 0
    unchanged_fail: int = 0
    pass_rate_delta: float = 0.0

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressed)

    def describe(self) -> list[str]:
        lines = [
            f"baseline: {self.run_a.get('run_name') or self.run_a.get('run_id')} "
            f"pass_rate={self.run_a.get('pass_rate') or 0.0:.4f}",
            f"current:  {self.run_b.get('run_name') or self.run_b.get('run_id')} "
            f"pass_rate={self.run_b.get('pass_rate') or 0.0:.4f} "
            f"(delta {self.pass_rate_delta:+.4f})",
            f"regressed={len(self.regressed)} improved={len(self.improved)} "
            f"unchanged_pass={self.unchanged_pass} unchanged_fail={self.unchanged_fail}",
        ]
        for entry in self.regressed:
            case = entry["a"]
            changed = [d["scorer"] for d in entry["scorer_deltas"] if d["changed"]]
            snippet = json.dumps(case.get("input"), default=str)
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            lines.append(f"  regressed: {snippet} (scorers: {', '.join(changed) or 'n/a'})")
        return lines


def bucket_cases(
    cases_a: list[dict[str, Any]], cases_b: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Bucket matched case pairs into (regressed, improved, unchanged_pass,
    unchanged_fail). Errored cases on either side are skipped entirely."""
    regressed: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    unchanged_pass = 0
    unchanged_fail = 0
    for ca, cb in match_cases(cases_a, cases_b):
        if case_outcome(ca) == "errored" or case_outcome(cb) == "errored":
            continue
        a_ok = case_outcome(ca) == "passed"
        b_ok = case_outcome(cb) == "passed"
        entry = {"a": ca, "b": cb, "scorer_deltas": scorer_deltas(ca, cb)}
        if a_ok and not b_ok:
            regressed.append(entry)
        elif b_ok and not a_ok:
            improved.append(entry)
        elif a_ok and b_ok:
            unchanged_pass += 1
        else:
            unchanged_fail += 1
    return regressed, improved, unchanged_pass, unchanged_fail


def load_run(run_ref: str, *, db_path: str | Path | None = None) -> dict[str, Any]:
    """Load one persisted eval run as ``{"run": {...}, "cases": [...]}``.

    ``run_ref`` is a ``run_id`` (exact) or a ``run_name`` (the most recent
    run with that name wins — so ``--eval-baseline main`` selects the last
    persisted run named ``main``). Raises :class:`EvalError` when not found.
    """
    from fastaiagent._internal.config import get_config
    from fastaiagent.ui.db import init_local_db

    resolved = Path(db_path) if db_path is not None else Path(get_config().local_db_path)
    if not resolved.exists():
        raise EvalError(f"No local eval DB at {resolved} — nothing has been persisted yet")
    db = init_local_db(resolved)
    try:
        row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_ref,))
        if row is None:
            row = db.fetchone(
                "SELECT * FROM eval_runs WHERE run_name = ? ORDER BY started_at DESC LIMIT 1",
                (run_ref,),
            )
        if row is None:
            raise EvalError(f"Eval run {run_ref!r} not found (tried run_id, then latest run_name)")
        cases = db.fetchall(
            "SELECT * FROM eval_cases WHERE run_id = ? ORDER BY ordinal",
            (row["run_id"],),
        )
        return {
            "run": _unpack(dict(row), _JSON_RUN_KEYS),
            "cases": [_unpack(dict(c), _JSON_CASE_KEYS) for c in cases],
        }
    finally:
        db.close()


def compare_runs(
    baseline: str | dict[str, Any],
    current: str | dict[str, Any],
    *,
    db_path: str | Path | None = None,
) -> RunComparison:
    """Compare two persisted runs (baseline first). Accepts run refs (see
    :func:`load_run`) or already-loaded ``{"run", "cases"}`` dicts."""
    a = load_run(baseline, db_path=db_path) if isinstance(baseline, str) else baseline
    b = load_run(current, db_path=db_path) if isinstance(current, str) else current
    regressed, improved, unchanged_pass, unchanged_fail = bucket_cases(a["cases"], b["cases"])
    return RunComparison(
        run_a=a["run"],
        run_b=b["run"],
        regressed=regressed,
        improved=improved,
        unchanged_pass=unchanged_pass,
        unchanged_fail=unchanged_fail,
        pass_rate_delta=round(
            (b["run"].get("pass_rate") or 0.0) - (a["run"].get("pass_rate") or 0.0), 4
        ),
    )
