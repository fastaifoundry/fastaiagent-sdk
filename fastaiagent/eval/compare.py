"""Load and compare persisted eval runs (Agent CI baselines).

The Local UI has compared two runs since the EvalComparePage shipped, but the
logic lived inside the authed HTTP route — unreachable from Python or CI.
This module is the single home for that logic:

* :func:`load_run` — read one persisted run (+ cases) straight from
  ``local.db`` by ``run_id`` or ``run_name`` (latest wins).
* :func:`align_cases` / :func:`scorer_deltas` — the ordinal-then-input case
  alignment and per-scorer delta computation the UI route now delegates to.
* :func:`compare_cases` — every case in one bucket: regressed / improved /
  unchanged / errored / only in one run, plus score drift (1.89.0).
* :func:`compare_runs` — the same, plus the rate deltas, as a plain dataclass
  for pytest and the CLI to gate on.

Errored (infra-failed, unscored) cases are non-signal on either side: an
outage must not read as a regression or an improvement. Since 1.89.0 they are
listed in their own bucket instead of being dropped.
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


def case_outcome_detail(case: dict[str, Any]) -> tuple[str, str | None]:
    """``(outcome, errored_kind)`` for one stored case. The one shared rule (1.89.0).

    First match wins:

    1. ``("errored", "agent")`` — the agent run failed or paused (``error``).
    2. ``("failed", None)`` — any applicable verdict failed. A fail stays a fail
       even when another scorer errored.
    3. ``("errored", "scorer")`` — a verdict errored (the check could not run).
    4. ``("errored", "unscored")`` — no verdict applied (all skipped, or none).
       This was ``"passed"``: a case nothing checked read as a pass.
    5. ``("passed", None)``.

    Mirrored by ``evalCaseOutcome`` in the UI.
    """
    from fastaiagent.eval.scorer import verdict_outcome

    if case.get("error"):
        return "errored", "agent"
    per = case.get("per_scorer") or {}
    if not isinstance(per, dict):
        per = {}
    outcomes = [verdict_outcome(v) for v in per.values()]
    if "failed" in outcomes:
        return "failed", None
    if "errored" in outcomes:
        return "errored", "scorer"
    if "passed" not in outcomes:
        return "errored", "unscored"
    return "passed", None


def case_outcome(case: dict[str, Any]) -> str:
    """``passed`` | ``failed`` | ``errored``; see :func:`case_outcome_detail`."""
    return case_outcome_detail(case)[0]


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Case and check counts for a run's stored cases (1.89.0).

    *Cases*: passed / failed / errored, and ``case_pass_rate`` = passed over
    passed + failed (``None`` when no case reached a verdict). *Checks*: every
    applicable verdict; ``check_pass_rate`` is passed over passed + failed, the
    rate ``overall.pass_rate`` gates and ``eval_runs.pass_rate`` stores. Errored
    and skipped verdicts are counted, never rated.
    """
    from fastaiagent.eval.scorer import verdict_outcome

    counts = {"passed": 0, "failed": 0, "errored": 0}
    checks = {"passed": 0, "failed": 0, "errored": 0, "skipped": 0}
    for case in cases:
        counts[case_outcome(case)] += 1
        if case.get("error"):
            continue
        per = case.get("per_scorer") or {}
        if isinstance(per, dict):
            for v in per.values():
                checks[verdict_outcome(v)] += 1
    judged = counts["passed"] + counts["failed"]
    rated = checks["passed"] + checks["failed"]
    return {
        "case_count": len(cases),
        "case_pass_count": counts["passed"],
        "case_fail_count": counts["failed"],
        "case_error_count": counts["errored"],
        "case_pass_rate": round(counts["passed"] / judged, 4) if judged else None,
        "check_pass_count": checks["passed"],
        "check_fail_count": checks["failed"],
        "check_error_count": checks["errored"],
        "check_skip_count": checks["skipped"],
        "check_pass_rate": round(checks["passed"] / rated, 4) if rated else None,
    }


def _score_of(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def scorer_deltas(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-scorer deltas for one case pair.

    ``passed_before`` / ``passed_after`` / ``changed`` as before; 1.89.0 adds
    ``outcome_before`` / ``outcome_after`` (passed, failed, errored, skipped or
    ``None`` when the scorer did not run on that side) and ``score_before`` /
    ``score_after`` / ``score_delta``. An errored or skipped verdict's score is
    ``None``, never the placeholder 0.0.
    """
    from fastaiagent.eval.scorer import verdict_outcome

    per_a = a.get("per_scorer") or {}
    per_b = b.get("per_scorer") or {}
    if not isinstance(per_a, dict) or not isinstance(per_b, dict):
        return []
    out = []
    for scorer in sorted(set(per_a) | set(per_b)):
        ra = per_a.get(scorer)
        rb = per_b.get(scorer)
        pa = bool(ra.get("passed")) if isinstance(ra, dict) else False
        pb = bool(rb.get("passed")) if isinstance(rb, dict) else False
        oa = verdict_outcome(ra) if isinstance(ra, dict) else None
        ob = verdict_outcome(rb) if isinstance(rb, dict) else None
        sa = _score_of(ra) if oa in ("passed", "failed") else None
        sb = _score_of(rb) if ob in ("passed", "failed") else None
        out.append(
            {
                "scorer": scorer,
                "passed_before": pa,
                "passed_after": pb,
                "changed": pa != pb,
                "outcome_before": oa,
                "outcome_after": ob,
                "score_before": sa,
                "score_after": sb,
                "score_delta": round(sb - sa, 4) if sa is not None and sb is not None else None,
            }
        )
    return out


def align_cases(
    cases_a: list[dict[str, Any]], cases_b: list[dict[str, Any]]
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """``(pairs, only_in_a, only_in_b)``: align cases across two runs.

    By ``ordinal`` when both runs hold the same input there, by ``input``
    equality otherwise, so reordered datasets still match. Each ``b`` case is
    used at most once (1.89.0): duplicate inputs pair in order instead of all
    pairing with the last duplicate. Cases with no partner are returned rather
    than dropped, so a compare can say what it could not compare.

    The input check matters: before 1.87.0 ``evaluate()`` stored ordinals in the
    order its concurrent cases *finished*, so the same ordinal in two runs of one
    dataset could hold different inputs — and a comparison paired them anyway.
    """

    def input_key(case: dict[str, Any]) -> str | None:
        try:
            return json.dumps(case.get("input"), sort_keys=True)
        except (TypeError, ValueError):
            logger.debug("Failed to serialize eval case input for comparison", exc_info=True)
            return None

    index_b: dict[Any, int] = {}
    by_input: dict[str, list[int]] = {}
    for i, c in enumerate(cases_b):
        key = c.get("ordinal")
        if key is not None:
            index_b.setdefault(key, i)
        ikey = input_key(c)
        if ikey is not None:
            by_input.setdefault(ikey, []).append(i)

    used: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    only_a: list[dict[str, Any]] = []
    for ca in cases_a:
        a_key = input_key(ca)
        j = index_b.get(ca.get("ordinal"))
        if j is not None and (j in used or (a_key is not None and input_key(cases_b[j]) != a_key)):
            j = None  # same position, different (or already paired) case
        if j is None and a_key is not None:
            j = next((k for k in by_input.get(a_key, ()) if k not in used), None)
        if j is None:
            only_a.append(ca)
            continue
        used.add(j)
        pairs.append((ca, cases_b[j]))
    only_b = [c for i, c in enumerate(cases_b) if i not in used]
    return pairs, only_a, only_b


def match_cases(
    cases_a: list[dict[str, Any]], cases_b: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The matched pairs of :func:`align_cases`; unmatched cases are left out."""
    return align_cases(cases_a, cases_b)[0]


@dataclass
class CaseComparison:
    """Every case of two runs, in exactly one of regressed / improved /
    unchanged / errored / only_in_a / only_in_b (1.89.0).

    ``errored`` holds pairs where either side errored, as
    ``{"a", "b", "side"}`` with ``side`` one of ``a`` / ``b`` / ``both``: an
    outage is neither a regression nor an improvement, but it is not silence
    either. ``drifted`` is an annotation on the unchanged pairs, not another
    bucket: those whose case outcome held while a scorer's score moved by at
    least ``drift_threshold`` (each entry lists only the scorers that moved).
    """

    regressed: list[dict[str, Any]] = field(default_factory=list)
    improved: list[dict[str, Any]] = field(default_factory=list)
    unchanged_pass: int = 0
    unchanged_fail: int = 0
    errored: list[dict[str, Any]] = field(default_factory=list)
    only_in_a: list[dict[str, Any]] = field(default_factory=list)
    only_in_b: list[dict[str, Any]] = field(default_factory=list)
    drifted: list[dict[str, Any]] = field(default_factory=list)


def compare_cases(
    cases_a: list[dict[str, Any]],
    cases_b: list[dict[str, Any]],
    *,
    drift_threshold: float = 0.1,
) -> CaseComparison:
    """Bucket two runs' cases; see :class:`CaseComparison`."""
    pairs, only_a, only_b = align_cases(cases_a, cases_b)
    out = CaseComparison(only_in_a=only_a, only_in_b=only_b)
    for ca, cb in pairs:
        oa, ob = case_outcome(ca), case_outcome(cb)
        deltas = scorer_deltas(ca, cb)
        entry = {"a": ca, "b": cb, "scorer_deltas": deltas}
        if oa == "errored" or ob == "errored":
            side = "both" if oa == ob else ("a" if oa == "errored" else "b")
            out.errored.append({**entry, "side": side})
        elif oa == "passed" and ob != "passed":
            out.regressed.append(entry)
        elif ob == "passed" and oa != "passed":
            out.improved.append(entry)
        else:
            if oa == "passed":
                out.unchanged_pass += 1
            else:
                out.unchanged_fail += 1
            moved = [
                d
                for d in deltas
                if d["score_delta"] is not None and abs(d["score_delta"]) >= drift_threshold
            ]
            if moved:
                out.drifted.append({**entry, "scorer_deltas": moved})
    return out


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
    """Checks rate delta (``eval_runs.pass_rate``). Stays a float, ``0.0`` when
    either side has no rate; read :attr:`pass_rate_comparable` first."""
    errored: list[dict[str, Any]] = field(default_factory=list)
    only_in_a: list[dict[str, Any]] = field(default_factory=list)
    only_in_b: list[dict[str, Any]] = field(default_factory=list)
    drifted: list[dict[str, Any]] = field(default_factory=list)
    case_pass_rate_delta: float | None = None
    """Cases rate delta; ``None`` when either run has no judged case."""
    pass_rate_comparable: bool = True
    """False when the two runs stored ``pass_rate`` under different rules: a
    run persisted before 1.89.0 counted errored and skipped verdicts as fails."""

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressed)

    def describe(self) -> list[str]:
        def rate(run: dict[str, Any]) -> str:
            value = run.get("pass_rate")
            return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

        lines = [
            f"baseline: {self.run_a.get('run_name') or self.run_a.get('run_id')} "
            f"pass_rate={rate(self.run_a)}",
            f"current:  {self.run_b.get('run_name') or self.run_b.get('run_id')} "
            f"pass_rate={rate(self.run_b)} "
            f"(delta {self.pass_rate_delta:+.4f})",
            f"regressed={len(self.regressed)} improved={len(self.improved)} "
            f"unchanged_pass={self.unchanged_pass} unchanged_fail={self.unchanged_fail}",
        ]
        if self.case_pass_rate_delta is not None:
            lines.append(f"cases pass_rate delta {self.case_pass_rate_delta:+.4f}")
        if self.errored or self.only_in_a or self.only_in_b or self.drifted:
            lines.append(
                f"errored={len(self.errored)} only_in_baseline={len(self.only_in_a)} "
                f"only_in_current={len(self.only_in_b)} score_drift={len(self.drifted)}"
            )
        if not self.pass_rate_comparable:
            lines.append(
                "warning: one run was persisted before 1.89.0, when pass_rate "
                "counted errored and skipped checks as failures; the pass_rate delta "
                "is not like for like. Re-run and re-publish the baseline."
            )
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
    """``(regressed, improved, unchanged_pass, unchanged_fail)`` of
    :func:`compare_cases`; errored and unmatched cases are left out."""
    c = compare_cases(cases_a, cases_b)
    return c.regressed, c.improved, c.unchanged_pass, c.unchanged_fail


def summary_version(run: dict[str, Any]) -> int:
    """``eval_runs.summary_version``; 1 for a run persisted before 1.89.0."""
    value = run.get("summary_version")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 1


#: SQL true for a run whose recorded gate did not reject it. A run never gated
#: counts as accepted; ``json_valid`` guards rows whose metadata is malformed.
_ACCEPTED_SQL = (
    "COALESCE(CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.gate_outcome') "
    "END, 'passed') = 'passed'"
)


def load_run(
    run_ref: str, *, db_path: str | Path | None = None, accepted_only: bool = False
) -> dict[str, Any]:
    """Load one persisted eval run as ``{"run": {...}, "cases": [...]}``.

    ``run_ref`` is a ``run_id`` (exact) or a ``run_name`` (the most recent
    run with that name wins — so ``--eval-baseline main`` selects the last
    persisted run named ``main``). Raises :class:`EvalError` when not found.

    ``accepted_only=True`` makes a *name* lookup skip runs whose recorded gate
    was ``failed`` or ``invalid``, so a regression can never become the next
    baseline just by being the newest run with that name (1.89.0). An exact
    ``run_id`` is always honoured.
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
            accepted = f" AND {_ACCEPTED_SQL}" if accepted_only else ""
            row = db.fetchone(
                f"SELECT * FROM eval_runs WHERE run_name = ?{accepted} "
                "ORDER BY started_at DESC, rowid DESC LIMIT 1",
                (run_ref,),
            )
        if row is None:
            which = "latest accepted run_name" if accepted_only else "latest run_name"
            raise EvalError(f"Eval run {run_ref!r} not found (tried run_id, then {which})")
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
    cmp = compare_cases(a["cases"], b["cases"])
    rate_a = summarize_cases(a["cases"])["case_pass_rate"]
    rate_b = summarize_cases(b["cases"])["case_pass_rate"]
    return RunComparison(
        run_a=a["run"],
        run_b=b["run"],
        regressed=cmp.regressed,
        improved=cmp.improved,
        unchanged_pass=cmp.unchanged_pass,
        unchanged_fail=cmp.unchanged_fail,
        pass_rate_delta=round(
            (b["run"].get("pass_rate") or 0.0) - (a["run"].get("pass_rate") or 0.0), 4
        ),
        errored=cmp.errored,
        only_in_a=cmp.only_in_a,
        only_in_b=cmp.only_in_b,
        drifted=cmp.drifted,
        case_pass_rate_delta=(
            round(rate_b - rate_a, 4) if rate_a is not None and rate_b is not None else None
        ),
        pass_rate_comparable=summary_version(a["run"]) == summary_version(b["run"]),
    )
