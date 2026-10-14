"""Aggregate quality gates over an eval run (Agent CI).

Shared by the pytest plugin (``--eval-fail-under``) and ``fastaiagent eval
run --fail-under`` so both surfaces gate with identical semantics:

* **Quality** thresholds — ``overall.pass_rate=0.9`` (the checks rate),
  ``cases.pass_rate=0.8`` (the case rate, 1.89.0), ``geval.avg_score=0.7``, or
  bare ``exact_match=0.9`` (bare scorer means its pass_rate).
* **Infra validity** — ``max_error_rate`` bounds the fraction of cases that
  infrastructure-failed (and were therefore never scored). A run over the
  bound — or with nothing scored at all — is *invalid*, not failed: an
  outage is not a quality signal, but it must never gate green either.

No LLM calls; pure aggregation over :class:`~fastaiagent.eval.results.EvalResults`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval.results import EvalResults, Scorecard

_FIELDS = ("pass_rate", "avg_score")


@dataclass
class Threshold:
    """One parsed ``--fail-under`` entry."""

    metric: str  # scorer name, or "overall"
    field: str  # "pass_rate" | "avg_score"
    minimum: float
    spec: str  # the raw user string, for error messages


@dataclass
class ThresholdCheck:
    """A threshold applied to a run. ``actual is None`` = metric not found."""

    threshold: Threshold
    actual: float | None
    passed: bool

    def describe(self) -> str:
        name = f"{self.threshold.metric}.{self.threshold.field}"
        if self.actual is None:
            return f"{name}: metric not present in this run (gate fails)"
        verdict = ">=" if self.passed else "<"
        return f"{name}: {self.actual:.4f} {verdict} {self.threshold.minimum} required"


@dataclass
class GateReport:
    """Outcome of gating one eval run."""

    checks: list[ThresholdCheck] = field(default_factory=list)
    scored: int = 0
    """Cases that reached a verdict (passed or failed)."""
    errored: int = 0
    """Cases whose outcome is errored: the agent failed, a scorer could not run,
    or no scorer applied (1.89.0; it counted agent failures only)."""
    error_rate: float = 0.0
    max_error_rate: float | None = None
    invalid_reasons: list[str] = field(default_factory=list)
    """Why a gated metric cannot be trusted (1.89.0): its scorer could not run
    on some cases, or no verdict of it applied."""

    @property
    def infra_invalid(self) -> bool:
        """True when infra failures disqualify the run from gating at all."""
        if self.max_error_rate is not None and self.error_rate > self.max_error_rate:
            return True
        if self.invalid_reasons:
            return True
        # Nothing scored: there is no quality signal to gate on.
        return self.scored == 0

    @property
    def quality_failed(self) -> bool:
        return any(not c.passed for c in self.checks)

    @property
    def outcome(self) -> str:
        """``invalid`` | ``failed`` | ``passed``.

        Invalid wins over failed: when the run's infra is broken, threshold
        misses are noise and must not be reported as agent regressions.
        """
        if self.infra_invalid:
            return "invalid"
        return "failed" if self.quality_failed else "passed"

    def describe(self) -> list[str]:
        # During an outage the threshold misses are noise — reporting them
        # alongside the invalid verdict sends readers debugging the agent.
        lines = [] if self.infra_invalid else [c.describe() for c in self.checks]
        lines.extend(f"invalid: {r}" for r in self.invalid_reasons)
        if self.errored:
            bound = (
                f", max allowed {self.max_error_rate}" if self.max_error_rate is not None else ""
            )
            lines.append(
                f"errored cases: {self.errored} of {self.scored + self.errored} "
                f"(error_rate={self.error_rate:.4f}{bound}) — unscored, excluded from pass/fail"
            )
        if self.scored == 0:
            lines.append("no cases were scored — run is invalid, not a quality verdict")
        return lines


#: Every verdict a gate can record, and the plane's ingest accepts.
VALID_GATE_OUTCOMES: tuple[str, ...] = ("passed", "failed", "invalid")


def parse_threshold(spec: str) -> Threshold:
    """Parse ``metric=value`` where metric is ``overall.pass_rate``,
    ``<scorer>.pass_rate``, ``<scorer>.avg_score``, or bare ``<scorer>``
    (meaning its pass_rate)."""
    if "=" not in spec:
        raise EvalError(
            f"Invalid threshold {spec!r}: expected '<metric>=<value>', "
            f"e.g. 'overall.pass_rate=0.9' or 'exact_match=0.9'"
        )
    name, _, raw_value = spec.partition("=")
    name = name.strip()
    try:
        minimum = float(raw_value.strip())
    except ValueError as e:
        raise EvalError(f"Invalid threshold {spec!r}: {raw_value.strip()!r} is not a number") from e
    if "." in name:
        metric, _, fld = name.rpartition(".")
        if fld not in _FIELDS:
            # A scorer name may itself contain a dot; only split on known fields.
            metric, fld = name, "pass_rate"
    else:
        metric, fld = name, "pass_rate"
    if not metric:
        raise EvalError(f"Invalid threshold {spec!r}: empty metric name")
    return Threshold(metric=metric, field=fld, minimum=minimum, spec=spec)


def gate(
    results: EvalResults,
    *,
    fail_under: list[str] | None = None,
    max_error_rate: float | None = None,
) -> GateReport:
    """Apply thresholds + infra validity to a run and return a :class:`GateReport`.

    Metrics: ``overall.pass_rate`` (the checks rate: applicable verdicts across
    scorers), ``cases.pass_rate`` (passed cases over passed + failed cases,
    1.89.0), ``<scorer>.pass_rate`` / ``<scorer>.avg_score``.

    A run is **invalid** (not failed) when:

    * ``max_error_rate`` is set and the case error rate exceeds it;
    * no case reached a verdict;
    * a gated metric's scorer errored on some cases and ``max_error_rate`` is
      unset, or its verdict error rate exceeds it (1.89.0: a judge outage is not
      a quality verdict, and must not pass either);
    * a gated metric has no applicable verdict at all (1.89.0).

    A misspelled metric still **fails**: a typo must never pass a build.
    """
    scorecard = Scorecard.from_eval_results(results)
    by_name = {m.name: m for m in scorecard.metrics}
    if results.cases:
        cc = results.case_counts
        scored_cases = cc["case_pass_count"] + cc["case_fail_count"]
        errored_cases = cc["case_error_count"]
        total_cases = cc["case_count"]
    else:
        # Callers that only used results.add() (no case records) still have
        # a real quality signal — don't misreport their run as invalid.
        scored_cases = max((m.n for m in scorecard.metrics), default=0)
        errored_cases = 0
        total_cases = scored_cases

    invalid: list[str] = []

    def check_validity(metric: Any, *, named: bool) -> None:
        # n == 0 invalidates a threshold on that scorer by name. Under
        # overall/cases a scorer that applied to no case adds nothing to the
        # rate (WS4 §1.6: candidate-C's overall.pass_rate=0.8 passes with
        # tool_usage_accuracy n/a ×4); its errors still invalidate.
        if metric.errored:
            rate = metric.errored / (metric.n + metric.errored)
            if max_error_rate is None or rate > max_error_rate:
                bound = (
                    f" > max_error_rate {max_error_rate}"
                    if max_error_rate is not None
                    else "; set max_error_rate to tolerate it"
                )
                invalid.append(
                    f"{metric.name}: the scorer could not run on {metric.errored} case(s) "
                    f"(verdict error rate {rate:.4f}{bound})"
                )
                return
        if named and metric.n == 0:
            invalid.append(
                f"{metric.name}: no applicable verdict (n=0, {metric.errored} errored, "
                f"{metric.skipped} not applicable) — nothing to gate on"
            )

    checks: list[ThresholdCheck] = []
    for spec in fail_under or []:
        t = parse_threshold(spec) if isinstance(spec, str) else spec
        actual: float | None
        gated: list[Any]
        if t.metric == "overall":
            if t.field == "avg_score":
                raise EvalError(
                    "overall.avg_score is not a metric (it silently gated on the pass rate "
                    "before 1.89.0). Gate on overall.pass_rate, cases.pass_rate, or "
                    "<scorer>.avg_score."
                )
            actual = scorecard.overall_pass_rate
            gated = list(scorecard.metrics)
        elif t.metric == "cases":
            if t.field != "pass_rate":
                raise EvalError(f"cases.{t.field} is not a metric; use cases.pass_rate.")
            actual = scorecard.cases.get("pass_rate") if scorecard.cases else None
            gated = list(scorecard.metrics)
        else:
            m = by_name.get(t.metric)
            actual = getattr(m, t.field) if m is not None else None
            gated = [m] if m is not None else []
        for metric in gated:
            check_validity(metric, named=t.metric not in ("overall", "cases"))
        # A missing metric fails the gate: a typo'd scorer name must never
        # silently pass a build.
        passed = actual is not None and actual >= t.minimum
        checks.append(ThresholdCheck(threshold=t, actual=actual, passed=passed))

    return GateReport(
        checks=checks,
        scored=scored_cases,
        errored=errored_cases,
        error_rate=round(errored_cases / total_cases, 4) if total_cases else 0.0,
        max_error_rate=max_error_rate,
        invalid_reasons=list(dict.fromkeys(invalid)),
    )
