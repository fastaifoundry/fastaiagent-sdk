"""Aggregate quality gates over an eval run (Agent CI).

Shared by the pytest plugin (``--eval-fail-under``) and ``fastaiagent eval
run --fail-under`` so both surfaces gate with identical semantics:

* **Quality** thresholds — ``overall.pass_rate=0.9``, ``geval.avg_score=0.7``,
  or bare ``exact_match=0.9`` (bare scorer means its pass_rate).
* **Infra validity** — ``max_error_rate`` bounds the fraction of cases that
  infrastructure-failed (and were therefore never scored). A run over the
  bound — or with nothing scored at all — is *invalid*, not failed: an
  outage is not a quality signal, but it must never gate green either.

No LLM calls; pure aggregation over :class:`~fastaiagent.eval.results.EvalResults`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    errored: int = 0
    error_rate: float = 0.0
    max_error_rate: float | None = None

    @property
    def infra_invalid(self) -> bool:
        """True when infra failures disqualify the run from gating at all."""
        if self.max_error_rate is not None and self.error_rate > self.max_error_rate:
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
    """Apply thresholds + infra validity to a run and return a :class:`GateReport`."""
    scorecard = Scorecard.from_eval_results(results)
    by_name = {m.name: m for m in scorecard.metrics}
    scored_cases = sum(1 for c in results.cases if not c.error)
    if not results.cases and results.scores:
        # Callers that only used results.add() (no case records) still have
        # a real quality signal — don't misreport their run as invalid.
        scored_cases = max(len(v) for v in results.scores.values())

    checks: list[ThresholdCheck] = []
    for spec in fail_under or []:
        t = parse_threshold(spec) if isinstance(spec, str) else spec
        actual: float | None
        if t.metric == "overall":
            actual = scorecard.overall_pass_rate
        else:
            m = by_name.get(t.metric)
            actual = getattr(m, t.field) if m is not None else None
        # A missing metric fails the gate: a typo'd scorer name must never
        # silently pass a build.
        passed = actual is not None and actual >= t.minimum
        checks.append(ThresholdCheck(threshold=t, actual=actual, passed=passed))

    return GateReport(
        checks=checks,
        scored=scored_cases,
        errored=results.errored_count,
        error_rate=round(results.error_rate, 4),
        max_error_rate=max_error_rate,
    )
