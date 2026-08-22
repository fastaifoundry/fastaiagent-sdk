"""Pytest plugin for fastaiagent's evaluation framework (Agent CI).

Registered via the ``[project.entry-points.pytest11]`` group in
``pyproject.toml`` so any project that has ``fastaiagent`` installed picks
it up automatically.

What you get:

- A :func:`case` decorator — turn a regular pytest function into a single
  evaluation case with ``input`` and ``expected``. The fixture
  ``evaluate_one`` exposes a one-call helper that runs the agent and
  scores against the expected output.

- A :func:`dataset` decorator — parametrise a test over every row of a
  JSONL or CSV dataset (uses :class:`fastaiagent.eval.Dataset`).

- ``evaluate_one`` fixture — runs a single eval inline, returns an
  :class:`fastaiagent.eval.results.EvalCaseRecord`, asserts pass on
  ``exact_match`` by default unless overridden.

- **Session aggregation** (1.48.0): all ``evaluate_one`` cases in a pytest
  session roll up into ONE ``eval_runs`` row (previously each case wrote its
  own single-case run, polluting the Local UI). Infra failures (agent raised)
  are recorded as *errored* — unscored, excluded from pass/fail, but counted.

- **Aggregate gates** (1.48.0), all optional:

  * ``--eval-fail-under "overall.pass_rate=0.9"`` (repeatable; also
    ``<scorer>.avg_score=…`` or bare ``<scorer>=…`` meaning its pass_rate)
  * ``--eval-max-error-rate 0.1`` — more errored cases than this and the
    run is INVALID (fails CI as infra, never reported as agent quality)
  * ``--eval-baseline <run_id|run_name>`` + ``--eval-tolerance 0.02`` —
    compare against a persisted baseline run; fail when the overall
    pass-rate drops more than the tolerance
  * ``--eval-run-name main`` — name this session's persisted run (so a
    later run can select it as ``--eval-baseline main``)

  A breached gate fails the pytest session (exit 1) and prints a
  ``fastaiagent eval`` terminal summary explaining exactly what missed.

Example:

    from fastaiagent.testing import TestModel
    from fastaiagent.agent import Agent
    from fastaiagent.eval import case

    @case(input="hello", expected="hi")
    def test_greet(evaluate_one):
        agent = Agent(name="g", llm=TestModel(response="hi"))
        evaluate_one(agent.run, scorers=["exact_match"])

The plugin is opt-in: tests that don't use these helpers are unaffected, and
no gate options means no gating. Not xdist-aware in this release (each worker
would persist its own run); run eval-gated suites in a single process.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.eval.builtins import BUILTIN_SCORERS
from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.evaluate import infer_agent_name
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import Scorer, ScorerResult

logger = logging.getLogger(__name__)

# Marker name we attach via raw setattr so the plugin can find decorated
# tests at fixture time. We avoid pytest.mark to keep the surface explicit.
_CASE_ATTR = "_fastaiagent_case"


def case(
    *,
    input: Any,
    expected: Any | None = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a pytest function as a single fastaiagent eval case.

    The ``evaluate_one`` fixture inside the test reads the tag and feeds
    ``input`` / ``expected`` automatically.

    Args:
        input: Whatever you'd pass to your agent's ``run()``.
        expected: Reference answer used by scorers like ``exact_match``.
        name: Optional case name (defaults to the test function name).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            fn,
            _CASE_ATTR,
            {"input": input, "expected": expected, "name": name or fn.__name__},
        )
        return fn

    return decorator


def dataset(
    path: str | Path,
    *,
    ids_from: str = "input",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Parametrise a test over every row of a JSONL/CSV dataset.

    Each row becomes a separate pytest invocation. The test signature must
    accept an ``eval_case`` argument (a dict with ``input`` and ``expected``
    keys); use the ``evaluate_one`` fixture to score against it.

    Args:
        path: Path to a ``.jsonl`` or ``.csv`` file.
        ids_from: Field used to label parametrised cases in pytest output.
    """
    p = Path(path)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if p.suffix == ".csv":
            ds = Dataset.from_csv(p)
        else:
            ds = Dataset.from_jsonl(p)
        cases = list(ds)
        ids = [str(c.get(ids_from, i)) for i, c in enumerate(cases)]
        wrapped: Callable[..., Any] = pytest.mark.parametrize("eval_case", cases, ids=ids)(fn)
        return wrapped

    return decorator


def _resolve_scorers(scorers: list[Scorer | str] | None) -> list[Scorer]:
    """Mirror ``aevaluate``'s scorer resolution but expose a clean error."""
    out: list[Scorer] = []
    for s in scorers or ["exact_match"]:
        if isinstance(s, str):
            cls = BUILTIN_SCORERS.get(s)
            if cls is None:
                pytest.fail(
                    f"Unknown scorer '{s}'. Available: {', '.join(sorted(BUILTIN_SCORERS))}."
                )
            out.append(cls())
        else:
            out.append(s)
    return out


def _format_failure(case_record: EvalCaseRecord) -> str:
    lines = [
        "fastaiagent eval case failed.",
        f"  input:    {case_record.input!r}",
        f"  expected: {case_record.expected_output!r}",
        f"  actual:   {case_record.actual_output!r}",
        "  scorers:",
    ]
    for name, info in (case_record.per_scorer or {}).items():
        passed = info.get("passed")
        score = info.get("score")
        reason = info.get("reason")
        lines.append(f"    - {name}: passed={passed} score={score} reason={reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session aggregation + gates (Agent CI)
# ---------------------------------------------------------------------------


@dataclass
class _CollectedCase:
    record: EvalCaseRecord
    scorer_results: dict[str, ScorerResult] = field(default_factory=dict)
    persist: bool = True
    # The agent this case exercised, when it can be inferred from the callable.
    # Connected mode needs it: the plane attaches eval evidence to a real agent,
    # so an unattributed run marks nothing.
    agent_name: str | None = None


@dataclass
class _SessionCollector:
    cases: list[_CollectedCase] = field(default_factory=list)
    run_id: str | None = None  # set at sessionfinish once persisted
    gate_lines: list[str] = field(default_factory=list)
    gate_failed: bool = False
    persist_error: str | None = None
    # Set when a --eval-baseline comparison ran; travels to the plane as evidence
    # of regression tracking (Part D).
    baseline_summary: dict[str, Any] | None = None

    def build_results(self, *, persisted_only: bool = False) -> EvalResults:
        results = EvalResults()
        for collected in self.cases:
            if persisted_only and not collected.persist:
                continue
            results.add_case(collected.record)
            for name, sr in collected.scorer_results.items():
                results.add(name, sr)
        return results


_COLLECTOR_KEY: pytest.StashKey[_SessionCollector] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("fastaiagent", "fastaiagent Agent CI eval gates")
    group.addoption(
        "--eval-fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Aggregate quality threshold over all evaluate_one cases, e.g. "
            "'overall.pass_rate=0.9', 'geval.avg_score=0.7', 'exact_match=0.9' "
            "(bare scorer = its pass_rate). Repeatable; any miss fails the session."
        ),
    )
    group.addoption(
        "--eval-max-error-rate",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "Maximum fraction of cases allowed to infrastructure-fail (agent "
            "raised; unscored). Above this the run is INVALID and the session fails."
        ),
    )
    group.addoption(
        "--eval-baseline",
        default=None,
        metavar="RUN",
        help=(
            "run_id or run_name of a persisted eval run to compare against; "
            "the session fails if overall pass-rate drops more than --eval-tolerance."
        ),
    )
    group.addoption(
        "--eval-tolerance",
        type=float,
        default=0.0,
        metavar="DELTA",
        help="Allowed overall pass-rate drop vs --eval-baseline (default 0.0).",
    )
    group.addoption(
        "--eval-run-name",
        default=None,
        metavar="NAME",
        help=(
            "run_name for this session's aggregated eval run "
            "(default 'pytest::<rootdir>'); use it to publish a baseline, "
            "e.g. --eval-run-name main."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_COLLECTOR_KEY] = _SessionCollector()


def _gating_requested(config: pytest.Config) -> bool:
    return bool(
        config.getoption("--eval-fail-under")
        or config.getoption("--eval-max-error-rate") is not None
        or config.getoption("--eval-baseline")
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Persist ONE aggregated eval run + apply gates.

    Only raises the exit status (0/`no tests` -> 1 on gate breach); never
    lowers it — genuinely failed tests keep the session red regardless.
    """
    config = session.config
    collector = config.stash.get(_COLLECTOR_KEY, None)
    if collector is None or not collector.cases:
        return

    from fastaiagent.eval.gate import gate as run_gate

    all_results = collector.build_results()

    # Persist one aggregated run (best-effort, never fails the session).
    run_name = config.getoption("--eval-run-name") or f"pytest::{config.rootpath.name}"
    persisted = collector.build_results(persisted_only=True)
    if persisted.cases:
        try:
            collector.run_id = persisted.persist_local(
                run_name=run_name, agent_name=_session_agent_name(collector)
            )
        except Exception as e:  # pragma: no cover — non-fatal
            collector.persist_error = str(e)
            logger.warning("Failed to persist aggregated pytest eval run", exc_info=True)

    if not _gating_requested(config):
        # No gate was demanded, but evals DID run — still evidence. Record a
        # trivial verdict (empty thresholds = "no gate demanded") so a connected
        # plane sees the activity instead of marking the agent as eval-less.
        _record_gate(collector, outcome=_ungated_outcome(all_results), thresholds={})
        return

    report = run_gate(
        all_results,
        fail_under=list(config.getoption("--eval-fail-under")),
        max_error_rate=config.getoption("--eval-max-error-rate"),
    )
    collector.gate_lines.extend(report.describe())

    baseline_ref = config.getoption("--eval-baseline")
    if baseline_ref and report.outcome != "invalid":
        collector.gate_failed |= _apply_baseline_gate(
            collector, all_results, baseline_ref, float(config.getoption("--eval-tolerance"))
        )

    _record_gate(
        collector,
        outcome=report.outcome,
        thresholds={
            f"{t.threshold.metric}.{t.threshold.field}": t.threshold.minimum for t in report.checks
        },
        baseline=collector.baseline_summary,
    )

    if report.outcome == "invalid":
        collector.gate_lines.append(
            "GATE: INVALID — infra failures disqualify this run (not an agent regression)"
        )
        collector.gate_failed = True
    elif report.outcome == "failed":
        collector.gate_lines.append("GATE: FAILED — quality threshold(s) missed")
        collector.gate_failed = True

    if collector.gate_failed and session.exitstatus in (0, 5):
        session.exitstatus = 1


def _session_agent_name(collector: _SessionCollector) -> str | None:
    """The agent this session evaluated, when the whole session agrees.

    A suite that exercises several agents has no single owner, so report none
    rather than crediting the evidence to whichever ran first.
    """
    names = {c.agent_name for c in collector.cases if c.agent_name}
    return names.pop() if len(names) == 1 else None


def _ungated_outcome(results: EvalResults) -> str:
    """Verdict for a run nobody gated: invalid when nothing was scored, else passed."""
    from fastaiagent.eval.gate import gate as run_gate

    return run_gate(results).outcome


def _record_gate(
    collector: _SessionCollector,
    *,
    outcome: str,
    thresholds: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> None:
    """Attach the verdict to the persisted run and offer it to the plane."""
    if not collector.run_id:
        return
    from fastaiagent.eval.results import record_gate_result

    record_gate_result(
        collector.run_id, gate_outcome=outcome, thresholds=thresholds, baseline=baseline
    )


def _apply_baseline_gate(
    collector: _SessionCollector,
    results: EvalResults,
    baseline_ref: str,
    tolerance: float,
) -> bool:
    """Compare vs the persisted baseline; returns True when the gate fails."""
    from fastaiagent.eval.compare import compare_runs
    from fastaiagent.eval.results import Scorecard

    try:
        current = {
            "run": {
                "run_id": collector.run_id or "(unpersisted)",
                "run_name": "(current session)",
                "pass_rate": Scorecard.from_eval_results(results).overall_pass_rate,
            },
            "cases": [
                {
                    "ordinal": i,
                    "input": c.record.input,
                    "per_scorer": c.record.per_scorer,
                    "error": c.record.error,
                }
                for i, c in enumerate(collector.cases)
            ],
        }
        comparison = compare_runs(baseline_ref, current)
    except Exception as e:
        collector.gate_lines.append(f"baseline comparison failed: {e} (gate fails)")
        return True

    collector.gate_lines.extend(comparison.describe())
    # Kept for the plane: a pass-rate alone evidences "evals ran"; the delta vs a
    # named baseline is what evidences "regressions tracked across versions".
    collector.baseline_summary = {
        "run_id": comparison.run_a.get("run_id"),
        "pass_rate_delta": comparison.pass_rate_delta,
        "regressed_count": len(comparison.regressed),
    }
    drop = -comparison.pass_rate_delta
    if drop > tolerance:
        collector.gate_lines.append(
            f"GATE: REGRESSION — overall pass-rate dropped {drop:.4f} "
            f"vs baseline (tolerance {tolerance})"
        )
        return True
    return False


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    collector = config.stash.get(_COLLECTOR_KEY, None)
    if collector is None or not collector.cases:
        return
    from fastaiagent.eval.results import Scorecard

    tr = terminalreporter
    tr.section("fastaiagent eval")
    scorecard = Scorecard.from_eval_results(collector.build_results())
    for line in scorecard.summary().splitlines():
        tr.write_line(line)
    if collector.run_id:
        tr.write_line(f"persisted run_id={collector.run_id} (Local UI: /evals/{collector.run_id})")
    elif collector.persist_error:
        tr.write_line(f"persist failed: {collector.persist_error}")
    for line in collector.gate_lines:
        tr.write_line(line)


@pytest.fixture
def evaluate_one(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    """Run a single agent invocation and score it against ``expected``.

    Reads the ``@case(input=..., expected=...)`` tag from the calling test
    or falls back to a parametrised ``eval_case`` arg from ``@dataset(...)``.

    All cases in the session aggregate into ONE persisted ``eval_runs`` row
    (see the module docstring); an agent exception records the case as
    *errored* (unscored) and still fails the test loudly.

    Returns the helper as a callable so the test body retains control over
    timing, error handling, and any assertions on the result besides the
    eval pass/fail.
    """
    test_fn = request.function
    case_meta = getattr(test_fn, _CASE_ATTR, None)
    eval_case = (
        request.getfixturevalue("eval_case") if "eval_case" in request.fixturenames else None
    )
    collector = request.config.stash.get(_COLLECTOR_KEY, None)

    def _run(
        agent_fn: Callable[..., Any],
        *,
        input: Any | None = None,
        expected: Any | None = None,
        scorers: list[Scorer | str] | None = None,
        assert_pass: bool = True,
        case_name: str | None = None,
        persist: bool = True,
    ) -> EvalCaseRecord:
        # Resolve input / expected: explicit args > @case tag > @dataset row.
        in_text = input
        exp = expected
        if in_text is None and case_meta:
            in_text = case_meta.get("input")
        if exp is None and case_meta:
            exp = case_meta.get("expected")
        if in_text is None and isinstance(eval_case, dict):
            in_text = eval_case.get("input")
        if exp is None and isinstance(eval_case, dict):
            exp = eval_case.get("expected_output", eval_case.get("expected"))
        if in_text is None:
            pytest.fail(
                "evaluate_one: no input. Use @case(input=..., expected=...) "
                "or @dataset(...) on the test, or pass input= explicitly."
            )

        sig = inspect.signature(agent_fn)
        error: str | None = None
        output: Any = None
        try:
            output = agent_fn(in_text) if len(sig.parameters) >= 1 else agent_fn()
        except Exception as e:
            # Infra failure: the agent never produced an output. Recorded as
            # errored (unscored) so it can't masquerade as a quality miss in
            # the aggregated run — and the test still fails loudly below.
            error = str(e)[:500]

        if error is not None:
            record = EvalCaseRecord(
                input=in_text,
                expected_output=exp,
                actual_output=None,
                trace_id=None,
                per_scorer={},
                error=error,
            )
            if collector is not None:
                collector.cases.append(
                    _CollectedCase(
                        record=record,
                        persist=persist,
                        agent_name=infer_agent_name(agent_fn),
                    )
                )
            pytest.fail(
                f"fastaiagent eval case errored (infra, not scored): {error}\n  input: {in_text!r}"
            )

        if hasattr(output, "output"):
            output_text = output.output
        else:
            output_text = str(output)
        trace_id = getattr(output, "trace_id", None)

        scorer_objs = _resolve_scorers(scorers)
        per_scorer: dict[str, dict[str, Any]] = {}
        scorer_results: dict[str, ScorerResult] = {}
        all_passed = True
        for scorer in scorer_objs:
            result = scorer.score(input=str(in_text), output=output_text, expected=exp)
            scorer_results[scorer.name] = result
            per_scorer[scorer.name] = {
                "passed": bool(result.passed),
                "score": float(result.score),
                "reason": result.reason,
            }
            all_passed = all_passed and bool(result.passed)

        record = EvalCaseRecord(
            input=in_text,
            expected_output=exp,
            actual_output=output_text,
            trace_id=trace_id,
            per_scorer=per_scorer,
        )
        if collector is not None:
            collector.cases.append(
                _CollectedCase(
                    record=record,
                    scorer_results=scorer_results,
                    persist=persist,
                    agent_name=infer_agent_name(agent_fn),
                )
            )

        if assert_pass and not all_passed:
            pytest.fail(_format_failure(record))
        return record

    return _run
