"""Evaluation results with summary, export, and local persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastaiagent.eval.scorer import ScorerResult

logger = logging.getLogger(__name__)


def _git_provenance() -> dict[str, str]:
    """Best-effort ``{git_sha, git_branch}`` for baseline selection in CI.

    Prefers the GitHub Actions env (present even on detached-HEAD checkouts),
    falls back to ``git rev-parse``; returns {} outside a repo.
    """
    import os
    import subprocess

    out: dict[str, str] = {}
    sha = os.environ.get("GITHUB_SHA")
    branch = os.environ.get("GITHUB_REF_NAME")
    if not sha:
        try:
            sha = (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                ).stdout.strip()
                or None
            )
        except (OSError, subprocess.SubprocessError):
            sha = None
    if not branch:
        try:
            branch = (
                subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                ).stdout.strip()
                or None
            )
        except (OSError, subprocess.SubprocessError):
            branch = None
    if sha:
        out["git_sha"] = sha
    if branch and branch != "HEAD":
        out["git_branch"] = branch
    return out


def record_gate_result(
    run_id: str,
    *,
    gate_outcome: str,
    thresholds: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Attach a gate verdict to an already-persisted run, then offer it for export.

    ``persist_local()`` necessarily runs *before* the gate — the gate scores the
    persisted results — so the verdict can't be written in the same INSERT. This
    merges it into the run's ``metadata`` JSON (no new columns) and is the point at
    which the run becomes exportable: the plane requires ``gate_outcome``, so a run
    without one is not yet evidence.

    Best-effort by design: a failure here must never change a gate verdict the user
    already saw, so everything is swallowed to debug.

    Args:
        run_id: the id returned by :meth:`EvalResults.persist_local`.
        gate_outcome: ``passed`` | ``failed`` | ``invalid``.
        thresholds: what the gate demanded, e.g. ``{"overall.pass_rate": 0.9}``.
            Empty/None means no gate was demanded (the run still counts as evidence
            that evals ran — see ``docs/evaluation/agent-ci.md``).
        baseline: ``{run_id, pass_rate_delta, regressed_count}`` when a baseline
            comparison ran. This is what evidences "track regressions across
            versions" rather than merely "evals were run".

    Raises:
        ValueError: ``gate_outcome`` is not one of ``passed`` / ``failed`` /
            ``invalid`` (1.89.0). This is the one thing not swallowed: the
            plane rejects any other value with a 4xx, which is never retried,
            so a typo would hold back every run batched with it.
    """
    from fastaiagent.eval.gate import VALID_GATE_OUTCOMES

    if gate_outcome not in VALID_GATE_OUTCOMES:
        raise ValueError(
            f"gate_outcome must be one of {VALID_GATE_OUTCOMES}, got {gate_outcome!r}"
        )
    try:
        from fastaiagent._internal.config import get_config
        from fastaiagent.ui.db import init_local_db

        resolved = Path(db_path) if db_path is not None else Path(get_config().local_db_path)
        db = init_local_db(resolved)
        try:
            row = db.fetchone("SELECT metadata FROM eval_runs WHERE run_id = ?", (run_id,))
            if row is None:
                return
            try:
                meta = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            meta["gate_outcome"] = gate_outcome
            if thresholds is not None:
                meta["thresholds"] = thresholds
            if baseline is not None:
                meta["baseline"] = baseline
            db.execute(
                "UPDATE eval_runs SET metadata = ? WHERE run_id = ?",
                (json.dumps(meta, default=str), run_id),
            )
        finally:
            db.close()
    except Exception:  # pragma: no cover — never disturb a verdict the user saw
        logger.debug("Failed to record gate result for run %s", run_id, exc_info=True)
        return

    # Only now is the run complete enough to be evidence. No-op when disconnected
    # or when eval export is off.
    try:
        from fastaiagent.eval.platform_export import record_eval_run_for_export

        record_eval_run_for_export(run_id, db_path=db_path)
    except Exception:  # pragma: no cover
        logger.debug("Eval export kick failed for run %s", run_id, exc_info=True)


@dataclass
class EvalCaseRecord:
    """Per-case capture used by :meth:`EvalResults.persist_local`."""

    input: Any = None
    expected_output: Any = None
    actual_output: Any = None
    trace_id: str | None = None
    per_scorer: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Set when the run itself infrastructure-failed during scoring (provider 500,
    # timeout, …). Such a case is NOT scored — it carries no per-scorer verdict and
    # is excluded from failure analysis — so an infra failure can't masquerade as
    # an agent-quality miss the optimizer would try to "fix".
    error: str | None = None


def _scrubbed_reason(reason: str | None) -> str | None:
    from fastaiagent.eval.platform_export import exportable_reason

    return exportable_reason(reason)


def _without_credentials(text: str | None) -> str | None:
    """``text`` scrubbed of credentials before it reaches ``local.db`` (1.89.0).

    Not the payload switch: local capture keeps the reason in full. But a judge
    or agent exception that is not a provider error (those are scrubbed where
    they are raised) can quote a header or a URL with a key in it, and the
    Local UI shows ``local.db`` as it is.
    """
    if not text:
        return text
    from fastaiagent._internal.scrub import known_credential_values, scrub_secrets

    return scrub_secrets(text, known_credential_values())


def _stored_per_scorer(per_scorer: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, entry in per_scorer.items():
        if isinstance(entry, dict) and isinstance(entry.get("reason"), str):
            entry = {**entry, "reason": _without_credentials(entry["reason"])}
        out[name] = entry
    return out


#: ``eval_runs.summary_version`` written by this SDK. ``NULL`` marks a run written
#: before 1.89.0, whose ``pass_*`` columns counted errored and skipped verdicts.
SUMMARY_VERSION = 2


def _counts(results: list[ScorerResult]) -> dict[str, int]:
    out = {"passed": 0, "failed": 0, "errored": 0, "skipped": 0}
    for r in results:
        out[r.outcome] += 1
    return out


def _case_value_json(value: Any, media: list[dict[str, Any]]) -> str:
    """A case's input/expected/actual as JSON, with images and PDFs as descriptors.

    ``json.dumps(default=str)`` turned an ``Image`` into its ``repr``, raw bytes
    included, into ``local.db`` (1.89.0). Each image or PDF becomes
    ``{"type", "media_type", "size_bytes", "sha256", "media_idx", ...}``; the
    bytes themselves are appended to ``media`` for an ``eval_case_media``
    thumbnail row.
    """
    from fastaiagent.multimodal.image import Image
    from fastaiagent.multimodal.pdf import PDF

    def walk(v: Any) -> Any:
        if isinstance(v, Image):
            idx = len(media)
            media.append({"kind": "image", "data": v.data, "media_type": v.media_type})
            return {
                "type": "image",
                "media_type": v.media_type,
                "size_bytes": len(v.data),
                "sha256": hashlib.sha256(v.data).hexdigest(),
                "source_url": v.source_url,
                "detail": v.detail,
                "media_idx": idx,
            }
        if isinstance(v, PDF):
            idx = len(media)
            media.append({"kind": "pdf", "data": v.data, "media_type": "application/pdf"})
            return {
                "type": "pdf",
                "media_type": "application/pdf",
                "size_bytes": len(v.data),
                "sha256": hashlib.sha256(v.data).hexdigest(),
                "source_path": v.source_path,
                "source_url": v.source_url,
                "media_idx": idx,
            }
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (bytes, bytearray)):
            return {"type": "bytes", "size_bytes": len(v)}
        return v

    return json.dumps(walk(value), default=str)


def _thumbnail(item: dict[str, Any]) -> tuple[bytes | None, dict[str, Any]]:
    try:
        from fastaiagent.trace.attachments import _build_pdf_thumbnail, _build_thumbnail

        if item["kind"] == "pdf":
            return _build_pdf_thumbnail(item["data"])
        return _build_thumbnail(item["data"], item["media_type"])
    except Exception:
        return None, {}


class EvalResults:
    """Results of an evaluation run."""

    def __init__(self, scores: dict[str, list[ScorerResult]] | None = None):
        self.scores: dict[str, list[ScorerResult]] = scores or {}
        self.cases: list[EvalCaseRecord] = []
        # Populated by ``persist_local()`` (or ``evaluate()`` when persist=True)
        # so callers can deep-link into the Local UI at /evals/<run_id>.
        self.run_id: str | None = None
        # 1.89.0: what ``evaluate()`` knows about the run, for the run record.
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.kind: str | None = None
        self.agent_version: str | None = None
        self.scorer_config: dict[str, dict[str, Any]] = {}

    @property
    def case_counts(self) -> dict[str, Any]:
        """Cases passed / failed / errored, and the case pass rate (1.89.0).

        See :func:`fastaiagent.eval.compare.summarize_cases`. A case is errored
        when the agent failed, a scorer could not run, or no scorer applied.
        """
        from fastaiagent.eval.compare import summarize_cases

        return summarize_cases(
            [{"error": c.error, "per_scorer": c.per_scorer} for c in self.cases]
        )

    @property
    def case_pass_rate(self) -> float | None:
        """Passed over passed + failed cases; ``None`` when no case reached a verdict."""
        rate: float | None = self.case_counts["case_pass_rate"]
        return rate

    @property
    def check_counts(self) -> dict[str, int]:
        """Every verdict across scorers: passed / failed / errored / skipped."""
        return _counts([r for rlist in self.scores.values() for r in rlist])

    def add(self, scorer_name: str, result: ScorerResult) -> None:
        self.scores.setdefault(scorer_name, []).append(result)

    def add_case(self, record: EvalCaseRecord) -> None:
        """Record one dataset case end-to-end for later persistence."""
        self.cases.append(record)

    @property
    def errored_count(self) -> int:
        """Cases that infrastructure-failed during the run (never scored)."""
        return sum(1 for c in self.cases if c.error)

    @property
    def error_rate(self) -> float:
        """errored_count over ALL recorded cases (scored + errored).

        0.0 when no cases were recorded at all — callers gating CI should
        also check that scored cases exist (see ``eval/gate.py``), since a
        run where everything errored has an error_rate of 1.0 but a run
        with zero cases is a configuration problem, not a quality signal.
        """
        total = len(self.cases)
        return (self.errored_count / total) if total else 0.0

    def summary(self) -> str:
        """Generate a summary table.

        Rates are over applicable verdicts only (1.89.0): a check that errored
        or did not apply is counted beside the rate, never inside it.
        """
        lines = ["Evaluation Results", "=" * 50]
        for name, results in self.scores.items():
            if not results:
                continue
            c = _counts(results)
            rated = [r for r in results if r.outcome in ("passed", "failed")]
            extra = "".join(f", {c[k]} {k}" for k in ("errored", "skipped") if c[k])
            if rated:
                avg_score = sum(r.score for r in rated) / len(rated)
                pass_rate = c["passed"] / len(rated)
                lines.append(
                    f"{name}: avg={avg_score:.2f} pass_rate={pass_rate:.0%} "
                    f"({len(rated)} scored{extra})"
                )
            else:
                lines.append(f"{name}: no applicable verdict ({extra.lstrip(', ') or 'none'})")
        if self.cases:
            cc = self.case_counts
            rate = cc["case_pass_rate"]
            lines.append(
                f"cases: {cc['case_pass_count']} passed, {cc['case_fail_count']} failed, "
                f"{cc['case_error_count']} errored"
                + (f" (pass_rate={rate:.0%})" if rate is not None else "")
            )
        if self.errored_count:
            lines.append(
                f"errored: {self.errored_count}/{len(self.cases)} cases "
                f"infrastructure-failed and were NOT scored"
            )
        return "\n".join(lines)

    def export(self, path: str | Path, format: str = "json") -> None:
        """Export results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: [r.model_dump() for r in results] for name, results in self.scores.items()}
        path.write_text(json.dumps(data, indent=2))

    def publish(self, run_name: str | None = None) -> None:
        """Publish eval results to platform."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError("Not connected to platform. Call fa.connect() first.")
        api = get_platform_api()
        # The legacy endpoint counts every entry as a case. Pinned to the three
        # keys it has always received (1.89.0); a skipped verdict is not sent.
        data = {
            name: [
                {"score": r.score, "passed": r.passed, "reason": _scrubbed_reason(r.reason)}
                for r in results
                if not r.skipped
            ]
            for name, results in self.scores.items()
        }
        api.post(
            "/public/v1/eval/runs",
            {"run_name": run_name, "scores": data},
        )

    def persist_local(
        self,
        *,
        db_path: str | Path | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        agent_name: str | None = None,
        agent_version: str | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> str:
        """Persist this run to the unified local.db.

        Writes one row to ``eval_runs`` and one per case to ``eval_cases``.
        Returns the ``run_id`` so callers can correlate.

        ``metadata`` lands in the run row's JSON column; git provenance
        (``git_sha`` / ``git_branch``) is merged in best-effort so a CI
        baseline can later be selected by ref — caller-provided keys win.

        1.89.0:

        * ``pass_count``/``fail_count``/``pass_rate`` count **applicable**
          verdicts (the checks rate ``overall.pass_rate`` gates); an errored or
          skipped verdict is counted in ``check_error_count``/``check_skip_count``
          instead. ``pass_rate`` is NULL when no verdict applied (it was 0.0).
        * ``case_*`` columns hold the case outcomes (a case is one dataset row,
          not one verdict); ``summary_version`` is 2.
        * ``started_at``/``finished_at`` are the run's own times (both were the
          save time), and ``kind`` says what produced it (``evaluate``, ``cli``,
          ``pytest``, ``optimize``, ``ui``, ``ui-echo``).
        * An image or PDF in a case is stored as a descriptor, never its bytes,
          with a thumbnail in ``eval_case_media``.
        """
        from fastaiagent._internal.config import get_config
        from fastaiagent.ui.db import init_local_db

        resolved = Path(db_path) if db_path is not None else Path(get_config().local_db_path)
        run_id = run_id or uuid.uuid4().hex
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        started_at = started_at or self.started_at or timestamp
        finished_at = finished_at or self.finished_at or timestamp
        kind = kind or self.kind
        agent_version = agent_version or self.agent_version

        checks = self.check_counts
        pass_count, fail_count = checks["passed"], checks["failed"]
        rated = pass_count + fail_count
        pass_rate = round(pass_count / rated, 4) if rated else None
        cc = self.case_counts

        from fastaiagent._internal.project import safe_get_project_id

        pid = safe_get_project_id()
        merged_metadata = {**_git_provenance(), **(metadata or {})}
        if self.scorer_config:
            merged_metadata.setdefault("scorer_config", self.scorer_config)
        # Part D: only queue for platform export when export is actually on. A
        # disabled/disconnected install writes synced=1 so it never accumulates an
        # outbox that can't drain; enabling later ships only NEW runs, no backlog.
        from fastaiagent.eval.platform_export import eval_export_enabled

        synced = 0 if eval_export_enabled() else 1
        db = init_local_db(resolved)
        try:
            db.execute(
                """INSERT INTO eval_runs
                   (run_id, run_name, dataset_name, agent_name, agent_version,
                    scorers, started_at, finished_at, pass_count, fail_count,
                    pass_rate, metadata, project_id, synced,
                    kind, summary_version, case_count, case_pass_count,
                    case_fail_count, case_error_count, case_pass_rate,
                    check_error_count, check_skip_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_name,
                    dataset_name,
                    agent_name,
                    agent_version,
                    json.dumps(sorted(self.scores.keys())),
                    started_at,
                    finished_at,
                    pass_count,
                    fail_count,
                    pass_rate,
                    json.dumps(merged_metadata, default=str),
                    pid,
                    synced,
                    kind,
                    SUMMARY_VERSION,
                    cc["case_count"],
                    cc["case_pass_count"],
                    cc["case_fail_count"],
                    cc["case_error_count"],
                    cc["case_pass_rate"],
                    checks["errored"],
                    checks["skipped"],
                ),
            )
            for ordinal, case in enumerate(self.cases):
                case_id = uuid.uuid4().hex
                media: list[dict[str, Any]] = []
                db.execute(
                    """INSERT INTO eval_cases
                       (case_id, run_id, ordinal, input, expected_output,
                        actual_output, trace_id, per_scorer, error, project_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        case_id,
                        run_id,
                        ordinal,
                        _case_value_json(case.input, media),
                        _case_value_json(case.expected_output, media),
                        _case_value_json(case.actual_output, media),
                        case.trace_id,
                        json.dumps(_stored_per_scorer(case.per_scorer), default=str),
                        _without_credentials(case.error),
                        pid,
                    ),
                )
                for idx, item in enumerate(media):
                    thumb, meta = _thumbnail(item)
                    db.execute(
                        """INSERT OR REPLACE INTO eval_case_media
                           (case_id, idx, media_type, size_bytes, sha256, thumbnail,
                            meta, project_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            case_id,
                            idx,
                            item["media_type"],
                            len(item["data"]),
                            hashlib.sha256(item["data"]).hexdigest(),
                            thumb,
                            json.dumps(meta, default=str),
                            pid,
                        ),
                    )
        finally:
            db.close()
        # Stash on self so callers who hold a reference can deep-link into
        # the UI without threading the return value through their code.
        self.run_id = run_id
        return run_id

    def compare(self, other: EvalResults) -> str:
        """Compare with another set of results."""
        lines = ["Comparison", "=" * 50]
        all_scorers = set(self.scores.keys()) | set(other.scores.keys())
        for name in sorted(all_scorers):
            # Verdicts only (1.89.0): an errored or skipped result's 0.0 is a
            # placeholder, not a score.
            a = [r for r in self.scores.get(name, []) if r.outcome in ("passed", "failed")]
            b = [r for r in other.scores.get(name, []) if r.outcome in ("passed", "failed")]
            avg_a = sum(r.score for r in a) / max(len(a), 1)
            avg_b = sum(r.score for r in b) / max(len(b), 1)
            diff = avg_b - avg_a
            sign = "+" if diff > 0 else ""
            lines.append(f"{name}: {avg_a:.2f} → {avg_b:.2f} ({sign}{diff:.2f})")
        return "\n".join(lines)


@dataclass
class MetricSummary:
    """One metric's roll-up in a :class:`Scorecard`."""

    name: str
    avg_score: float | None
    pass_rate: float | None
    n: int
    """Applicable verdicts (passed + failed). Errored and skipped are counted
    apart (1.89.0); ``n == 0`` means the metric could not be evaluated, and its
    rates are ``None``."""
    errored: int = 0
    skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "avg_score": self.avg_score,
            "pass_rate": self.pass_rate,
            "n": self.n,
            "errored": self.errored,
            "skipped": self.skipped,
        }


@dataclass
class Scorecard:
    """A compact, named roll-up of an eval or simulation run.

    Per-metric average score + pass-rate, plus an overall pass-rate. Build it
    from an :class:`EvalResults` (per-scorer) or a ``SimulationResults``
    (per-scenario). Aggregation only — no LLM calls.

    Example::

        results = evaluate(agent.run, dataset, scorers=["task_completion", "faithfulness"])
        print(Scorecard.from_eval_results(results).summary())
    """

    metrics: list[MetricSummary] = field(default_factory=list)
    overall_pass_rate: float | None = 0.0
    """The checks pass rate: passed over passed + failed verdicts, all scorers.
    ``None`` when no verdict applied (1.89.0)."""
    label: str | None = None
    # Cases that infrastructure-failed and were never scored. Kept separate
    # from pass/fail so an outage can't read as either quality signal —
    # but always VISIBLE, so a run of mostly-errored cases can't look green.
    errored: int = 0
    cases: dict[str, Any] = field(default_factory=dict)
    """Case outcomes (1.89.0): ``passed``/``failed``/``errored``/``count`` and
    ``pass_rate`` (the ``cases.pass_rate`` gate metric). Empty for a run built
    without case records."""
    checks_errored: int = 0
    checks_skipped: int = 0

    @classmethod
    def from_eval_results(cls, results: EvalResults, *, label: str | None = None) -> Scorecard:
        metrics: list[MetricSummary] = []
        total = 0
        passed = 0
        for name, rlist in results.scores.items():
            if not rlist:
                continue
            c = _counts(rlist)
            rated = [r for r in rlist if r.outcome in ("passed", "failed")]
            n = len(rated)
            avg = round(sum(r.score for r in rated) / n, 4) if n else None
            pr = round(c["passed"] / n, 4) if n else None
            metrics.append(
                MetricSummary(
                    name=name,
                    avg_score=avg,
                    pass_rate=pr,
                    n=n,
                    errored=c["errored"],
                    skipped=c["skipped"],
                )
            )
            total += n
            passed += c["passed"]
        checks = results.check_counts
        cases: dict[str, Any] = {}
        if results.cases:
            cc = results.case_counts
            cases = {
                "count": cc["case_count"],
                "passed": cc["case_pass_count"],
                "failed": cc["case_fail_count"],
                "errored": cc["case_error_count"],
                "pass_rate": cc["case_pass_rate"],
            }
        return cls(
            metrics=metrics,
            overall_pass_rate=round(passed / total, 4) if total else None,
            label=label,
            errored=results.errored_count,
            cases=cases,
            checks_errored=checks["errored"],
            checks_skipped=checks["skipped"],
        )

    @classmethod
    def from_simulation(cls, results: Any, *, label: str | None = None) -> Scorecard:
        """Build from a ``SimulationResults`` (duck-typed: ``.results`` of pass/fail).

        D15 (1.89.0): the denominator stays every scenario, so an errored one
        (did not finish, or could not be judged) counts as not passed, as
        before; it is now also reported in ``errored``.
        """
        sims = getattr(results, "results", []) or []
        n = len(sims)
        passed = sum(1 for r in sims if getattr(r, "passed", False))
        errored = sum(1 for r in sims if getattr(r, "error", None))
        pr = round(passed / n, 4) if n else None
        metrics = [
            MetricSummary(name="scenarios", avg_score=pr, pass_rate=pr, n=n, errored=errored)
        ]
        return cls(
            metrics=metrics,
            overall_pass_rate=pr,
            label=label or getattr(results, "agent_name", None),
            errored=errored,
        )

    def summary(self) -> str:
        head = f"Scorecard — {self.label}" if self.label else "Scorecard"
        lines = [head, "=" * 50]
        for m in self.metrics:
            extra = "".join(
                f", {k}={v}" for k, v in (("errored", m.errored), ("n/a", m.skipped)) if v
            )
            if m.n and m.avg_score is not None and m.pass_rate is not None:
                lines.append(
                    f"{m.name:<22} avg={m.avg_score:.2f}  pass_rate={m.pass_rate:.0%}  "
                    f"(n={m.n}{extra})"
                )
            else:
                lines.append(f"{m.name:<22} not evaluable  (n=0{extra})")
        lines.append("-" * 50)
        overall = (
            f"{self.overall_pass_rate:.0%}" if self.overall_pass_rate is not None else "n/a"
        )
        lines.append(f"checks pass_rate={overall}  (overall.pass_rate)")
        if self.cases:
            rate = self.cases.get("pass_rate")
            lines.append(
                f"cases: {self.cases.get('passed', 0)} passed, {self.cases.get('failed', 0)} "
                f"failed, {self.cases.get('errored', 0)} errored  (cases.pass_rate="
                + (f"{rate:.0%})" if rate is not None else "n/a)")
            )
        if self.errored:
            lines.append(f"errored (unscored)={self.errored}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "overall_pass_rate": self.overall_pass_rate,
            "errored": self.errored,
            "cases": self.cases,
            "checks_errored": self.checks_errored,
            "checks_skipped": self.checks_skipped,
            "metrics": [m.to_dict() for m in self.metrics],
        }
