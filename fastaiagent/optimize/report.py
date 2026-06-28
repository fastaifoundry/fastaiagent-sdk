"""OptimizationReport + the baseline → steps → holdout-guarded-winner print."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastaiagent.optimize.candidate import Candidate, CandidateScore, apply_candidate

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent


@dataclass
class TrajectoryPoint:
    """One scored candidate in the search, in order (``iteration=0`` = baseline)."""

    iteration: int
    lever: str
    candidate_id: str
    dev_score: float
    accepted: bool
    rationale: str = ""
    # A lever skipped because it had nothing to propose (e.g. the memory lever
    # when there are no learned facts at the resolved scope). Distinct from a
    # reject (a candidate was scored but didn't beat the current best).
    skipped: bool = False
    # The eval_runs.run_id this candidate's dev eval produced (when persisted),
    # so the UI can drill from a trajectory row into the existing eval rows.
    eval_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "lever": self.lever,
            "candidate_id": self.candidate_id,
            "dev_score": self.dev_score,
            "accepted": self.accepted,
            "rationale": self.rationale,
            "skipped": self.skipped,
            "eval_run_id": self.eval_run_id,
        }


@dataclass
class OptimizationReport:
    """Result of an optimization run.

    Mirrors ``HardeningReport`` (``.summary()`` / ``.to_dict()``) but adds the
    score ``trajectory`` and a winner you can apply via :meth:`apply_to`.
    """

    agent_name: str
    baseline: CandidateScore
    best: CandidateScore
    best_candidate: Candidate
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    holdout_baseline: CandidateScore | None = None
    holdout_best: CandidateScore | None = None
    reverted: bool = False
    run_id: str | None = None
    # Reproducibility metadata, set at construction in the loop driver.
    seed: int = 0
    levers: tuple[str, ...] = ()
    run_name: str | None = None

    @property
    def improved(self) -> bool:
        """Did the winner beat the baseline on dev (and survive the holdout guard)?"""
        return self.best.score > self.baseline.score and not self.reverted

    def apply_to(self, agent: Agent) -> Agent:
        """Return a fresh agent with the winning candidate applied.

        Uses ``allow_writable_memory=True``: a successful optimize run already
        cleared the isolation guard, and the returned agent is the real
        deployment, so memory blocks share their handles as usual.
        """
        return apply_candidate(agent, self.best_candidate, allow_writable_memory=True)

    def summary(self) -> str:
        lines = [
            f"Optimization — {self.agent_name} (stopped: {self.stopped_reason})",
            "=" * 60,
            f"baseline   dev={self.baseline.score:.3f}",
        ]
        for p in self.trajectory:
            if p.skipped:
                lines.append(f" [{p.lever}] SKIPPED — {p.rationale}")
                continue
            if p.iteration == 0:
                continue
            delta = p.dev_score - self.baseline.score
            tag = "ACCEPT" if p.accepted else "reject"
            line = f" iter {p.iteration} [{p.lever}]  dev={p.dev_score:.3f} ({delta:+.3f})  {tag}"
            if p.accepted and p.rationale:
                line += f"  — {p.rationale[:80]}"
            lines.append(line)
        lines.append("-" * 60)
        if self.reverted:
            lines.append(
                f"best dev={self.best.score:.3f} REGRESSED on holdout → reverted to baseline"
            )
        else:
            lines.append(f"best        dev={self.best.score:.3f}")
        if self.holdout_best is not None and self.holdout_baseline is not None:
            hd = self.holdout_best.score - self.holdout_baseline.score
            verdict = "reverted" if self.reverted else "winner kept"
            lines.append(
                f"holdout     best={self.holdout_best.score:.3f} "
                f"(baseline={self.holdout_baseline.score:.3f}, Δ{hd:+.3f}) → {verdict}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "baseline": {"score": self.baseline.score, "per_metric": self.baseline.per_metric},
            "best": {"score": self.best.score, "per_metric": self.best.per_metric},
            "best_candidate": self.best_candidate.to_dict(),
            "trajectory": [p.to_dict() for p in self.trajectory],
            "accepted": self.accepted,
            "stopped_reason": self.stopped_reason,
            "reverted": self.reverted,
            "improved": self.improved,
            "holdout_baseline": self.holdout_baseline.score if self.holdout_baseline else None,
            "holdout_best": self.holdout_best.score if self.holdout_best else None,
            "run_id": self.run_id,
        }

    def persist_local(
        self,
        *,
        db_path: str | Path | None = None,
        run_name: str | None = None,
        agent_name: str | None = None,
    ) -> str:
        """Persist this optimization run to the unified local.db.

        Writes one row to ``optimize_runs`` and one per trajectory point to
        ``optimize_iterations``. Iteration rows carry ``eval_run_id`` — a link
        into the ``eval_runs`` row each candidate already produced via
        ``aevaluate(persist=)`` — so the UI drills into existing eval data
        without duplicating it. Returns the generated ``run_id`` (also stashed
        on ``self.run_id``) so callers can correlate / deep-link.

        Mirrors :meth:`fastaiagent.eval.results.EvalResults.persist_local`.
        """
        import json
        import uuid
        from datetime import datetime, timezone

        from fastaiagent._internal.config import get_config
        from fastaiagent._internal.project import safe_get_project_id
        from fastaiagent.ui.db import init_local_db

        resolved = (
            Path(db_path) if db_path is not None else Path(get_config().local_db_path)
        )
        run_id = uuid.uuid4().hex
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        pid = safe_get_project_id()
        agent = agent_name or self.agent_name

        db = init_local_db(resolved)
        try:
            db.execute(
                """INSERT INTO optimize_runs
                   (run_id, run_name, agent_name, baseline_score, best_score,
                    holdout_baseline_score, holdout_best_score, reverted,
                    stopped_reason, seed, levers, config, best_candidate,
                    baseline_eval_run_id, best_eval_run_id, iteration_count,
                    started_at, finished_at, metadata, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_name or self.run_name,
                    agent,
                    self.baseline.score,
                    self.best.score,
                    self.holdout_baseline.score if self.holdout_baseline else None,
                    self.holdout_best.score if self.holdout_best else None,
                    1 if self.reverted else 0,
                    self.stopped_reason,
                    self.seed,
                    json.dumps(list(self.levers)),
                    json.dumps({"improved": self.improved}),
                    json.dumps(self.best_candidate.to_dict(), default=str),
                    self.baseline.eval_run_id,
                    self.best.eval_run_id,
                    len(self.trajectory),
                    timestamp,
                    timestamp,
                    json.dumps({}),
                    pid,
                ),
            )
            for ordinal, p in enumerate(self.trajectory):
                db.execute(
                    """INSERT INTO optimize_iterations
                       (iteration_id, run_id, ordinal, iteration, lever,
                        candidate_id, dev_score, accepted, skipped, rationale,
                        eval_run_id, project_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex,
                        run_id,
                        ordinal,
                        p.iteration,
                        p.lever,
                        p.candidate_id,
                        p.dev_score,
                        1 if p.accepted else 0,
                        1 if p.skipped else 0,
                        p.rationale,
                        p.eval_run_id,
                        pid,
                    ),
                )
        finally:
            db.close()
        self.run_id = run_id
        return run_id
