"""OptimizationReport + the baseline → steps → holdout-guarded-winner print."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "lever": self.lever,
            "candidate_id": self.candidate_id,
            "dev_score": self.dev_score,
            "accepted": self.accepted,
            "rationale": self.rationale,
            "skipped": self.skipped,
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
