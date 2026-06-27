"""Configuration for the eval-driven optimization loop (``fastaiagent.optimize``).

P1 moves only the ``instructions`` lever with a greedy coordinate-ascent search.
The remaining fields are forward-compatible so P2–P4 add drivers, not config
reshape (see the build spec's frozen-contract list).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastaiagent.eval.scorer import Scorer

# Levers with a driver: instructions (P1), fewshot (P2), memory (P3).
_SUPPORTED_LEVERS = frozenset({"instructions", "fewshot", "memory"})


@dataclass
class OptimizeConfig:
    """Tuning knobs for :func:`fastaiagent.optimize.aoptimize`.

    Args:
        levers: which levers the search may move. Default ``("instructions",)``
            (prompt only); add ``"fewshot"`` (few-shot examples) and/or
            ``"memory"`` (which learned facts to inject). Coordinate ascent cycles
            the active levers one per round.
        strategy: search strategy. P1: ``"greedy"`` (coordinate ascent).
        max_iterations: hard cap on optimization rounds.
        patience: stop after this many consecutive non-improving rounds.
        target_score: stop early once the dev score reaches this.
        max_eval_runs: hard cap on candidate evaluations (cost governor).
        max_judge_calls: hard cap on judge invocations (cost governor).
        splits: ``(train, dev, holdout)`` fractions, summing to 1.0.
        min_delta: dev improvement smaller than this counts as "no improvement".
        holdout_regression_tol: revert the winner if its holdout score drops more
            than this below the baseline holdout score.
        seed: seeds the deterministic train/dev/holdout shuffle.
        primary_metric: scorer name whose ``avg_score`` drives selection; falls
            back to overall pass-rate (via ``Scorecard.from_eval_results``).
        candidates_per_iteration: proposals generated per round.
        selection_judge: a ``Scorer`` (e.g. ``LLMJudge``/``GEval``) used *inside*
            the loop for accept/reject. Composed into the scorers list, not a
            ``judge=`` kwarg (``aevaluate`` has none).
        audit_judge: a ``Scorer`` used only on the holdout guard. Defaults to
            ``selection_judge`` with a warning (see :meth:`resolve_audit_judge`).
        allow_writable_memory: opt in to optimizing agents whose memory writes to
            an external store during a run (e.g. ``VectorBlock``). Off by default
            — such agents are refused, since sharing the store bleeds candidates.
    """

    levers: tuple[str, ...] = ("instructions",)
    strategy: str = "greedy"
    max_iterations: int = 8
    patience: int = 3
    target_score: float | None = None
    max_eval_runs: int | None = None
    max_judge_calls: int | None = None
    splits: tuple[float, float, float] = (0.5, 0.25, 0.25)
    min_delta: float = 0.01
    holdout_regression_tol: float = 0.0
    seed: int = 0
    primary_metric: str | None = None
    candidates_per_iteration: int = 3
    selection_judge: Scorer | None = None
    audit_judge: Scorer | None = None
    allow_writable_memory: bool = False

    def __post_init__(self) -> None:
        if not self.levers:
            raise ValueError("levers must be non-empty")
        extra = set(self.levers) - _SUPPORTED_LEVERS
        if extra:
            raise ValueError(
                f"supported levers are {sorted(_SUPPORTED_LEVERS)}; {sorted(extra)} not available."
            )
        if self.strategy != "greedy":
            raise ValueError(f"P1 supports strategy='greedy' only, got {self.strategy!r}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.candidates_per_iteration < 1:
            raise ValueError("candidates_per_iteration must be >= 1")
        if len(self.splits) != 3 or abs(sum(self.splits) - 1.0) > 1e-6:
            raise ValueError(f"splits must be three fractions summing to 1.0, got {self.splits}")
        if any(s < 0 for s in self.splits):
            raise ValueError("splits fractions must be >= 0")

    def resolve_audit_judge(self) -> Scorer | None:
        """Return the holdout-audit judge, defaulting to the selection judge.

        CONTRACT 2: when ``audit_judge`` is unset but a ``selection_judge`` is
        present, selection and audit share an identity — a reference-free agent
        would be auditing itself — so we warn. Wiring a distinct audit judge later
        needs no config reshape.
        """
        if self.audit_judge is not None:
            return self.audit_judge
        if self.selection_judge is not None:
            warnings.warn(
                "audit_judge is None; falling back to selection_judge. Selection and the "
                "holdout audit now share a judge, so a reference-free agent is optimizing "
                "against its own reported metric. Pass a distinct audit_judge (different "
                "model or judge prompt) for a trustworthy holdout guard.",
                stacklevel=2,
            )
        return self.selection_judge
