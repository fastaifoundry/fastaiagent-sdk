"""Eval-driven self-improvement loop (P1: prompt-only).

Closes the loop ``harden`` left open: instead of *recommending* prompt fixes, it
proposes, applies, re-evaluates, and keeps the best — gated by a held-out split.

Public surface::

    from fastaiagent.optimize import optimize, OptimizeConfig

    report = optimize(agent, "cases.jsonl", scorers=["exact_match"],
                      config=OptimizeConfig(max_iterations=5))
    print(report.summary())
    better_agent = report.apply_to(agent)   # apply the winning prompt

``aoptimize`` is the async implementation; ``optimize`` is the sync wrapper. The
prompt-rewrite proposer lives here (``fastaiagent.optimize.proposers``); the
``eval`` public API is unchanged.
"""

from __future__ import annotations

from fastaiagent.optimize.candidate import (
    Candidate,
    CandidateScore,
    apply_candidate,
)
from fastaiagent.optimize.config import OptimizeConfig
from fastaiagent.optimize.loop import aoptimize, optimize
from fastaiagent.optimize.report import OptimizationReport, TrajectoryPoint

__all__ = [
    "optimize",
    "aoptimize",
    "OptimizeConfig",
    "Candidate",
    "CandidateScore",
    "OptimizationReport",
    "TrajectoryPoint",
    "apply_candidate",
]
