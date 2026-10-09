"""The eval-driven optimization loop — greedy coordinate ascent.

P1 moves the system prompt; P2 adds the few-shot lever and cycles
``instructions → fewshot`` one lever per round.

CONTRACT 1: the loop body calls ``score_candidate(...)`` only — never
``aevaluate`` directly. OSS ships cold-eval scoring; the seam stays a single
swappable interface so the Enterprise plane's replay-grounded scoring can drop
in as a second implementation with no driver changes. (``bootstrap_demos`` is a
proposal-time teacher step, not the selection seam.)
"""

from __future__ import annotations

import logging
import random
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.optimize.candidate import (
    Candidate,
    CandidateScore,
    _clone_memory_blocks,
    _FactSource,
    _resolve_memory_source,
    apply_candidate,
    is_llm_scorer,
    scorer_name,
    scorer_present,
)
from fastaiagent.optimize.config import OptimizeConfig
from fastaiagent.optimize.proposers import (
    _arequest_rewrites,
    bootstrap_demos,
    propose_fact_subsets,
)
from fastaiagent.optimize.report import OptimizationReport, TrajectoryPoint

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.eval.dataset import Dataset
    from fastaiagent.eval.scorer import Scorer
    from fastaiagent.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Demo-set sizes the few-shot lever tries as candidate variants per round.
_FEWSHOT_KS = (2, 4, 8)


def _to_items(dataset: Dataset | str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce the accepted dataset forms into a plain list of case dicts."""
    from pathlib import Path

    from fastaiagent.eval.dataset import Dataset

    if isinstance(dataset, Dataset):
        return list(dataset)
    if isinstance(dataset, list):
        return list(dataset)
    if isinstance(dataset, str):
        p = Path(dataset)
        ds = Dataset.from_csv(p) if p.suffix == ".csv" else Dataset.from_jsonl(p)
        return list(ds)
    raise TypeError(f"unsupported dataset type: {type(dataset)!r}")


def _split(
    items: list[dict[str, Any]], splits: tuple[float, float, float], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Seeded, deterministic train/dev/holdout partition.

    Guarantees at least one case in each split when ``len(items) >= 3``.
    """
    n = len(items)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_train = int(round(splits[0] * n))
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n_train)
    rest = n - n_train
    denom = splits[1] + splits[2]
    n_dev = int(round(rest * (splits[1] / denom))) if denom > 0 else 0
    n_dev = max(1, min(n_dev, rest - 1)) if rest >= 2 else rest

    train = [items[i] for i in idx[:n_train]]
    dev = [items[i] for i in idx[n_train : n_train + n_dev]]
    holdout = [items[i] for i in idx[n_train + n_dev :]]
    return train, dev, holdout


def _candidate_for(
    best: Candidate,
    *,
    system_prompt: str | None = None,
    fewshot_demos: list[dict[str, Any]] | None = None,
    fact_ids: list[int] | None = None,
    origin: str,
    rationale: str,
) -> Candidate:
    """A new candidate = current best with one lever overridden (coordinate ascent).

    Carries best's other levers so each step builds on the accepted state.
    """
    return Candidate(
        system_prompt=system_prompt if system_prompt is not None else best.system_prompt,
        fewshot_demos=fewshot_demos if fewshot_demos is not None else best.fewshot_demos,
        fact_ids=fact_ids if fact_ids is not None else best.fact_ids,
        parent_id=best.id,
        origin=origin,
        rationale=rationale,
    )


def _scorers_for(base: list[Any], judge: Scorer | None) -> list[Any]:
    """The scorers one evaluation runs: the caller's, plus ``judge`` unless an
    equivalent is already there (CONTRACT 3); ``exact_match`` when there are none."""
    run = list(base)
    if judge is not None and not scorer_present(run, judge):
        run.append(judge)
    return run or ["exact_match"]


@dataclass
class _Budget:
    """``max_eval_runs`` / ``max_judge_calls``, kept as hard caps.

    An evaluation starts only when it fits together with what the holdout guard
    still needs (``reserve_*``), so the guard always runs and the totals never pass
    a cap. They were checked only between candidates, with the train split, the
    few-shot teacher pass and the guard all outside them, and only a
    ``selection_judge`` counted as a judge.
    """

    max_runs: int | None
    max_calls: int | None
    runs: int = 0
    calls: int = 0
    reserve_runs: int = 0
    reserve_calls: int = 0

    def fits(self, runs: int, calls: int, *, reserve: bool = True) -> bool:
        need_runs = runs + (self.reserve_runs if reserve else 0)
        need_calls = calls + (self.reserve_calls if reserve else 0)
        if self.max_runs is not None and self.runs + need_runs > self.max_runs:
            return False
        if self.max_calls is not None and self.calls + need_calls > self.max_calls:
            return False
        return True

    def spend(self, runs: int, calls: int) -> None:
        self.runs += runs
        self.calls += calls


def _check_audit_judge(audit: Scorer | None, base_scorers: list[Any]) -> None:
    """The holdout guard must score with the audit judge that was asked for.

    Judges are deduped by name (CONTRACT 3), so a different scorer of the same
    name in ``scorers`` stood in for the audit judge without a word — the guard
    then audited with the selection judge it was meant to check.
    """
    if audit is None:
        return
    name = scorer_name(audit)
    for s in base_scorers:
        if s is audit:
            warnings.warn(
                "audit_judge is also in scorers, so it drives selection as well as the "
                "holdout audit. Pass a distinct audit_judge for a trustworthy guard.",
                stacklevel=3,
            )
            return
        if name is not None and scorer_name(s) == name:
            raise ValueError(
                f"audit_judge is named {name!r}, like a different scorer in scorers, so "
                "the holdout guard would score with that scorer instead of the audit "
                "judge. Give the audit judge its own name, e.g. name='audit'."
            )


async def aoptimize(
    agent: Agent,
    dataset: Dataset | str | list[dict[str, Any]],
    scorers: list[Scorer | str] | None = None,
    *,
    config: OptimizeConfig | None = None,
    proposer_llm: LLMClient | None = None,
    run_name: str | None = None,
    persist: bool = True,
) -> OptimizationReport:
    """Optimize an agent against ``dataset`` (async).

    Greedy coordinate ascent over the configured levers (``instructions`` and/or
    ``fewshot``), one lever per round: propose candidates on top of the current
    best, keep the best on dev when it beats the current best by ``min_delta``,
    stop on patience/budget/target, then a holdout guard reverts the winner if it
    regressed on selection-blind data.
    """
    from fastaiagent.agent.agent import Agent

    if not isinstance(agent, Agent):
        raise TypeError(
            f"optimize() takes an Agent, got {type(agent).__name__}. To tune a "
            "Supervisor, Swarm or Chain, optimize the Agent behind each step."
        )
    cfg = config or OptimizeConfig()
    base_scorers: list[Any] = list(scorers or [])
    selection_judge = cfg.selection_judge
    audit_judge = cfg.resolve_audit_judge()
    _check_audit_judge(cfg.audit_judge, base_scorers)

    # The instructions lever rewrites a static string prompt only.
    if "instructions" in cfg.levers and callable(agent.system_prompt):
        raise ValueError(
            "optimize: the agent's system_prompt is callable (dynamic); the "
            "instructions lever can only rewrite a static string prompt. Drop "
            "'instructions' from levers or pass a string system_prompt."
        )

    # Fail fast if memory can't be isolated per candidate (e.g. VectorBlock
    # without allow_writable_memory) — raises MemoryIsolationError up front rather
    # than mid-run. (Replaces P1's blanket writable-memory warning; memory-bearing
    # agents are now supported via block.isolated_copy().)
    _clone_memory_blocks(agent.memory, allow_writable_memory=cfg.allow_writable_memory)

    items = _to_items(dataset)
    if len(items) < 3:
        raise ValueError(
            f"optimize needs at least 3 cases to form train/dev/holdout splits; got {len(items)}."
        )
    if len(items) < 15:
        warnings.warn(
            f"optimize: only {len(items)} cases — splits will be small and scores noisy. "
            "~15+ cases recommended (or just run harden() once).",
            stacklevel=2,
        )
    train, dev, holdout = _split(items, cfg.splits, cfg.seed)
    if not dev or not holdout:
        raise ValueError(
            f"splits {cfg.splits} on {len(items)} cases left dev or holdout empty; "
            "use more cases or different split fractions."
        )
    splits_by_name = {"train": train, "dev": dev, "holdout": holdout}

    from fastaiagent.llm import LLMClient

    proposer = proposer_llm or LLMClient()

    # Memory lever: resolve where the agent's facts live once — its own store,
    # scope and project (used by the lever + the skip check).
    mem_source = _resolve_memory_source(agent) if "memory" in cfg.levers else _FactSource("", "")

    def judge_cost(n_cases: int, judge: Scorer | None) -> int:
        """Judge calls one evaluation makes: one per case per model-backed scorer."""
        return n_cases * sum(1 for s in _scorers_for(base_scorers, judge) if is_llm_scorer(s))

    def split_cost(split: str, judge: Scorer | None) -> int:
        return judge_cost(len(splits_by_name[split]), judge)

    # The holdout guard scores the baseline and, once one is accepted, the winner.
    # The loop reserves both, so the guard always runs inside the caps.
    budget = _Budget(cfg.max_eval_runs, cfg.max_judge_calls)
    holdout_calls = split_cost("holdout", audit_judge)
    floor_calls = split_cost("dev", selection_judge) + holdout_calls
    if not budget.fits(2, floor_calls, reserve=False):
        raise ValueError(
            f"max_eval_runs={cfg.max_eval_runs}, max_judge_calls={cfg.max_judge_calls} "
            "can't cover the baseline and the holdout guard, which need 2 evaluations "
            f"and {floor_calls} judge calls on this dataset. Raise the caps."
        )
    budget.reserve_runs, budget.reserve_calls = 2, 2 * holdout_calls

    async def score_candidate(
        candidate: Candidate, split: str, *, judge: Scorer | None
    ) -> CandidateScore:
        # CONTRACT 1: the loop calls only this; CONTRACT 3: judge composed + deduped.
        from fastaiagent.eval.dataset import Dataset
        from fastaiagent.eval.evaluate import aevaluate

        split_items = splits_by_name[split]
        cand_agent = apply_candidate(
            agent, candidate, allow_writable_memory=cfg.allow_writable_memory
        )
        results = await aevaluate(
            cand_agent.arun,
            Dataset.from_list(split_items),
            _scorers_for(base_scorers, judge),
            persist=persist,
            run_name=f"{run_name or 'optimize'}:{candidate.id[:8]}:{split}",
            agent_name=agent.name,
        )
        budget.spend(1, split_cost(split, judge))
        return CandidateScore.from_eval(
            candidate.id, split, results, primary_metric=cfg.primary_metric
        )

    def before_teacher_eval(n_cases: int) -> bool:
        """The few-shot teacher pass is an evaluation too: it runs inside the caps."""
        calls = judge_cost(n_cases, selection_judge)
        if not budget.fits(1, calls):
            return False
        budget.spend(1, calls)
        return True

    # No few-shot demo may carry an input the candidates are scored on.
    scored_inputs = [it.get("input") for it in dev + holdout]

    async def _propose(
        lever: str, best: Candidate, best_train: CandidateScore | None
    ) -> tuple[list[Candidate], str | None]:
        """Candidate variants for the active lever, on top of ``best``, and the
        proposer's error when it could not run."""
        if lever == "instructions":
            effective = (
                best.system_prompt if best.system_prompt is not None else agent.system_prompt
            )
            # Callable prompts are refused up front when the instructions lever is
            # active, so ``effective`` is always a concrete string here.
            assert isinstance(effective, str) and best_train is not None
            rewrites, error = await _arequest_rewrites(
                current_prompt=effective,
                results=best_train.results,
                llm=proposer,
                n=cfg.candidates_per_iteration,
                agent_name=agent.name,
            )
            return [
                _candidate_for(best, system_prompt=p, origin="prompt:rewrite", rationale=r)
                for p, r in rewrites
            ], error
        if lever == "fewshot":
            ks = sorted({k for k in _FEWSHOT_KS})[: cfg.candidates_per_iteration]
            best_agent = apply_candidate(
                agent, best, allow_writable_memory=cfg.allow_writable_memory
            )
            pool = await bootstrap_demos(
                agent=best_agent,
                train_items=train,
                scorers=base_scorers,
                judge=selection_judge,
                k=max(ks),
                exclude_inputs=scored_inputs,
                before_teacher_eval=before_teacher_eval,
            )
            cands: list[Candidate] = []
            seen_sizes: set[int] = set()
            for k in ks:
                kk = min(k, len(pool))
                if kk == 0 or kk in seen_sizes:
                    continue
                seen_sizes.add(kk)
                cands.append(
                    _candidate_for(
                        best,
                        fewshot_demos=pool[:kk],
                        origin="fewshot:bootstrap",
                        rationale=f"few-shot k={kk}",
                    )
                )
            return cands, None
        if lever == "memory":
            subsets = propose_fact_subsets(
                scope=mem_source.scope,
                scope_id=mem_source.scope_id,
                n=cfg.candidates_per_iteration,
                store=mem_source.store,
                project_id=mem_source.project_id,
            )
            return [
                _candidate_for(
                    best, fact_ids=ids, origin="memory:subset", rationale=f"facts k={len(ids)}"
                )
                for ids in subsets
            ], None
        return [], None

    # ── Baseline ──────────────────────────────────────────────────────────────
    # NOTE (deviation from spec §4): baseline-on-dev uses the SELECTION judge so the
    # dev deltas driving accept/reject are computed against a single judge; the audit
    # judge is used for the holdout guard. Identical when selection == audit (default).
    base_candidate = Candidate(origin="baseline")
    baseline_dev = await score_candidate(base_candidate, "dev", judge=selection_judge)
    best = base_candidate
    best_dev = baseline_dev
    trajectory = [
        TrajectoryPoint(
            0,
            "baseline",
            base_candidate.id,
            baseline_dev.score,
            True,
            "baseline",
            eval_run_id=baseline_dev.eval_run_id,
            errored=baseline_dev.errored,
        )
    ]
    accepted: list[str] = []

    # Train results for the current best feed the instructions proposer. Scored
    # when that lever first needs them and again whenever best changes — never for
    # a run that doesn't move the instructions lever.
    best_train: CandidateScore | None = None
    train_scored_for: str | None = None

    active_levers = list(cfg.levers)

    # Memory lever needs facts at the resolved scope; with none, skip it (don't
    # error, don't burn patience) and record the skip distinctly from a reject.
    if "memory" in active_levers:
        where = f"scope={mem_source.scope}:{mem_source.scope_id}"
        if mem_source.project_id:
            where += f" project={mem_source.project_id}"
        if not mem_source.resolved_store().list_active(
            mem_source.scope, mem_source.scope_id, mem_source.project_id
        ):
            active_levers = [lv for lv in active_levers if lv != "memory"]
            # `fastaiagent learn` writes to local.db, so it only helps an agent
            # whose facts live there.
            hint = (
                "add facts where the agent's memory reads them "
                "(`fastaiagent learn` writes to local.db)"
            )
            trajectory.append(
                TrajectoryPoint(
                    0,
                    "memory",
                    "",
                    baseline_dev.score,
                    accepted=False,
                    rationale=f"no learned facts at {where} — {hint}",
                    skipped=True,
                )
            )
            logger.info("optimize: memory lever skipped — no learned facts at %s", where)

    no_improve = 0
    stopped_reason = ""
    iteration = 0
    # Consecutive rounds whose proposer could not run, and every such error.
    failed_streak = 0
    proposer_errors: list[str] = []

    # A baseline already at the target has nothing to climb to.
    if cfg.target_score is not None and baseline_dev.score >= cfg.target_score:
        stopped_reason = "target_score"

    while not stopped_reason and active_levers and iteration < cfg.max_iterations:
        iteration += 1
        lever = active_levers[(iteration - 1) % len(active_levers)]

        # Start a round only when its first candidate (and, for the instructions
        # lever, a train re-score) fits the budget.
        need_train = lever == "instructions" and best.id != train_scored_for
        round_calls = split_cost("dev", selection_judge)
        if need_train:
            round_calls += split_cost("train", selection_judge)
        if not budget.fits(2 if need_train else 1, round_calls):
            stopped_reason = "budget"
            break
        if need_train:
            best_train = await score_candidate(best, "train", judge=selection_judge)
            train_scored_for = best.id

        candidates, error = await _propose(lever, best, best_train)
        if error:
            # A proposer that could not run is not "no improvement": say so.
            failed_streak += 1
            proposer_errors.append(error)
            logger.warning("optimize: the prompt proposer failed: %s", error)
            trajectory.append(
                TrajectoryPoint(
                    iteration,
                    lever,
                    "",
                    best_dev.score,
                    accepted=False,
                    rationale=f"proposer failed: {error}",
                    skipped=True,
                )
            )
        else:
            failed_streak = 0
        if not candidates:
            no_improve += 1
            if no_improve >= cfg.patience:
                stopped_reason = "patience"
                break
            continue

        scored: list[tuple[Candidate, CandidateScore]] = []
        for cand in candidates:
            if not budget.fits(1, split_cost("dev", selection_judge)):
                break
            cs = await score_candidate(cand, "dev", judge=selection_judge)
            scored.append((cand, cs))
            trajectory.append(
                TrajectoryPoint(
                    iteration,
                    lever,
                    cand.id,
                    cs.score,
                    False,
                    cand.rationale,
                    eval_run_id=cs.eval_run_id,
                    errored=cs.errored,
                )
            )

        if not scored:  # the few-shot teacher pass spent what the round had
            stopped_reason = "budget"
            break

        local_cand, local_cs = max(scored, key=lambda t: t[1].score)
        if local_cs.score - best_dev.score >= cfg.min_delta:
            best = local_cand
            best_dev = local_cs
            accepted.append(best.id)
            for tp in trajectory:
                if tp.candidate_id == local_cand.id:
                    tp.accepted = True
            no_improve = 0
        else:
            no_improve += 1

        if cfg.target_score is not None and best_dev.score >= cfg.target_score:
            stopped_reason = "target_score"
            break
        if no_improve >= cfg.patience:
            stopped_reason = "patience"
            break

    if not stopped_reason:
        stopped_reason = "max_iterations" if active_levers else "no_active_levers"
    # Every round since the last improvement failed to propose: the proposer, not
    # the search, is why the run ended.
    if stopped_reason in ("patience", "max_iterations") and 0 < failed_streak == no_improve:
        stopped_reason = "proposer_failed"

    # ── Holdout regression guard (selection-blind, audit judge) ─────────────────
    holdout_baseline = await score_candidate(base_candidate, "holdout", judge=audit_judge)
    reverted = False
    if best.id != base_candidate.id:
        holdout_best = await score_candidate(best, "holdout", judge=audit_judge)
        if holdout_best.score < holdout_baseline.score - cfg.holdout_regression_tol:
            reverted = True
            best = base_candidate
            best_dev = baseline_dev
    else:
        holdout_best = holdout_baseline

    baseline_prompt = agent.system_prompt if isinstance(agent.system_prompt, str) else None
    report = OptimizationReport(
        agent_name=agent.name,
        baseline=baseline_dev,
        best=best_dev,
        best_candidate=best,
        trajectory=trajectory,
        accepted=accepted,
        stopped_reason=stopped_reason + ("+reverted" if reverted else ""),
        holdout_baseline=holdout_baseline,
        holdout_best=holdout_best,
        reverted=reverted,
        seed=cfg.seed,
        levers=tuple(cfg.levers),
        run_name=run_name,
        proposer_errors=proposer_errors,
        baseline_system_prompt=baseline_prompt,
    )

    # Persist the run record (gated by the same flag that gates per-candidate
    # eval persistence). Each iteration links to the eval_runs row its candidate
    # already produced — no duplicate eval storage.
    if persist:
        try:
            report.persist_local(run_name=run_name, agent_name=agent.name)
        except Exception as exc:  # persistence is best-effort, never fail a run
            logger.warning("optimize: failed to persist run record: %s", exc)

    return report


def optimize(
    agent: Agent,
    dataset: Dataset | str | list[dict[str, Any]],
    scorers: list[Scorer | str] | None = None,
    *,
    config: OptimizeConfig | None = None,
    proposer_llm: LLMClient | None = None,
    run_name: str | None = None,
    persist: bool = True,
) -> OptimizationReport:
    """Synchronous wrapper around :func:`aoptimize` (for CLI / notebooks)."""
    return run_sync(
        aoptimize(
            agent,
            dataset,
            scorers,
            config=config,
            proposer_llm=proposer_llm,
            run_name=run_name,
            persist=persist,
        )
    )
