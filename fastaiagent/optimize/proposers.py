"""Prompt-rewrite proposer for the optimization loop (P1).

Per §H of the build spec, ``harden``/``aharden`` and the entire ``eval`` public
API stay 100% unchanged. The metaprompt that turns failing cases into a complete
revised system prompt lives here, so *optimize* (not *eval*) owns the
generation-for-apply step. We reuse only harden's internal failure-rendering
helper, ``_failures_text`` — a deliberate, documented private import.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

# DOCUMENTED PRIVATE REUSE (build spec §H): _failures_text renders failing cases
# from an EvalResults / SimulationResults into the same text block harden feeds
# its LLM. We depend on this internal rather than duplicating the logic; if
# harden's internals move, THIS import is the seam to update. harden's *public*
# API is untouched.
from fastaiagent.eval.harden import MAX_FAILURES_SHOWN, _failures_text

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.eval.results import EvalResults
    from fastaiagent.eval.scorer import Scorer
    from fastaiagent.llm.client import LLMClient

logger = logging.getLogger(__name__)

# How many failing train cases the proposer is shown — the same cap as harden().
_MAX_PROPOSER_FAILURES = MAX_FAILURES_SHOWN

_REWRITE_SYSTEM = (
    "You are an expert prompt engineer improving an AI agent's system prompt. "
    "You are given the current system prompt and cases where the agent failed. "
    "Produce complete, ready-to-use revised system prompts that fix the failures "
    "while preserving the agent's existing correct behavior. Make minimal, "
    "targeted edits; do not drop important instructions and do not pad length. "
    "Respond with JSON only."
)


def _strip_fences(raw: str) -> str:
    """Strip a ```` ```json ```` / ```` ``` ```` fence the same way harden does."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()


def _parse_proposals(raw: str, n: int) -> list[tuple[str, str]]:
    """Read up to ``n`` ``(system_prompt, rationale)`` pairs from a proposer reply.

    Accepts the requested ``{"proposals": [...]}`` object and the bare list of
    proposal objects some models send instead. Raises ``ValueError`` when the reply
    holds no usable proposal, so a reply nobody can read is reported, not taken
    for "nothing to improve".
    """
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not JSON ({exc.msg})") from exc
    items = data.get("proposals") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("reply has no list of proposals")

    out: list[tuple[str, str]] = []
    for item in items:
        if len(out) >= n:
            break
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("system_prompt") or "").strip()
        if not prompt:
            continue
        out.append((prompt, str(item.get("rationale") or "").strip()))
    if not out:
        raise ValueError("reply has no proposal with a system_prompt")
    return out


async def _arequest_rewrites(
    *,
    current_prompt: str,
    results: EvalResults,
    llm: LLMClient,
    n: int,
    agent_name: str = "agent",
) -> tuple[list[tuple[str, str]], str | None]:
    """``(proposals, error)``. ``error`` is set when the proposer could not run or
    its reply could not be read; it is ``None`` when it ran, and when there was
    nothing to learn from (no failing case — ``proposals`` is then empty)."""
    failures, count = _failures_text(results, max_cases=_MAX_PROPOSER_FAILURES)
    if count == 0:
        return [], None

    from fastaiagent.llm import SystemMessage, UserMessage

    shown = min(count, _MAX_PROPOSER_FAILURES)
    heading = "Failing cases" if shown == count else f"Failing cases ({shown} of {count})"
    user = (
        f"Agent name: {agent_name}\n\n"
        f'Current system prompt:\n"""\n{current_prompt}\n"""\n\n'
        f"{heading}:\n{failures}\n\n"
        f"Propose {n} distinct revised system prompts that would make these cases pass. "
        "Vary the approach across proposals. Respond with JSON only:\n"
        '{"proposals": [{"system_prompt": "<full revised prompt>", '
        '"rationale": "<what changed and why>"}]}'
    )

    def one_line(text: str) -> str:  # provider errors carry multi-line JSON bodies
        return " ".join(text.split())[:300]

    try:
        resp = await llm.acomplete([SystemMessage(_REWRITE_SYSTEM), UserMessage(user)])
    except Exception as exc:
        return [], one_line(f"{type(exc).__name__}: {exc}")
    try:
        return _parse_proposals(resp.content or "", n), None
    except ValueError as exc:
        return [], one_line(f"unreadable reply: {exc}")


async def propose_prompt_rewrites(
    *,
    current_prompt: str,
    results: EvalResults,
    llm: LLMClient,
    n: int,
    agent_name: str = "agent",
) -> list[tuple[str, str]]:
    """Return up to ``n`` ``(revised_system_prompt, rationale)`` proposals.

    Empty when there are no failing cases to learn from (the prompt already passes
    the split), and when the proposer can't run or its reply can't be read — that
    case is logged as a warning. ``optimize()`` itself records a failed proposer in
    the report rather than treating it as "no improvement".
    """
    proposals, error = await _arequest_rewrites(
        current_prompt=current_prompt,
        results=results,
        llm=llm,
        n=n,
        agent_name=agent_name,
    )
    if error:
        logger.warning("optimize: the prompt proposer failed: %s", error)
    return proposals


async def bootstrap_demos(
    *,
    agent: Agent,
    train_items: list[dict[str, Any]],
    scorers: list[Any],
    judge: Scorer | None,
    k: int,
    include_favorites: bool = True,
    exclude_inputs: Collection[Any] = (),
    before_teacher_eval: Callable[[int], bool] | None = None,
) -> list[dict[str, Any]]:
    """Build up to ``k`` few-shot demos from the TRAIN split (P2 KB lever).

    Gold-first: train items with a non-empty ``expected_output`` become free
    ``{"input", "output"}`` demos (no teacher call). Optionally augments with
    ``curate_from_traces(filter="favorites")`` (captured-good production runs). If
    still short of ``k``, teacher-bootstraps the no-gold items — runs the agent and
    metric-filters with the same scorers (+ judge), keeping passing
    ``(input, actual_output)`` pairs (DSPy ``BootstrapFewShot``).

    This is a **proposal-time** step (a teacher), analogous to
    :func:`propose_prompt_rewrites`. It uses ``aevaluate`` as a metric filter
    (spec §5) — it is *not* the selection seam; the loop's accept/reject still
    goes only through ``score_candidate``.

    ``exclude_inputs`` are inputs no demo may carry — the loop passes every dev and
    holdout input. Favorites come from production traces, not from the train split,
    and an eval set curated from those same favorites otherwise handed the agent
    the answers it was then scored on (1.85.0). ``before_teacher_eval`` is called
    with the number of cases the teacher pass would run; returning ``False`` skips
    it (the loop's cost governors).
    """
    demos: list[dict[str, Any]] = []
    seen: set[str] = {str(x).strip() for x in exclude_inputs}

    def _add(inp: Any, out: Any) -> None:
        key = str(inp).strip()
        if key in seen or not str(out).strip():
            return
        seen.add(key)
        demos.append({"input": inp, "output": out})

    # 1. gold demos from train (free — no teacher call).
    for it in train_items:
        if str(it.get("expected_output") or "").strip():
            _add(it["input"], it["expected_output"])

    # 2. favorites traces (captured-good outputs for this agent).
    if include_favorites:
        try:
            from fastaiagent.eval.curate import curate_from_traces

            for f in curate_from_traces(filter="favorites", agent=agent.name, limit=50):
                if str(f.get("expected_output") or "").strip():
                    _add(f["input"], f["expected_output"])
        except Exception:
            pass  # best-effort — no favorites is fine

    # 3. teacher-bootstrap the gap from no-gold train items.
    if len(demos) < k:
        no_gold = [
            it
            for it in train_items
            if not str(it.get("expected_output") or "").strip()
            and str(it.get("input")).strip() not in seen
        ]
        if no_gold and (before_teacher_eval is None or before_teacher_eval(len(no_gold))):
            from fastaiagent.eval.dataset import Dataset
            from fastaiagent.eval.evaluate import aevaluate
            from fastaiagent.optimize.candidate import scorer_present

            run_scorers = list(scorers)
            if judge is not None and not scorer_present(run_scorers, judge):
                run_scorers.append(judge)
            if not run_scorers:
                run_scorers = ["exact_match"]
            # Proposal-time only (teacher metric-filter, spec §5) — NOT the
            # selection seam; the loop's accept/reject still goes solely through
            # score_candidate (CONTRACT 1).
            results = await aevaluate(
                agent.arun, Dataset.from_list(no_gold), run_scorers, persist=False
            )
            for case in results.cases:
                per = case.per_scorer or {}
                if per and all(d.get("passed") for d in per.values()):
                    _add(case.input, case.actual_output)

    return demos[:k]


def propose_fact_subsets(
    *,
    scope: str,
    scope_id: str,
    n: int,
    store: Any = None,
    project_id: str = "",
) -> list[list[int]]:
    """Propose up to ``n`` candidate fact-id subsets for the memory lever (P3).

    Pure **selection / ablation**: reads ``store.list_active(scope, scope_id,
    project_id)`` (``store`` defaults to the local ``MemoryStore``)
    and returns subsets of the *existing* fact ids, ranked by confidence then
    recency (full set + progressively smaller high-confidence subsets). It never
    creates/edits/deletes/supersedes facts — the selection is run-local and leaves
    the audit chain untouched. Returns ``[]`` when there are no active facts (the
    caller skips the lever).
    """
    from fastaiagent.learn.store import MemoryStore

    facts = (store or MemoryStore()).list_active(scope, scope_id, project_id)  # type: ignore[arg-type]
    if not facts:
        return []
    ranked = sorted(
        facts,
        key=lambda f: (
            f.confidence if f.confidence is not None else 1.0,
            f.created_at or 0.0,
        ),
        reverse=True,
    )
    total = len(ranked)
    subsets: list[list[int]] = []
    seen_sizes: set[int] = set()
    for frac in (1.0, 0.66, 0.33):  # ablate down from "inject all"
        k = max(1, round(frac * total))
        if k in seen_sizes:
            continue
        seen_sizes.add(k)
        subsets.append([f.id for f in ranked[:k]])
        if len(subsets) >= n:
            break
    return subsets
