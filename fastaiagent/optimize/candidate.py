"""Candidate (a point in the search space) + the clone-and-patch seam.

``apply_candidate`` builds a *fresh* Agent per candidate by re-invoking the
constructor — the user's agent is never mutated. From P2, memory is isolated per
candidate via ``block.isolated_copy()`` (share external handles, reset in-process
state) and the few-shot lever injects a ``FewShotBlock``.
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory, MemoryLike
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.eval.results import EvalResults
    from fastaiagent.eval.scorer import Scorer


@dataclass
class Candidate:
    """A point in the optimization search space.

    ``None`` on a lever field means "inherit from the current best". P1 moves
    ``system_prompt``; P2 activates ``fewshot_demos``; ``fact_ids`` (P3) is part
    of the frozen data model so later phases add drivers, not schema.
    """

    system_prompt: str | None = None
    fewshot_demos: list[dict[str, Any]] | None = None
    fact_ids: list[int] | None = None
    parent_id: str | None = None
    origin: str = ""
    rationale: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "origin": self.origin,
            "rationale": self.rationale,
            "system_prompt": self.system_prompt,
            "fewshot_demos": self.fewshot_demos,
            "fact_ids": self.fact_ids,
        }


@dataclass
class CandidateScore:
    """A candidate's score on one split."""

    candidate_id: str
    split: str
    score: float
    pass_rate: float
    n: int
    per_metric: dict[str, float] = field(default_factory=dict)
    eval_run_id: str | None = None
    # Underlying EvalResults — kept in-memory for the proposer (it needs the
    # failing cases). Not serialized / not part of equality.
    results: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_eval(
        cls,
        candidate_id: str,
        split: str,
        results: EvalResults,
        *,
        primary_metric: str | None,
    ) -> CandidateScore:
        """Roll an ``EvalResults`` up into a scalar selection score.

        Uses ``Scorecard.from_eval_results``: the ``primary_metric``'s ``avg_score``
        when set and present, otherwise the overall pass-rate.
        """
        from fastaiagent.eval.results import Scorecard

        card = Scorecard.from_eval_results(results)
        per_metric = {m.name: m.avg_score for m in card.metrics}
        if primary_metric is not None and primary_metric in per_metric:
            score = per_metric[primary_metric]
        else:
            if primary_metric is not None:
                warnings.warn(
                    f"primary_metric={primary_metric!r} not among scored metrics "
                    f"{sorted(per_metric)}; falling back to overall pass-rate.",
                    stacklevel=2,
                )
            score = card.overall_pass_rate
        n = max((m.n for m in card.metrics), default=0)
        return cls(
            candidate_id=candidate_id,
            split=split,
            score=score,
            pass_rate=card.overall_pass_rate,
            n=n,
            per_metric=per_metric,
            eval_run_id=getattr(results, "run_id", None),
            results=results,
        )


def _clone_memory_blocks(
    memory: MemoryLike | None,
    *,
    allow_writable_memory: bool = False,
) -> AgentMemory | ComposableMemory | Memory | None:
    """Return a per-candidate-isolated copy of ``memory`` (P2).

    Each candidate eval gets fresh in-process memory state so one candidate's
    turns never bleed into another's. External handles (llm, store) are shared
    via ``block.isolated_copy()``; the primary window starts empty.

    ``None`` → ``None``. A plain ``AgentMemory`` → a fresh empty one. A
    ``ComposableMemory`` → fresh blocks (via ``isolated_copy``) + fresh primary.
    A ``Memory`` → a ``Memory`` with the same configuration and empty windows,
    so per-user routing survives. A block that can't be isolated
    (``VectorBlock`` raises ``MemoryIsolationError``) aborts the run unless
    ``allow_writable_memory=True``, which shares it with a warning (accepting
    cross-candidate bleed).
    """
    if memory is None:
        return None
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import MemoryIsolationError
    from fastaiagent.agent.memory_simple import Memory

    if isinstance(memory, Memory):
        # Its blocks are the current caller's (outside a run, the anonymous
        # caller's), so copying them would drop every user's tier and share
        # one window between all users.
        return memory._derive(allow_writable_memory=allow_writable_memory)

    blocks = getattr(memory, "blocks", None)
    if blocks is None:
        # Plain AgentMemory — just a sliding window; fresh empty copy.
        return AgentMemory(max_messages=getattr(memory, "max_messages", None))

    new_blocks = []
    for b in blocks:
        try:
            new_blocks.append(b.isolated_copy())
        except MemoryIsolationError:
            if not allow_writable_memory:
                raise
            warnings.warn(
                f"{type(b).__name__}: sharing external state across candidate "
                "evaluations (allow_writable_memory=True); dev scores may be affected "
                "by cross-candidate writes.",
                stacklevel=2,
            )
            new_blocks.append(b)
    primary = getattr(memory, "primary", None)
    new_primary = AgentMemory(max_messages=getattr(primary, "max_messages", None))
    return ComposableMemory(blocks=new_blocks, primary=new_primary)


class _AllowlistStore:
    """Read-only ``MemoryStore`` wrapper restricting ``list_active`` to a fixed set
    of fact ids — the memory lever's run-local selection (P3).

    It only *reads* (``list_active``); it never creates, edits, deletes, or
    supersedes facts, so the learned-memory audit chain is untouched. Filters by
    id first, then applies the block's ``limit`` (so a small allowlist isn't
    pre-truncated by ``max_facts``).
    """

    def __init__(self, fact_ids: list[int], inner: Any = None):
        from fastaiagent.learn.store import MemoryStore

        self._ids = set(fact_ids)
        self._inner = inner if inner is not None else MemoryStore()

    def list_active(
        self, scope: str, scope_id: str = "", project_id: str = "", limit: int | None = None
    ) -> list[Any]:
        facts = [
            f
            for f in self._inner.list_active(scope, scope_id, project_id, limit=None)  # type: ignore[arg-type]
            if f.id in self._ids
        ]
        return facts[:limit] if limit is not None else facts


@dataclass(frozen=True)
class _FactSource:
    """Where the memory lever finds facts: the agent's own store, scope and project.

    ``store=None`` is the default local ``MemoryStore`` (``local.db``).
    """

    scope: str
    scope_id: str
    project_id: str = ""
    store: Any = None

    def resolved_store(self) -> Any:
        if self.store is not None:
            return self.store
        from fastaiagent.learn.store import MemoryStore

        return MemoryStore()


def _resolve_memory_source(agent: Agent) -> _FactSource:
    """Where the agent's memory reads the facts the lever selects among.

    A ``Memory`` → its global tier in its own store and project. A
    ``PersistentFactBlock`` with a fixed id → that block's scope, project and
    store; one that resolves its id per run (per user) is skipped — its facts
    differ for every user, and the lever selects the agent's shared facts.
    Otherwise ``("agent", agent.name)`` in the default local store.
    """
    from fastaiagent.agent.memory_simple import Memory

    mem = getattr(agent, "memory", None)
    if isinstance(mem, Memory):
        # The lever replaces the global fact block, so it selects among the
        # agent's global facts — never the facts of whichever user is current.
        return _FactSource("agent", mem._agent_id or agent.name, mem._project_id, mem._store)
    for b in getattr(mem, "blocks", None) or []:
        if type(b).__name__ == "PersistentFactBlock" and not callable(b.scope_id):
            return _FactSource(b.scope, b.scope_id, b.project_id, b._store)
    return _FactSource("agent", agent.name)


def _resolve_memory_scope(agent: Agent) -> tuple[str, str]:
    """``(scope, scope_id)`` of :func:`_resolve_memory_source`."""
    src = _resolve_memory_source(agent)
    return src.scope, src.scope_id


def _inject_block(
    memory: AgentMemory | ComposableMemory | Memory | None, block: Any, replace_name: str
) -> ComposableMemory | Memory:
    """Add ``block`` to ``memory``, replacing any existing block of the same name
    (so re-optimization doesn't stack). Wraps a plain ``AgentMemory`` / ``None``
    in a ``ComposableMemory`` as needed; a ``Memory`` puts it in every user's
    memory and stays a ``Memory``.
    """
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_simple import Memory

    if isinstance(memory, Memory):
        memory._override_block(block, replace_name)
        return memory
    if memory is None:
        return ComposableMemory(blocks=[block], primary=AgentMemory())
    if isinstance(memory, ComposableMemory):
        optimized_name = f"{replace_name}.optimized"

        def per_user(b: Any) -> bool:
            return callable(getattr(b, "scope_id", None))

        # A block that resolves its subject per run holds each user's own facts:
        # it stays. Anything this lever injected before is replaced (no stacking).
        kept = [
            b
            for b in memory.blocks
            if per_user(b) or getattr(b, "name", "") not in (replace_name, optimized_name)
        ]
        if any(getattr(b, "name", "") == replace_name for b in kept):
            block.name = optimized_name  # keep span labels and by_block keys distinct
        memory.blocks = [*kept, block]
        return memory
    return ComposableMemory(blocks=[block], primary=memory)


def apply_candidate(
    base: Agent, candidate: Candidate, *, allow_writable_memory: bool = False
) -> Agent:
    """Return a fresh Agent with the candidate's levers applied.

    P1 patches ``system_prompt``. P2 injects a ``FewShotBlock`` when
    ``fewshot_demos`` is set. P3 injects a ``PersistentFactBlock`` backed by an
    ``_AllowlistStore`` when ``fact_ids`` is set (the memory lever's fact subset).
    Both replace any prior block of the same name (no stacking) inside an isolated
    copy of the agent's memory; facts render before examples. The user's agent is
    never mutated.
    """
    from fastaiagent.agent.agent import Agent

    new_prompt = (
        candidate.system_prompt if candidate.system_prompt is not None else base.system_prompt
    )
    new_memory = _clone_memory_blocks(base.memory, allow_writable_memory=allow_writable_memory)

    if candidate.fact_ids is not None:
        from fastaiagent.agent.memory_blocks import PersistentFactBlock

        src = _resolve_memory_source(base)
        new_memory = _inject_block(
            new_memory,
            PersistentFactBlock(
                scope=src.scope,
                scope_id=src.scope_id,
                project_id=src.project_id,
                store=_AllowlistStore(candidate.fact_ids, inner=src.store),
            ),
            "persistent_facts",
        )

    if candidate.fewshot_demos is not None:
        from fastaiagent.agent.memory_blocks import FewShotBlock

        new_memory = _inject_block(new_memory, FewShotBlock(candidate.fewshot_demos), "fewshot")

    return Agent(
        name=base.name,
        system_prompt=new_prompt,
        llm=base.llm,
        tools=base.tools,
        guardrails=base.guardrails,
        memory=new_memory,
        config=base.config,
        output_type=base.output_type,
        middleware=base.middleware,
        checkpointer=base._checkpointer,
        agent_id=base.agent_id,
    )


def scorer_present(scorers: list[Any], judge: Scorer) -> bool:
    """CONTRACT 3: is an equivalent judge already in ``scorers``?

    Match on identity first, then on ``name`` so a user-supplied judge with the
    same name isn't appended (and billed) twice.
    """
    judge_name = getattr(judge, "name", None)
    for s in scorers:
        if s is judge:
            return True
        if judge_name is not None and getattr(s, "name", None) == judge_name:
            return True
    return False
