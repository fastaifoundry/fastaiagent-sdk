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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastaiagent.agent.memory import AgentMemory
from fastaiagent.llm.message import Message

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.agent.memory import ComposableMemory, MemoryLike
    from fastaiagent.agent.memory_simple import Memory
    from fastaiagent.eval.results import EvalResults
    from fastaiagent.eval.scorer import Scorer


class _NoConversation(AgentMemory):
    """The primary window of an agent that had no memory: it keeps nothing.

    The few-shot and memory levers carry their block in a ``ComposableMemory``,
    and a ``ComposableMemory`` needs a primary window. A real one turned a
    stateless agent into one that remembered every earlier run: eval cases bled
    into each other, and an applied winner put one user's request into the next
    user's prompt (1.87.0). This window stores and returns nothing, so the agent
    stays as stateless as it was.
    """

    def add(self, message: Message) -> None:
        return None

    def get_context(self, query: str = "", max_messages: int | None = None) -> list[Message]:
        return []

    def load(self, path: str | Path) -> None:
        return None


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
    """A candidate's score on one split.

    ``n`` counts every case in the split; ``errored`` of them raised instead of
    answering and count as failures in ``score`` and ``pass_rate``.
    """

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
    errored: int = 0

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

        The ``primary_metric``'s average score when set and present, otherwise the
        overall pass-rate. A case that raised instead of answering — a guardrail
        block, ``MaxIterationsError``, a provider error — has no scorer results and
        counts as a failure scoring 0 on every metric. ``evaluate()`` leaves such a
        case out of its scores, and selecting on those let a candidate that crashed
        on its hard cases outscore one that answered them (1.85.0).
        """
        errored = int(getattr(results, "errored_count", 0) or 0)
        per_metric: dict[str, float] = {}
        passed = total = 0
        for name, rlist in results.scores.items():
            n_metric = len(rlist) + errored
            if not n_metric:
                continue
            per_metric[name] = round(sum(r.score for r in rlist) / n_metric, 4)
            passed += sum(1 for r in rlist if r.passed)
            total += n_metric
        pass_rate = round(passed / total, 4) if total else 0.0

        if primary_metric is not None and primary_metric in per_metric:
            score = per_metric[primary_metric]
        else:
            # Every case erroring leaves no metric at all: that is a 0, not a
            # misnamed primary metric.
            if primary_metric is not None and per_metric:
                warnings.warn(
                    f"primary_metric={primary_metric!r} not among scored metrics "
                    f"{sorted(per_metric)}; falling back to overall pass-rate.",
                    stacklevel=2,
                )
            score = pass_rate
        n = max((len(rlist) for rlist in results.scores.values()), default=0) + errored
        return cls(
            candidate_id=candidate_id,
            split=split,
            score=score,
            pass_rate=pass_rate,
            n=n,
            per_metric=per_metric,
            eval_run_id=getattr(results, "run_id", None),
            results=results,
            errored=errored,
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
    from fastaiagent.agent.memory import ComposableMemory
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
    new_primary = (
        _NoConversation()
        if isinstance(primary, _NoConversation)
        else AgentMemory(max_messages=getattr(primary, "max_messages", None))
    )
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
    memory and stays a ``Memory``. An agent with no memory gets a primary window
    that keeps no conversation — it had none, and must not gain one.
    """
    from fastaiagent.agent.memory import ComposableMemory
    from fastaiagent.agent.memory_simple import Memory

    if isinstance(memory, Memory):
        memory._override_block(block, replace_name)
        return memory
    if memory is None:
        return ComposableMemory(blocks=[block], primary=_NoConversation())
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

    The candidate is a copy of ``base`` — same class, agent path label and
    registry-prompt link — with only the levers changed, so what is scored is the
    user's agent (rebuilding a plain ``Agent`` dropped a subclass and the
    ``prompt_slug``, 1.85.0). A changed prompt drops the ``prompt_slug``: the
    registry prompt it names is no longer the prompt the agent runs.
    """
    import copy

    from fastaiagent.agent.middleware import _MiddlewarePipeline

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

    new = copy.copy(base)
    if candidate.system_prompt is not None and candidate.system_prompt != base.system_prompt:
        new.system_prompt = candidate.system_prompt
        new.prompt_slug = None
        new._prompt_provenance = None
    new.memory = new_memory
    # Fresh containers, so nothing a candidate run does reaches the user's agent.
    new.tools = list(base.tools)
    new.guardrails = list(base.guardrails)
    new.middleware = list(base.middleware)
    new._mw_pipeline = _MiddlewarePipeline(new.middleware)
    try:
        from fastaiagent._platform.push import track_agent

        track_agent(new)  # as Agent.__init__ does, so connect() can register it
    except Exception:
        pass
    return new


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


def scorer_name(scorer: Any) -> str | None:
    """The name a scorer reports results under (a built-in's string is its name)."""
    return scorer if isinstance(scorer, str) else getattr(scorer, "name", None)


def is_llm_scorer(scorer: Any) -> bool:
    """Does scoring one case call a model? Counted by ``max_judge_calls``.

    The SDK's model-backed scorers — ``LLMJudge``/``GEval``, ``DecisionJudge`` and
    the RAG, agent, session and safety metrics — all hold their client as
    ``_llm``. A built-in named by string is resolved the way ``evaluate()`` does.
    """
    if isinstance(scorer, str):
        from fastaiagent.eval.builtins import BUILTIN_SCORERS

        cls = BUILTIN_SCORERS.get(scorer)
        if cls is None:
            return False
        try:
            scorer = cls()
        except Exception:
            return False
    return hasattr(scorer, "_llm")
