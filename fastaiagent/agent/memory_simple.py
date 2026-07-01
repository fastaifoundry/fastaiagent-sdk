"""``Memory`` — the simple, front-door memory API.

One object, tier-aware, with progressive-disclosure keywords. It is both:

- **agent-attachable** — implements the memory contract
  (``get_context`` / ``add`` / …) so ``Agent(memory=Memory(...))`` is a drop-in;
- **a direct store** — ``persist`` / ``retrieve`` / ``forget`` for tiered facts.

Under the hood it composes the existing block engine
(:class:`~fastaiagent.agent.memory.ComposableMemory` + blocks), so the shipped
``memory.*`` trace spans, the fact store, and safe-by-default scoping all apply.
The raw blocks remain available for advanced/custom behaviours.

Mental model — three tiers:

- ``global``  → facts true for everyone using the agent (store scope ``agent``)
- ``user``    → per-user personalization (store scope ``user``; needs an id)
- ``session`` → the ephemeral conversation window (not a durable store)

``project_id`` is an orthogonal tenant partition applied across tiers.

Example::

    from fastaiagent import Agent, LLMClient, Memory

    mem = Memory(location="sqlite")
    mem.persist("Return policy is 30 days", tier="global")

    agent = Agent(name="support", llm=llm,
                  memory=Memory(user_id=lambda ctx: ctx.state.user_id, learn=llm))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastaiagent.agent._memory_tracing import memory_store_span
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock,
    PersistentFactBlock,
    ScopeId,
    SummaryBlock,
    VectorBlock,
)

if TYPE_CHECKING:
    from fastaiagent.learn import Fact
    from fastaiagent.llm.client import LLMClient
    from fastaiagent.llm.message import Message

_TIERS = ("global", "user", "session")


def _tier_to_scope(tier: str) -> str:
    if tier == "global":
        return "agent"
    if tier == "user":
        return "user"
    if tier == "session":
        raise NotImplementedError(
            "the 'session' tier is the ephemeral conversation window, not a "
            "durable store — it can't be persisted/retrieved in Phase 1"
        )
    raise ValueError(f"tier must be one of global|user|session, got {tier!r}")


def _make_store(location: Any):
    """Resolve ``location`` to a fact store. Phase 1: sqlite or an instance."""
    from fastaiagent.learn import MemoryStore

    if location in (None, "sqlite"):
        return MemoryStore()
    # A MemoryStore / FactStore-like instance (duck-typed).
    if hasattr(location, "add") and hasattr(location, "list_active"):
        return location
    if isinstance(location, str):
        raise NotImplementedError(
            f"external memory location {location!r} is Phase 2 — pass 'sqlite' "
            "or a MemoryStore instance for now"
        )
    raise TypeError("location must be 'sqlite', a store instance, or a connection string")


def _make_recall_store(recall: Any):
    if recall == "auto":
        from fastaiagent.kb.backends.faiss import FaissVectorStore

        # In-process store — fine for dev/single-node. For real multi-user, pass
        # a shared VectorStore instance instead of "auto".
        return FaissVectorStore(dimension=384, index_type="flat")
    return recall  # assume a VectorStore instance


class Memory:
    """Tiered, pluggable memory — the recommended default for agents.

    Args:
        location: ``"sqlite"`` (default local.db) or a ``MemoryStore`` instance.
            External connection strings are Phase 2.
        user_id: personalization key for the user tier — a string, or a
            ``(RunContext) -> str`` resolver evaluated per run (one agent, many
            users). Unresolved ⇒ no personal facts (safe).
        agent_id: partition for the global tier (shared truth). If set, global
            facts are injected on every turn.
        project_id: tenant partition applied across tiers.
        window: recent turns kept in the session/working window.
        learn: an ``LLMClient`` → extract + persist durable user facts each turn.
        summarize: an ``LLMClient`` → roll older turns into a running summary.
        recall: ``"auto"`` (in-process FAISS) or a ``VectorStore`` → semantic
            recall over past exchanges.
        dedupe: drop recalled content an earlier tier already injected.
    """

    def __init__(
        self,
        *,
        location: Any = "sqlite",
        user_id: ScopeId | None = None,
        agent_id: str | None = None,
        project_id: str = "",
        window: int = 20,
        learn: LLMClient | None = None,
        summarize: LLMClient | None = None,
        recall: Any = None,
        dedupe: bool = False,
    ):
        self._store = _make_store(location)
        self._project_id = project_id
        self._agent_id = agent_id
        self._user_id = user_id
        self._window = window
        self._learn = learn
        self._summarize = summarize
        self._recall = recall
        self._dedupe = dedupe

        # When user_id is a per-run resolver, each user gets their OWN working
        # memory (window + in-conversation blocks) so concurrent/interleaved
        # sessions on one Memory instance never cross-contaminate — not just the
        # durable facts, but the live window too. Static/absent user_id → one.
        self._dynamic = callable(user_id)
        self._per_user: dict[str, ComposableMemory] = {}
        self._single: ComposableMemory | None = (
            None
            if self._dynamic
            else self._build_composable(user_id if isinstance(user_id, str) else None)
        )

    def _build_composable(self, user_scope_id: str | None) -> ComposableMemory:
        """Compose the block engine for one subject (or an anonymous window)."""
        blocks: list[Any] = []
        # Global tier (shared truth) — only when an agent_id is given.
        if self._agent_id is not None:
            blocks.append(
                PersistentFactBlock(
                    scope="agent",
                    scope_id=self._agent_id,
                    project_id=self._project_id,
                    store=self._store,
                )
            )
        # User tier — write (learn) then read — only when we have a subject.
        if user_scope_id:
            if self._learn is not None:
                blocks.append(
                    FactExtractionBlock(
                        llm=self._learn,
                        persist=True,
                        scope="user",
                        scope_id=user_scope_id,
                        project_id=self._project_id,
                        store=self._store,
                    )
                )
            blocks.append(
                PersistentFactBlock(
                    scope="user",
                    scope_id=user_scope_id,
                    project_id=self._project_id,
                    store=self._store,
                )
            )
        if self._summarize is not None:
            blocks.append(SummaryBlock(llm=self._summarize))
        if self._recall is not None:
            blocks.append(
                VectorBlock(
                    store=_make_recall_store(self._recall),
                    dedupe_against_upstream=self._dedupe,
                )
            )
        return ComposableMemory(blocks=blocks, primary=AgentMemory(max_messages=self._window))

    def _resolve_user_scope_id(self) -> str:
        from fastaiagent.agent.context import get_active_run_context

        ctx = get_active_run_context()
        if ctx is None:
            return ""
        try:
            return str(self._user_id(ctx))  # type: ignore[misc]
        except Exception:
            return ""

    def _active(self) -> ComposableMemory:
        """The working memory for the current run (per-user when dynamic)."""
        if not self._dynamic:
            assert self._single is not None
            return self._single
        uid = self._resolve_user_scope_id()
        mem = self._per_user.get(uid)
        if mem is None:
            mem = self._build_composable(uid or None)
            self._per_user[uid] = mem
        return mem

    # ── Agent-attachable contract (routes to the active working memory) ───────
    @property
    def blocks(self) -> list[Any]:
        """The active subject's blocks — exposed so tracing emits child spans."""
        return self._active().blocks

    def get_context(self, query: str = "", max_messages: int | None = None) -> list[Message]:
        return self._active().get_context(query=query, max_messages=max_messages)

    def add(self, message: Message) -> None:
        self._active().add(message)

    def clear(self) -> None:
        self._active().clear()

    def reset_blocks(self) -> None:
        self._active().reset_blocks()

    @property
    def messages(self) -> list[Message]:
        return self._active().messages

    def save(self, path: Any) -> None:
        self._active().save(path)

    def load(self, path: Any) -> None:
        self._active().load(path)

    def __len__(self) -> int:
        return len(self._active())

    def __bool__(self) -> bool:
        return True

    # ── Direct store verbs ────────────────────────────────────────────────────
    def _scope_and_id(self, tier: str, id: str) -> tuple[str, str]:
        scope = _tier_to_scope(tier)
        scope_id = id or (self._agent_id or "" if tier == "global" else "")
        return scope, scope_id

    def persist(
        self, content: str, *, tier: str = "user", id: str = "", confidence: float = 1.0
    ) -> int:
        """Store a durable fact verbatim. Returns its row id.

        ``tier="user"`` requires an explicit ``id``. Direct persists are
        recorded as source ``manual`` (no trace) — automatic, trace-stamped
        facts come from ``learn=`` on attach.
        """
        from fastaiagent.learn import Fact

        scope, scope_id = self._scope_and_id(tier, id)
        if tier == "user" and not scope_id:
            raise ValueError("persist(tier='user') requires id=<user id>")
        with memory_store_span(
            "persist", tier=tier, scope=scope, scope_id=scope_id, project_id=self._project_id
        ) as h:
            fid = self._store.add(
                Fact(
                    scope=scope,  # type: ignore[arg-type]
                    scope_id=scope_id,
                    fact=content,
                    confidence=confidence,
                    project_id=self._project_id,
                )
            )
            h.count = 1
        return fid

    def retrieve(
        self,
        query: str | None = None,
        *,
        tier: str = "user",
        id: str = "",
        limit: int | None = None,
    ) -> list[Fact]:
        """Return durable facts for a tier/id. Semantic ``query`` recall is Phase 2.

        Safe-by-default: ``tier="user"`` with no ``id`` returns ``[]``.
        """
        if query is not None:
            raise NotImplementedError(
                "semantic retrieve(query=...) is Phase 2; use retrieve(tier=, id=) "
                "for scope-based recall"
            )
        scope, scope_id = self._scope_and_id(tier, id)
        with memory_store_span(
            "retrieve", tier=tier, scope=scope, scope_id=scope_id, project_id=self._project_id
        ) as h:
            facts = self._store.list_active(
                scope=scope,  # type: ignore[arg-type]
                scope_id=scope_id,
                project_id=self._project_id,
                limit=limit,
            )
            h.count = len(facts)
        return facts

    def forget(self, *, tier: str, id: str = "", fact: str | None = None) -> int:
        """Hard-delete durable facts for a tier/id. Returns rows removed.

        Refuses to mass-delete at user/project scope without an id (pass
        ``id="*"`` to delete every subject on purpose). With ``fact`` given,
        only that exact fact is removed.
        """
        scope, scope_id = self._scope_and_id(tier, id)
        with memory_store_span(
            "forget", tier=tier, scope=scope, scope_id=scope_id, project_id=self._project_id
        ) as h:
            n = self._store.delete(
                scope=scope,  # type: ignore[arg-type]
                scope_id=scope_id,
                project_id=self._project_id,
                fact=fact,
            )
            h.count = n
        return n
