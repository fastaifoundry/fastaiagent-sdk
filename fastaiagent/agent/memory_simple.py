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
    _resolve_dynamic_id,
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
        # e.g. "postgres://…" or "redis://…" → external FactStore backend.
        from fastaiagent.learn import make_fact_store

        return make_fact_store(location)
    raise TypeError("location must be 'sqlite', a store instance, or a connection string")


_NO_USER = (
    "this Memory keeps one window per user, and no user is resolved here; "
    "use mem.for_user(user_id).save(path) / .load(path)"
)


class _AnonymousMemory(ComposableMemory):
    """What a per-user ``Memory`` gives a caller whose user can't be resolved.

    Global facts, and nothing else: no window, no summary, no recall. Anything
    kept here would be shared by every unresolved caller, which is exactly the
    cross-user leak this exists to prevent — so writes are dropped.
    """

    def add(self, message: Message) -> None:
        return None

    def clear(self) -> None:
        return None

    def reset_blocks(self) -> None:
        return None

    def save(self, path: Any) -> None:
        raise ValueError(_NO_USER)

    def load(self, path: Any) -> None:
        raise ValueError(_NO_USER)


def _wrap_semantic(store: Any, semantic: Any, embedder: Any):
    """Wrap a fact store so facts are vector-indexed for semantic retrieve()."""
    from fastaiagent.learn.faststore import SemanticFactStore

    if embedder is None:
        from fastaiagent.kb.embedding import get_default_embedder

        embedder = get_default_embedder()
    if semantic == "auto":
        from fastaiagent.kb.backends.faiss import FaissVectorStore

        dim = len(embedder.embed(["probe"])[0])  # match the embedder's dimension
        index = FaissVectorStore(dimension=dim, index_type="flat")
    else:
        index = semantic  # assume a VectorStore instance
    return SemanticFactStore(store, index, embedder)


class Memory:
    """Tiered, pluggable memory — the recommended default for agents.

    Args:
        location: ``"sqlite"`` (default local.db), a store instance, or a
            ``postgres://`` / ``redis://`` connection string.
        user_id: personalization key for the user tier — a string, or a
            ``(RunContext) -> str`` resolver evaluated per run (one agent, many
            users). A resolver gives each user their own window. A caller it
            can't resolve (no context, ``None``, or the resolver raises) gets no
            conversation memory — global facts only — and writes nothing.
        agent_id: partition for the global tier (shared truth). If set, global
            facts are injected on every turn. Must be non-empty: ``""`` would
            read every agent's global facts, so it raises.
        project_id: tenant partition applied across tiers.
        window: messages kept in the session/working window.
        learn: an ``LLMClient`` → extract + persist durable user facts each turn.
        summarize: an ``LLMClient`` → roll older turns into a running summary.
        recall: ``"auto"`` (an in-process FAISS index per user) or a
            ``VectorStore`` → semantic recall over past exchanges. A store you
            pass is shared by every user; each user's recall is namespaced.
        dedupe: drop recalled content an earlier tier already injected.
        semantic: ``"auto"`` or a ``VectorStore`` → ``retrieve(query=...)`` by
            meaning.
        embedder: the embedder for ``semantic=`` and ``recall=``. Defaults to
            the best available one; ``"auto"`` indexes are sized to match it.
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
        semantic: Any = None,
        embedder: Any = None,
    ):
        if agent_id is not None and not agent_id:
            # An unset setting (e.g. os.environ.get("AGENT_ID", "")) must not
            # become "read every agent's global facts" on every turn.
            raise ValueError(
                'agent_id="" would read every agent\'s global facts; pass the '
                "agent's id, or leave agent_id unset for no global tier"
            )
        self._store = _make_store(location)
        # One embedder for everything that embeds (semantic facts, recall), so
        # an "auto" index is always sized to it and the model loads once.
        if embedder is None and (semantic is not None or recall is not None):
            from fastaiagent.kb.embedding import get_default_embedder

            embedder = get_default_embedder()
        self._embedder = embedder
        self._embedding_dim: int | None = None
        # Semantic layer: mirror facts into a vector index so retrieve(query=...)
        # works by meaning. Wraps the base store, so facts written by learn= are
        # indexed too (they share this handle).
        self._semantic = semantic is not None
        if self._semantic:
            self._store = _wrap_semantic(self._store, semantic, embedder)
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
        self._anonymous: _AnonymousMemory | None = None
        self._single: ComposableMemory | None = (
            None
            if self._dynamic
            else self._build_composable(user_id if isinstance(user_id, str) else None)
        )

    def _global_blocks(self) -> list[Any]:
        """The global tier (shared truth) — only when an agent_id is given."""
        if self._agent_id is None:
            return []
        return [
            PersistentFactBlock(
                scope="agent",
                scope_id=self._agent_id,
                project_id=self._project_id,
                store=self._store,
            )
        ]

    def _dimension(self) -> int:
        """The embedder's output dimension, probed once."""
        if self._embedding_dim is None:
            self._embedding_dim = len(self._embedder.embed(["probe"])[0])
        return self._embedding_dim

    def _recall_store(self) -> Any:
        if self._recall == "auto":
            from fastaiagent.kb.backends.faiss import FaissVectorStore

            # In-process index, one per subject. Pass a VectorStore instead to
            # share one store (each user's recall is namespaced).
            return FaissVectorStore(dimension=self._dimension(), index_type="flat")
        return self._recall  # assume a VectorStore instance

    def _recall_namespace(self, user_scope_id: str | None) -> str:
        if not user_scope_id:
            return "default"
        ns = f"user:{user_scope_id}"
        return f"{self._project_id}:{ns}" if self._project_id else ns

    def _build_composable(self, user_scope_id: str | None) -> ComposableMemory:
        """Compose the block engine for one subject (or the single window)."""
        blocks: list[Any] = self._global_blocks()
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
                    store=self._recall_store(),
                    embedder=self._embedder,
                    namespace=self._recall_namespace(user_scope_id),
                    dedupe_against_upstream=self._dedupe,
                )
            )
        return ComposableMemory(blocks=blocks, primary=AgentMemory(max_messages=self._window))

    def _active(self) -> ComposableMemory:
        """The working memory for the current run (per-user when dynamic)."""
        if not self._dynamic:
            assert self._single is not None
            return self._single
        uid = _resolve_dynamic_id(self._user_id)  # type: ignore[arg-type]
        if not uid:
            if self._anonymous is None:
                self._anonymous = _AnonymousMemory(blocks=self._global_blocks())
            return self._anonymous
        return self.for_user(uid)

    def for_user(self, user_id: str) -> ComposableMemory:
        """One user's working memory (window + in-conversation blocks).

        For use outside a run, where there is no current user — e.g.
        ``mem.for_user("alice").save(path)`` to persist Alice's window, or
        ``.load(path)`` to restore it before her next run.
        """
        if not self._dynamic:
            raise ValueError(
                "for_user() is for a per-user Memory(user_id=<resolver>); this "
                "Memory has a single window — call save()/load() on it directly"
            )
        if not user_id:
            raise ValueError("for_user() needs a non-empty user id")
        mem = self._per_user.get(user_id)
        if mem is None:
            mem = self._build_composable(user_id)
            self._per_user[user_id] = mem
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
        scope, scope_id = self._scope_and_id(tier, id)
        if query is not None:
            if not self._semantic:
                raise NotImplementedError(
                    "semantic retrieve(query=...) needs semantic= on Memory; "
                    "use retrieve(tier=, id=) for scope-based recall"
                )
            with memory_store_span(
                "retrieve", tier=tier, scope=scope, scope_id=scope_id, project_id=self._project_id
            ) as h:
                hits = self._store.search(
                    query,
                    scope=scope,  # type: ignore[arg-type]
                    scope_id=scope_id,
                    project_id=self._project_id,
                    top_k=limit or 10,
                )
                h.count = len(hits)
            return [f for f, _score in hits]
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

    def update(
        self,
        new_content: str,
        *,
        old: str | int,
        tier: str = "user",
        id: str = "",
        confidence: float = 1.0,
    ) -> int:
        """Replace a fact: persist ``new_content`` and supersede the ``old`` one.

        ``old`` is the exact old fact text or its id. The old row is preserved
        (marked superseded — it stays in the audit history), and the new fact's
        id is returned. Versioning by append + supersede, never overwrite.
        """
        from fastaiagent.learn import Fact

        scope, scope_id = self._scope_and_id(tier, id)
        if tier == "user" and not scope_id:
            raise ValueError("update(tier='user') requires id=<user id>")
        with memory_store_span(
            "update", tier=tier, scope=scope, scope_id=scope_id, project_id=self._project_id
        ) as h:
            if isinstance(old, int):
                old_id = old
            else:
                matches = [
                    f
                    for f in self._store.list_active(
                        scope=scope,  # type: ignore[arg-type]
                        scope_id=scope_id,
                        project_id=self._project_id,
                    )
                    if f.fact == old
                ]
                if not matches:
                    raise ValueError(f"update: no active fact matching {old!r} to supersede")
                old_id = matches[0].id  # type: ignore[assignment]
            new_id = self._store.add(
                Fact(
                    scope=scope,  # type: ignore[arg-type]
                    scope_id=scope_id,
                    fact=new_content,
                    confidence=confidence,
                    project_id=self._project_id,
                )
            )
            self._store.supersede(old_id, new_id)
            h.count = 1
        return new_id

    def forget(self, *, tier: str, id: str = "", fact: str | None = None) -> int:
        """Hard-delete durable facts for a tier/id. Returns rows removed.

        Refuses to mass-delete without an id — at user scope, and at global
        scope when the Memory has no ``agent_id`` (pass ``id="*"`` to delete
        every subject on purpose). With ``fact`` given, only that exact fact is
        removed.
        """
        scope, scope_id = self._scope_and_id(tier, id)
        if scope == "agent" and not scope_id:
            # At agent scope an empty id matches every agent — a one-line call
            # would wipe all global facts. Mirror the user tier's refusal.
            raise ValueError(
                'forget(tier="global") needs agent_id= on the Memory or id=; '
                'pass id="*" to delete every agent\'s global facts on purpose'
            )
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
