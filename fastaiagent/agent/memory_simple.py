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

    mem = Memory(location="sqlite", agent_id="support")
    mem.persist("Return policy is 30 days", tier="global")

    agent = Agent(name="support", llm=llm,
                  memory=Memory(agent_id="support",
                                user_id=lambda ctx: ctx.state.user_id, learn=llm))
"""

from __future__ import annotations

import copy
import logging
import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from fastaiagent.agent._memory_tracing import memory_store_span
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock,
    MemoryIsolationError,
    PersistentFactBlock,
    PlaneFactBlock,
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


class _LazyLocalStore:
    """The default local store, opened on first use.

    Constructing a ``Memory`` must not create ``./.fastaiagent/`` in whatever
    directory the process happens to run in; persisting or reading a fact does.
    """

    def __init__(self) -> None:
        self._store: Any = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._store is None:
            from fastaiagent.learn import MemoryStore

            self._store = MemoryStore()
        return getattr(self._store, name)


def _make_store(location: Any):
    """Resolve ``location`` to a fact store: local SQLite, a URL, or an instance."""
    if location in (None, "sqlite"):
        return _LazyLocalStore()
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
        learn: an ``LLMClient`` → extract + persist durable facts from each
            user message (never from the model's replies).
        max_learned_facts: keep at most this many learned facts per user,
            deleting the oldest; facts you ``persist`` yourself are never
            touched. ``None`` = no cap. Default ``200``.
        summarize: an ``LLMClient`` → roll older turns into a running summary.
        recall: ``"auto"`` (an in-process FAISS index per user) or a
            ``VectorStore`` → semantic recall over past exchanges. A store you
            pass is shared by every user; each user's recall is namespaced.
        dedupe: drop recalled content an earlier tier already injected.
        semantic: ``"auto"`` or a ``VectorStore`` → ``retrieve(query=...)`` by
            meaning.
        embedder: the embedder for ``semantic=`` and ``recall=``. Defaults to
            the best available one; ``"auto"`` indexes are sized to match it.
        max_users: with a per-user resolver, keep at most this many users'
            windows in the process, dropping the least recently used. A dropped
            user's window, summary and ``"auto"`` recall start afresh; their
            durable facts are in the store and stay. ``None`` = no cap.
            Default ``10_000``.
        plane_agent_id: the agent's id on a connected Enterprise plane → also
            inject the curated facts the plane serves for it
            (:class:`PlaneFactBlock`), for every user and for unresolved
            callers. Read-only; with no connection it injects nothing.
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
        max_learned_facts: int | None = 200,
        max_users: int | None = 10_000,
        plane_agent_id: str | None = None,
    ):
        if max_users is not None and max_users < 1:
            raise ValueError("max_users must be at least 1, or None for no cap")
        if plane_agent_id is not None and not plane_agent_id:
            raise ValueError('plane_agent_id="" names no agent; pass the id or leave it unset')
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
        self._max_learned_facts = max_learned_facts
        self._summarize = summarize
        self._recall = recall
        self._dedupe = dedupe
        self._max_users = max_users
        self._plane_agent_id = plane_agent_id
        self._warned_eviction = False

        # When user_id is a per-run resolver, each user gets their OWN working
        # memory (window + in-conversation blocks) so concurrent/interleaved
        # sessions on one Memory instance never cross-contaminate — not just the
        # durable facts, but the live window too. Static/absent user_id → one.
        self._dynamic = callable(user_id)
        # Blocks optimize put in place of (or beside) the built-in ones, by name.
        self._overrides: dict[str, Any] = {}
        self._reset_windows()

    def _reset_windows(self) -> None:
        """Start every subject's working memory afresh."""
        self._per_user: OrderedDict[str, ComposableMemory] = OrderedDict()
        self._per_user_lock = threading.Lock()
        self._anonymous: _AnonymousMemory | None = None
        self._single: ComposableMemory | None = (
            None
            if self._dynamic
            else self._build_composable(self._user_id if isinstance(self._user_id, str) else None)
        )

    def _derive(self, *, allow_writable_memory: bool = False) -> Memory:
        """A copy with the same configuration and handles, and empty windows.

        For ``fastaiagent.optimize``: each candidate runs on its own copy, and
        the optimized agent gets one — still a ``Memory``, so per-user routing,
        the window size and the store verbs all survive. ``learn=`` and a
        ``recall=`` store you passed write to shared stores during a run, so
        they can't be isolated; they are refused unless
        ``allow_writable_memory=True``, which shares them with a warning.
        ``recall="auto"`` is an in-process index built per user of each copy.
        """
        writers = []
        if self._learn is not None:
            writers.append("learn=")
        if self._recall is not None and self._recall != "auto":
            writers.append("recall=")
        if writers and not allow_writable_memory:
            raise MemoryIsolationError(
                f"Memory({', '.join(writers)}...) writes to a shared store during a "
                "run, so it can't be isolated per candidate. Remove it to "
                "optimize, or pass allow_writable_memory=True to share it."
            )
        if writers:
            warnings.warn(
                f"Memory({', '.join(writers)}...): sharing external state across "
                "candidate evaluations (allow_writable_memory=True); dev scores may "
                "be affected by cross-candidate writes.",
                stacklevel=3,
            )
        new = copy.copy(self)
        new._overrides = dict(self._overrides)
        new._reset_windows()
        return new

    def _override_block(self, block: Any, replace_name: str) -> None:
        """Put ``block`` in every subject's memory in place of ``replace_name``.

        Windows start afresh, so call it on a copy from :meth:`_derive`.
        """
        self._overrides.pop(replace_name, None)
        self._overrides[replace_name] = block
        self._reset_windows()

    def _with_overrides(self, blocks: list[Any]) -> list[Any]:
        if not self._overrides:
            return blocks
        kept = [b for b in blocks if getattr(b, "name", "") not in self._overrides]
        # Each subject gets its own copy: an override holds no state across users.
        return kept + [b.isolated_copy() for b in self._overrides.values()]

    def _global_blocks(self) -> list[Any]:
        """The global tier (shared truth): the agent's facts when an agent_id is
        given, then the plane's curated facts when a plane_agent_id is."""
        blocks: list[Any] = []
        if self._agent_id is not None:
            blocks.append(
                PersistentFactBlock(
                    scope="agent",
                    scope_id=self._agent_id,
                    project_id=self._project_id,
                    store=self._store,
                )
            )
        if self._plane_agent_id is not None:
            # One per subject, like every block here: it caches per question.
            blocks.append(PlaneFactBlock(self._plane_agent_id))
        return blocks

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
                        # The user's own messages only: the model's replies are
                        # not facts about the user.
                        roles=("user",),
                        # The user-tier PersistentFactBlock below reads these
                        # same facts back from the store — inject them once.
                        inject=False,
                        max_persisted=self._max_learned_facts,
                    )
                )
            user_facts = PersistentFactBlock(
                scope="user",
                scope_id=user_scope_id,
                project_id=self._project_id,
                store=self._store,
            )
            # The global block above is "persistent_facts"; a second block of
            # the same name made one span label, one by_block key and one
            # optimize replace target out of two different blocks.
            user_facts.name = "persistent_facts.user"
            blocks.append(user_facts)
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
        return ComposableMemory(
            blocks=self._with_overrides(blocks), primary=AgentMemory(max_messages=self._window)
        )

    def _active(self) -> ComposableMemory:
        """The working memory for the current run (per-user when dynamic)."""
        if not self._dynamic:
            assert self._single is not None
            return self._single
        uid = _resolve_dynamic_id(self._user_id)  # type: ignore[arg-type]
        if not uid:
            if self._anonymous is None:
                self._anonymous = _AnonymousMemory(
                    blocks=self._with_overrides(self._global_blocks())
                )
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
        with self._per_user_lock:
            mem = self._per_user.get(user_id)
            if mem is not None:
                self._per_user.move_to_end(user_id)
                return mem
        # Built outside the lock so one user's build never stalls another's; if
        # two threads race, the first one in wins and the other build is unused.
        built = self._build_composable(user_id)
        with self._per_user_lock:
            mem = self._per_user.get(user_id)
            if mem is not None:
                self._per_user.move_to_end(user_id)
                return mem
            self._per_user[user_id] = built
            self._evict_over_cap()
        return built

    def _evict_over_cap(self) -> None:
        """Drop the least recently used users' windows beyond ``max_users``."""
        if self._max_users is None:
            return
        while len(self._per_user) > self._max_users:
            self._per_user.popitem(last=False)
            if not self._warned_eviction:
                self._warned_eviction = True
                logging.getLogger(__name__).warning(
                    "Memory: more than max_users=%d users' windows in this process; "
                    "dropping the least recently used (their durable facts stay). "
                    "Windows are in-process: persist them with "
                    "mem.for_user(id).save(path), or raise max_users.",
                    self._max_users,
                )

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

    @staticmethod
    def _warn_unscoped_global(verb: str) -> None:
        warnings.warn(
            f'{verb}(tier="global") with no agent_id on the Memory and no id= '
            "stores the fact under an empty agent id, which no "
            "Memory(agent_id=...) will ever inject. Set agent_id= on this Memory.",
            UserWarning,
            stacklevel=3,
        )

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
        if tier == "global" and not scope_id:
            self._warn_unscoped_global("persist")
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
        if tier == "global" and not scope_id:
            self._warn_unscoped_global("update")
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
