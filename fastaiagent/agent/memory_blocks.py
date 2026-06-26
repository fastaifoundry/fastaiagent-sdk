"""Memory blocks — composable long-term memory primitives.

A ``MemoryBlock`` observes every message passing through an agent
(``on_message``) and renders fragments back into the next prompt
(``render``). Compose any number of blocks with
:class:`fastaiagent.agent.memory.ComposableMemory` to augment an agent's
sliding-window history with persistent facts, summaries, and semantic
recall over past conversations.

Four block types ship:

- :class:`StaticBlock`          — a fixed system-level fact, injected every turn
- :class:`SummaryBlock`         — rolling LLM-generated summary of older turns
- :class:`VectorBlock`          — semantic recall over past messages via a VectorStore
- :class:`FactExtractionBlock`  — structured-output LLM extracts facts from each turn

Write your own by subclassing :class:`MemoryBlock` and implementing
``on_message`` and ``render``.

Future work: async parallel methods (``aon_message`` / ``arender``) are
not shipped in 0.4.0 — see the note in
:mod:`fastaiagent.kb.protocols` for the broader async roadmap.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastaiagent.llm.message import Message, MessageRole, SystemMessage

if TYPE_CHECKING:
    from fastaiagent.kb.chunking import Chunk
    from fastaiagent.kb.embedding import Embedder
    from fastaiagent.kb.protocols import VectorStore
    from fastaiagent.llm.client import LLMClient

_log = logging.getLogger(__name__)


__all__ = [
    "FactExtractionBlock",
    "MemoryBlock",
    "PersistentFactBlock",
    "PlaneFactBlock",
    "StaticBlock",
    "SummaryBlock",
    "VectorBlock",
]


class MemoryBlock(ABC):
    """Base class for pluggable memory blocks.

    Subclass and implement :meth:`on_message` and :meth:`render`. Persistence
    (``save`` / ``load``) is optional — default is a no-op so stateless blocks
    like :class:`StaticBlock` require no boilerplate.

    Blocks must be tolerant of partial state: a ``render`` call before any
    ``on_message`` call must not raise, just return an empty list.
    """

    name: str = ""

    @abstractmethod
    def on_message(self, message: Message) -> None:
        """Called for every message added to memory (user, assistant, tool)."""

    @abstractmethod
    def render(self, query: str) -> list[Message]:
        """Return message fragments to prepend to the next LLM prompt.

        ``query`` is the current user input, useful for query-conditioned
        blocks (e.g. :class:`VectorBlock`).
        """

    def save(self, path: Path) -> None:
        """Persist block state. Default: no-op."""

    def load(self, path: Path) -> None:
        """Load block state. Default: no-op."""


# ---------------------------------------------------------------------------
# StaticBlock
# ---------------------------------------------------------------------------


class StaticBlock(MemoryBlock):
    """A fixed system-level fact injected on every turn.

    Example::

        StaticBlock("The user's name is Upendra. They prefer terse answers.")
    """

    def __init__(self, text: str, name: str = "static"):
        self.text = text
        self.name = name

    def on_message(self, message: Message) -> None:
        return

    def render(self, query: str) -> list[Message]:
        if not self.text:
            return []
        return [SystemMessage(self.text)]


# ---------------------------------------------------------------------------
# SummaryBlock
# ---------------------------------------------------------------------------


class SummaryBlock(MemoryBlock):
    """Maintain a rolling summary of older turns using an LLM to compress.

    Every ``summarize_every`` messages, takes everything older than
    ``keep_last`` messages and asks the LLM for a concise one-paragraph
    summary. The running summary is injected as a single SystemMessage on
    every :meth:`render`.

    Args:
        llm: The :class:`fastaiagent.llm.client.LLMClient` to use for summarization.
        keep_last: Number of recent messages *not* to summarize.
        summarize_every: Refresh the summary every N messages seen.
        max_chars: Soft cap on the summary length; the LLM is asked to stay under.
    """

    name = "summary"

    def __init__(
        self,
        llm: LLMClient,
        keep_last: int = 10,
        summarize_every: int = 5,
        max_chars: int = 800,
    ):
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        if summarize_every < 1:
            raise ValueError("summarize_every must be >= 1")
        self.llm = llm
        self.keep_last = keep_last
        self.summarize_every = summarize_every
        self.max_chars = max_chars
        self._messages_seen: int = 0
        self._summary: str = ""
        self._archive: list[Message] = []

    def on_message(self, message: Message) -> None:
        self._archive.append(message)
        self._messages_seen += 1
        # Refresh summary when we cross a threshold AND there is something
        # older than keep_last to summarize.
        if self._messages_seen < self.summarize_every:
            return
        if self._messages_seen % self.summarize_every != 0:
            return
        if len(self._archive) <= self.keep_last:
            return
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        to_summarize = self._archive[: -self.keep_last]
        transcript = "\n".join(
            f"{m.role.value}: {m.content or ''}" for m in to_summarize if m.content
        )
        if not transcript.strip():
            return

        prev = f"\n\nPrevious summary:\n{self._summary}" if self._summary else ""
        prompt = (
            "You are maintaining a running conversation summary. Write a single "
            f"concise paragraph (<= {self.max_chars} chars) that captures the key "
            "facts, decisions, user preferences, and unresolved items from the "
            f"transcript below.{prev}\n\nTranscript:\n{transcript}\n\nSummary:"
        )
        try:
            from fastaiagent.llm.message import UserMessage

            response = self.llm.complete([UserMessage(prompt)])
            if response.content:
                self._summary = response.content.strip()[: self.max_chars]
        except Exception as err:
            _log.warning("SummaryBlock failed to refresh: %s", err)

    def render(self, query: str) -> list[Message]:
        if not self._summary:
            return []
        return [SystemMessage(f"Conversation summary so far: {self._summary}")]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self._summary,
            "messages_seen": self._messages_seen,
        }
        path.write_text(json.dumps(payload))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        self._summary = payload.get("summary", "")
        self._messages_seen = payload.get("messages_seen", 0)


# ---------------------------------------------------------------------------
# VectorBlock
# ---------------------------------------------------------------------------


class VectorBlock(MemoryBlock):
    """Semantic recall over past messages via a :class:`VectorStore`.

    Every incoming message with text content is embedded and stored. On
    :meth:`render`, the query is embedded and the top-``top_k`` past
    messages are returned as SystemMessage fragments.

    By default retrieval is ranked by cosine similarity alone. Long-running
    agents often want recent or important messages to outrank stale
    high-similarity messages — set ``recency_weight`` and/or
    ``importance_weight`` to enable the weighted-sum scorer:

        final_score = (1 - recency_weight - importance_weight) * cos_sim
                    + recency_weight    * exp(-age_seconds / half_life)
                    + importance_weight * importance

    The ``importance`` field is read from each chunk's metadata
    (``metadata['importance']``); messages without it default to ``1.0``.
    The two weights default to ``0.0`` so existing callers see no change.

    Args:
        store: Any object implementing the
            :class:`fastaiagent.kb.protocols.VectorStore` protocol.
        embedder: Any :class:`fastaiagent.kb.embedding.Embedder`. If ``None``,
            the default auto-selected embedder is used.
        top_k: Number of past messages to recall per turn.
        namespace: Metadata tag so multiple VectorBlocks over different
            stores don't collide. Stored on each chunk's metadata.
        min_content_chars: Messages shorter than this are not indexed
            (skip trivial "ok" / "yes" messages).
        recency_weight: Boost for recent chunks, in ``[0.0, 1.0]``. Default
            ``0.0`` (similarity-only — historical behaviour).
        importance_weight: Boost for high-importance chunks, in ``[0.0, 1.0]``.
            Default ``0.0``.
        recency_half_life_seconds: Time after which a chunk's recency
            contribution halves. Default ``3600`` (one hour). Lower values
            decay faster.
    """

    name = "vector"

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        top_k: int = 5,
        namespace: str = "default",
        min_content_chars: int = 10,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
        recency_half_life_seconds: float = 3600.0,
    ):
        from fastaiagent.kb.embedding import get_default_embedder

        if recency_weight < 0.0 or importance_weight < 0.0:
            raise ValueError("recency_weight and importance_weight must be >= 0")
        if recency_weight + importance_weight > 1.0:
            raise ValueError(
                "recency_weight + importance_weight must not exceed 1.0; "
                f"got {recency_weight + importance_weight:.3f}"
            )
        if recency_half_life_seconds <= 0.0:
            raise ValueError("recency_half_life_seconds must be > 0")

        self.store = store
        self.embedder: Embedder = embedder or get_default_embedder()
        self.top_k = top_k
        self.namespace = namespace
        self.min_content_chars = min_content_chars
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.recency_half_life_seconds = recency_half_life_seconds

    def _make_chunk(self, message: Message) -> Chunk | None:
        from fastaiagent.kb.chunking import Chunk

        content = (message.content or "").strip()
        if len(content) < self.min_content_chars:
            return None
        text = f"[{message.role.value}] {content}"
        # Stamp ``created_at`` (and optional ``importance`` if the message
        # carries one) so the recency/importance scorer has signal to work
        # with on retrieval. Existing chunks without these keys keep
        # working — ``_score_hits`` falls back to neutral defaults.
        metadata: dict[str, Any] = {
            "namespace": self.namespace,
            "role": message.role.value,
            "created_at": time.time(),
        }
        msg_importance = getattr(message, "importance", None)
        if isinstance(msg_importance, (int, float)):
            metadata["importance"] = float(msg_importance)
        return Chunk(
            id=str(uuid.uuid4()),
            content=text,
            metadata=metadata,
            index=0,
            start_char=0,
            end_char=len(text),
        )

    def on_message(self, message: Message) -> None:
        chunk = self._make_chunk(message)
        if chunk is None:
            return
        try:
            embedding = self.embedder.embed([chunk.content])[0]
            self.store.add([chunk], [embedding])
        except Exception as err:
            _log.warning("VectorBlock failed to index message: %s", err)

    def _score_hits(
        self, hits: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        """Apply optional recency + importance weights on top of similarity.

        Returns ``(chunk, final_score)`` sorted by ``final_score`` descending.
        With both weights at zero this preserves the input order — keeping
        identical behaviour for callers who don't opt in.
        """
        if self.recency_weight == 0.0 and self.importance_weight == 0.0:
            return hits

        sim_w = max(0.0, 1.0 - self.recency_weight - self.importance_weight)
        now = time.time()
        scored: list[tuple[Chunk, float]] = []
        for chunk, similarity in hits:
            created_at = chunk.metadata.get("created_at")
            if isinstance(created_at, (int, float)):
                age = max(0.0, now - float(created_at))
                recency = math.exp(-age / self.recency_half_life_seconds)
            else:
                recency = 0.0  # unknown age — no boost
            importance_raw = chunk.metadata.get("importance", 1.0)
            try:
                importance = float(importance_raw)
            except (TypeError, ValueError):
                importance = 1.0
            final = (
                sim_w * float(similarity)
                + self.recency_weight * recency
                + self.importance_weight * importance
            )
            scored.append((chunk, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def render(self, query: str) -> list[Message]:
        if not query or not query.strip():
            return []
        try:
            embedding = self.embedder.embed([query])[0]
        except Exception as err:
            _log.warning("VectorBlock failed to embed query: %s", err)
            return []
        try:
            hits = self.store.search(embedding, self.top_k)
        except Exception as err:
            _log.warning("VectorBlock search failed: %s", err)
            return []
        scored = self._score_hits(list(hits))
        relevant = [
            c for c, _ in scored
            if c.metadata.get("namespace", self.namespace) == self.namespace
        ]
        if not relevant:
            return []
        body = "\n".join(f"- {c.content}" for c in relevant)
        return [SystemMessage(f"Relevant prior exchanges:\n{body}")]


# ---------------------------------------------------------------------------
# FactExtractionBlock
# ---------------------------------------------------------------------------


class FactExtractionBlock(MemoryBlock):
    """Use a structured-output LLM call to extract facts from each turn and
    persist them as a deduplicated list. Rendered as a bullet list.

    Only user and assistant messages are inspected; tool messages are skipped.
    Facts are short, self-contained statements about the user or world state.

    Args:
        llm: LLM client for fact extraction. Use a cheap fast model
            (e.g. ``gpt-4o-mini``, ``claude-haiku-4-5``).
        max_facts: Cap the running fact list; oldest facts drop when exceeded.
        extract_every: Run extraction every N messages (1 = every message).
    """

    name = "facts"

    def __init__(
        self,
        llm: LLMClient,
        max_facts: int = 200,
        extract_every: int = 1,
    ):
        if max_facts < 1:
            raise ValueError("max_facts must be >= 1")
        if extract_every < 1:
            raise ValueError("extract_every must be >= 1")
        self.llm = llm
        self.max_facts = max_facts
        self.extract_every = extract_every
        self._messages_seen = 0
        self._facts: list[str] = []

    def on_message(self, message: Message) -> None:
        if message.role not in (MessageRole.user, MessageRole.assistant):
            return
        content = (message.content or "").strip()
        if not content:
            return
        self._messages_seen += 1
        if self._messages_seen % self.extract_every != 0:
            return
        new_facts = self._extract(content)
        for fact in new_facts:
            if fact and fact not in self._facts:
                self._facts.append(fact)
        # Enforce cap — drop the oldest facts first.
        if len(self._facts) > self.max_facts:
            self._facts = self._facts[-self.max_facts :]

    def _extract(self, content: str) -> list[str]:
        from fastaiagent.llm.message import UserMessage

        prompt = (
            "Extract standalone facts about the user or world state from the "
            "message below. Return a JSON array of short strings (at most 10 "
            "items, each under 120 characters). Return [] if no durable facts "
            "are present. Do not include opinions, questions, or tool chatter.\n\n"
            f"Message: {content}\n\nJSON:"
        )
        try:
            response = self.llm.complete([UserMessage(prompt)])
        except Exception as err:
            _log.warning("FactExtractionBlock LLM call failed: %s", err)
            return []
        text = (response.content or "").strip()
        if not text:
            return []
        # Tolerate code fences the LLM may wrap the JSON in.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(item).strip() for item in data if item and isinstance(item, str)]

    def render(self, query: str) -> list[Message]:
        if not self._facts:
            return []
        bullets = "\n".join(f"- {f}" for f in self._facts)
        return [SystemMessage(f"Known facts:\n{bullets}")]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._facts))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                self._facts = [str(x) for x in data]
        except json.JSONDecodeError:
            _log.warning("FactExtractionBlock.load: invalid JSON at %s", path)


# ---------------------------------------------------------------------------
# PersistentFactBlock
# ---------------------------------------------------------------------------


class PersistentFactBlock(MemoryBlock):
    """Inject durable facts loaded from the ``learned_memory`` table.

    Read-only at runtime — facts are produced offline by ``fastaiagent
    learn`` (the Trace Learning Loop) and re-injected here. ``on_message``
    is a no-op; ``render`` returns a single ``SystemMessage`` containing
    the active (non-superseded) facts for the configured scope.

    Pairs with :class:`FactExtractionBlock`, which extracts facts online
    during a single conversation. ``PersistentFactBlock`` carries facts
    *across* runs — the dreaming / continual-learning side of memory.

    Args:
        scope: ``"user"``, ``"project"``, or ``"agent"``.
        scope_id: identifier within the scope (agent name, user id, …).
            Empty string matches every scope_id within ``scope``.
        project_id: project to scope DB queries to. Defaults to "" which
            matches the unproject-scoped rows the SDK writes by default.
        max_facts: cap on facts injected per turn. Newest facts win.
        store: dependency injection for tests; defaults to a fresh
            :class:`fastaiagent.learn.MemoryStore` against the configured
            local.db.
        refresh_every: re-query the store every N renders. ``1`` (default)
            re-queries every turn; higher values cache for performance.
        recency_weight: in ``[0.0, 1.0]``. With ``recency_weight > 0`` the
            facts list is reordered so newer rows (by ``learned_memory.
            created_at``) outrank older ones. Default ``0.0`` preserves the
            existing newest-first behaviour from ``list_active``.
        importance_weight: in ``[0.0, 1.0]``. With ``importance_weight > 0``
            facts with higher ``confidence`` (the existing column on
            ``learned_memory``) outrank lower-confidence ones. Default
            ``0.0``.
        recency_half_life_seconds: time after which a fact's recency
            contribution halves. Default ``86400`` (one day) — facts decay
            slower than chat messages because they're meant to be durable.

    Example::

        from fastaiagent import Agent, AgentMemory, ComposableMemory
        from fastaiagent.agent.memory_blocks import PersistentFactBlock

        memory = ComposableMemory(
            primary=AgentMemory(),
            blocks=[PersistentFactBlock(scope="agent", scope_id="my-agent")],
        )
        agent = Agent(name="my-agent", system_prompt="...", llm=llm, memory=memory)
    """

    name = "persistent_facts"

    def __init__(
        self,
        scope: str = "agent",
        scope_id: str = "",
        project_id: str = "",
        max_facts: int = 50,
        store: object | None = None,
        refresh_every: int = 1,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
        recency_half_life_seconds: float = 86400.0,
    ):
        if scope not in ("user", "project", "agent"):
            raise ValueError(
                f"scope must be one of user|project|agent, got {scope!r}"
            )
        if max_facts < 1:
            raise ValueError("max_facts must be >= 1")
        if refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        if recency_weight < 0.0 or importance_weight < 0.0:
            raise ValueError("recency_weight and importance_weight must be >= 0")
        if recency_weight + importance_weight > 1.0:
            raise ValueError(
                "recency_weight + importance_weight must not exceed 1.0; "
                f"got {recency_weight + importance_weight:.3f}"
            )
        if recency_half_life_seconds <= 0.0:
            raise ValueError("recency_half_life_seconds must be > 0")
        self.scope = scope
        self.scope_id = scope_id
        self.project_id = project_id
        self.max_facts = max_facts
        self.refresh_every = refresh_every
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.recency_half_life_seconds = recency_half_life_seconds
        self._store = store  # may be None — lazy init in render()
        self._cached: list[str] | None = None
        self._renders_since_refresh = 0

    def on_message(self, message: Message) -> None:
        # Read-only block. The offline ``fastaiagent learn`` CLI is what
        # writes new facts.
        return

    def _resolve_store(self):
        if self._store is None:
            # Lazy import — avoids a hard dep on the learn module if a user
            # imports memory_blocks but never instantiates this block.
            from fastaiagent.learn.store import MemoryStore

            self._store = MemoryStore()
        return self._store

    def _refresh(self) -> list[str]:
        store = self._resolve_store()
        facts = store.list_active(
            scope=self.scope,  # type: ignore[arg-type]
            scope_id=self.scope_id,
            project_id=self.project_id,
            limit=self.max_facts,
        )
        # ``list_active`` returns newest first. With both scoring weights at
        # zero we preserve that order so existing callers see no change.
        if self.recency_weight == 0.0 and self.importance_weight == 0.0:
            return [f.fact for f in facts]

        # Otherwise rank by a weighted blend of recency (decayed against
        # ``created_at``) and importance (sourced from ``confidence``).
        # ``list_active``'s newest-first order is the implicit "similarity"
        # signal here — we treat it as 1.0 for all rows and let the two
        # explicit weights add the polish.
        sim_w = max(0.0, 1.0 - self.recency_weight - self.importance_weight)
        now = time.time()
        scored: list[tuple[float, str]] = []
        for f in facts:
            created = float(f.created_at) if f.created_at is not None else now
            age = max(0.0, now - created)
            recency = math.exp(-age / self.recency_half_life_seconds)
            importance = float(f.confidence) if f.confidence is not None else 1.0
            score = (
                sim_w * 1.0
                + self.recency_weight * recency
                + self.importance_weight * importance
            )
            scored.append((score, f.fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored]

    def render(self, query: str) -> list[Message]:
        if self._cached is None or self._renders_since_refresh >= self.refresh_every:
            try:
                self._cached = self._refresh()
            except Exception as err:
                _log.warning("PersistentFactBlock refresh failed: %s", err)
                self._cached = self._cached or []
            # Count this render as the first since the refresh, so the
            # next refresh fires on render N+refresh_every (not N+refresh_every+1).
            self._renders_since_refresh = 1
        else:
            self._renders_since_refresh += 1

        if not self._cached:
            return []
        bullets = "\n".join(f"- {fact}" for fact in self._cached)
        scope_label = f"{self.scope}:{self.scope_id}" if self.scope_id else self.scope
        return [
            SystemMessage(
                f"Learned facts ({scope_label}):\n{bullets}"
            )
        ]


# ---------------------------------------------------------------------------
# PlaneFactBlock (connected Enterprise plane — WS3 central governed memory)
# ---------------------------------------------------------------------------


class PlaneFactBlock(MemoryBlock):
    """Inject curated, human-approved facts read from a connected Enterprise plane.

    The plane's central learning loop extracts durable facts from already-ingested
    traces; a human curates/approves them; this block reads the approved facts back
    at the start of each turn via ``GET /public/v1/memory/facts`` and renders them
    as a system message. This is the read side of central governed memory (WS3).

    The sibling :class:`PersistentFactBlock` reads facts from the **local**
    ``learned_memory`` table; ``PlaneFactBlock`` reads the **governed/curated**
    facts the plane serves. Compose both to merge local + central knowledge.

    **Read-only and degradable.** ``on_message`` is a no-op. When the SDK is not
    connected, the plane is unreachable, the domain is not entitled (403), or any
    error occurs, ``render`` returns ``[]`` and the agent runs normally — central
    facts are an enhancement, never a dependency. The plane never runs agent code:
    it serves facts; recall + injection happen locally (CLAUDE §1). The read is a
    bounded start-of-run network GET (like :class:`VectorBlock`'s search), cached
    per ``refresh_every``; it never pushes anything (D5: central-extraction-only —
    there is no SDK fact-push path).

    Args:
        agent_id: the agent's id on the plane whose curated facts to read
            (required by the plane endpoint).
        category: optional category filter.
        max_facts: cap on facts injected per turn (1..200).
        query_conditioned: when True (default) pass the current user input as the
            plane's ``query`` for semantic recall; the plane falls back to a flat
            approved-facts list if its embedder is unavailable.
        score_threshold: minimum similarity for query-conditioned recall (0..1).
        refresh_every: re-read the plane every N renders. ``1`` (default) re-reads
            each turn; raise it to cache (the read is a bounded network GET, so
            higher values cut per-turn latency).
        timeout: per-request timeout (seconds) for the bounded read.

    Example::

        import fastaiagent as fa
        from fastaiagent import Agent, AgentMemory, ComposableMemory
        from fastaiagent.agent.memory_blocks import PlaneFactBlock

        fa.connect(api_key="fa-...", target="https://your-plane.example.com")
        memory = ComposableMemory(
            primary=AgentMemory(),
            blocks=[PlaneFactBlock(agent_id="my-agent-id")],
        )
        agent = Agent(name="support", system_prompt="...", llm=llm, memory=memory)
    """

    name = "plane_facts"

    def __init__(
        self,
        agent_id: str,
        *,
        category: str | None = None,
        max_facts: int = 50,
        query_conditioned: bool = True,
        score_threshold: float = 0.0,
        refresh_every: int = 1,
        timeout: float = 10.0,
    ):
        if not agent_id:
            raise ValueError("agent_id is required")
        if not 1 <= max_facts <= 200:
            raise ValueError("max_facts must be in 1..200")
        if refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in 0.0..1.0")
        self.agent_id = agent_id
        self.category = category
        self.max_facts = max_facts
        self.query_conditioned = query_conditioned
        self.score_threshold = score_threshold
        self.refresh_every = refresh_every
        self.timeout = timeout
        self._cached: list[str] | None = None
        self._renders_since_refresh = 0

    def on_message(self, message: Message) -> None:
        # Read-only block — facts are produced + curated centrally on the plane.
        return

    def _fetch_from_plane(self, query: str) -> list[str]:
        """GET the agent's approved facts from the plane; return their contents.

        A no-op (``[]``) when not connected; degradable on any non-2xx or error.
        """
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return []

        import httpx

        params: dict[str, Any] = {
            "agent_id": self.agent_id,
            "limit": self.max_facts,
            "score_threshold": self.score_threshold,
        }
        if self.category:
            params["category"] = self.category
        if self.query_conditioned and query:
            params["query"] = query

        url = f"{_connection.target}/public/v1/memory/facts"
        with httpx.Client(timeout=self.timeout, verify=True) as client:
            resp = client.get(url, params=params, headers=_connection.headers)
        if resp.status_code != 200:
            # 403 (domain not entitled) / 404 (unknown agent) / 5xx → degrade to
            # no facts. The agent still runs; central memory is an enhancement.
            if resp.status_code not in (403, 404):
                _log.warning("PlaneFactBlock read got HTTP %d", resp.status_code)
            return []
        data = resp.json()
        return [
            f["content"]
            for f in data.get("facts", [])
            if isinstance(f, dict) and f.get("content")
        ]

    def render(self, query: str) -> list[Message]:
        if self._cached is None or self._renders_since_refresh >= self.refresh_every:
            try:
                self._cached = self._fetch_from_plane(query)
            except Exception as err:
                # Never let a plane read break the run — degrade to last-known / empty.
                _log.warning("PlaneFactBlock refresh failed: %s", err)
                self._cached = self._cached or []
            self._renders_since_refresh = 1
        else:
            self._renders_since_refresh += 1

        if not self._cached:
            return []
        bullets = "\n".join(f"- {fact}" for fact in self._cached)
        return [SystemMessage(f"Curated facts (agent:{self.agent_id}):\n{bullets}")]
