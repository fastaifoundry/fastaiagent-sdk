"""Memory blocks — composable long-term memory primitives.

A ``MemoryBlock`` observes every message passing through an agent
(``on_message``) and renders fragments back into the next prompt
(``render``). Compose any number of blocks with
:class:`fastaiagent.agent.memory.ComposableMemory` to augment an agent's
sliding-window history with persistent facts, summaries, and semantic
recall over past conversations.

Block types that ship:

- :class:`StaticBlock`          — a fixed system-level fact, injected every turn
- :class:`FewShotBlock`         — few-shot input/output exemplars (optimize's KB lever)
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
import re
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastaiagent.llm.message import Message, MessageRole, SystemMessage

if TYPE_CHECKING:
    from fastaiagent.kb.chunking import Chunk
    from fastaiagent.kb.embedding import Embedder
    from fastaiagent.kb.protocols import VectorStore
    from fastaiagent.llm.client import LLMClient

_log = logging.getLogger(__name__)


class MemoryIsolationError(RuntimeError):
    """Raised by :meth:`MemoryBlock.isolated_copy` for blocks that can't be
    isolated per candidate — currently :class:`VectorBlock`, which upserts to an
    external ``VectorStore`` during a run, so sharing the handle would bleed one
    candidate's turns into another's retrieval. Callers (e.g.
    ``fastaiagent.optimize``) catch this to refuse, or opt in to sharing.
    """


__all__ = [
    "BlockRenderReport",
    "BlockWriteReport",
    "FactExtractionBlock",
    "FewShotBlock",
    "MemoryBlock",
    "MemoryIsolationError",
    "PersistentFactBlock",
    "PlaneFactBlock",
    "SharedMemoryContext",
    "StaticBlock",
    "SummaryBlock",
    "VectorBlock",
]

# Bounded length for recalled-content snippets captured in trace spans. Mirrors
# the KB span's "reference, not full content" discipline — memory has no backing
# store to rehydrate from, so we keep a short, payload-gated, maskable preview.
_SNIPPET_MAX_CHARS = 200


def _snippet(text: str) -> str:
    """Return a length-capped single-line preview of ``text`` for tracing."""
    flat = " ".join((text or "").split())
    return flat[:_SNIPPET_MAX_CHARS]


def _strip_role_tag(text: str) -> str:
    """Drop a leading ``[role] `` tag VectorBlock stamps on stored messages."""
    return re.sub(r"^\[[^\]]*\]\s*", "", text or "")


def _norm_for_dedupe(text: str) -> str:
    """Lowercase + collapse whitespace so upstream-dedupe matching is stable."""
    return " ".join((text or "").lower().split())


@dataclass
class BlockRenderReport:
    """What a block contributed to a single ``render()`` — captured for tracing.

    Blocks populate this during ``render()`` and expose it via
    :meth:`MemoryBlock.last_render_report`. The tracing layer reads it without
    reflecting on a block's private state. All content fields are bounded
    snippets, gated + masked downstream.
    """

    block_name: str
    block_type: str
    rendered_count: int = 0
    # VectorBlock-only: per-item final scores, rank order (highest first).
    scores: list[float] | None = None
    # Bounded, length-capped previews of the recalled/injected items.
    snippets: list[str] | None = None
    # Number of items dropped because an upstream block already surfaced them
    # (VectorBlock with ``dedupe_against_upstream=True``). ``None`` = not applicable.
    deduped_count: int | None = None


@dataclass
class BlockWriteReport:
    """What a block did on a single ``on_message()`` — captured for tracing."""

    block_name: str
    block_type: str
    action: str = "noop"  # "stored" | "summarized" | "extracted_facts" | "embedded" | "noop"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedMemoryContext:
    """One-directional pipe passed to :meth:`MemoryBlock.render_with_context`.

    ``ComposableMemory`` renders blocks in declaration order and, before each
    block, hands it what the *earlier* blocks already produced this turn — so a
    later block can avoid repeating content or condition on it. Only block
    output flows through here; the raw primary window is appended afterwards and
    is not shared.
    """

    query: str
    # (block_name, rendered messages) in render order, populated as we go.
    rendered: list[tuple[str, list[Message]]] = field(default_factory=list)

    def upstream_messages(self) -> list[Message]:
        """All messages produced by earlier blocks this turn, in order."""
        out: list[Message] = []
        for _name, msgs in self.rendered:
            out.extend(msgs)
        return out

    def upstream_text(self) -> str:
        """Concatenated text content of all earlier blocks' output."""
        return "\n".join(m.content or "" for m in self.upstream_messages())

    def by_block(self, name: str) -> list[Message]:
        """Messages produced by a specific earlier block (by ``name``)."""
        out: list[Message] = []
        for n, msgs in self.rendered:
            if n == name:
                out.extend(msgs)
        return out


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

    def render_with_context(
        self, query: str, shared: SharedMemoryContext
    ) -> list[Message]:
        """Render, optionally reading what earlier blocks produced this turn.

        ``ComposableMemory`` calls this (not :meth:`render`) and passes a
        :class:`SharedMemoryContext` holding the output of blocks that ran
        earlier in the same read pass. The default **ignores** ``shared`` and
        delegates to :meth:`render`, so existing and third-party blocks need no
        changes. Override to condition on / dedupe against upstream output.
        """
        return self.render(query)

    def isolated_copy(self) -> MemoryBlock:
        """Return a copy safe to use in a single isolated evaluation.

        Used by ``fastaiagent.optimize`` to give every candidate its own memory
        so one candidate's turns never bleed into another's. The contract is
        **share the external handle, reset in-process state**: a block holding an
        ``llm`` / store handle passes it through (deepcopy would duplicate live
        handles) but starts with empty per-conversation state.

        This concrete default **warns and shares ``self``** so existing and
        third-party subclasses keep working without edits — but stateful blocks
        should override it (the shipped blocks do). A block that writes to an
        *external* store during a run should raise :class:`MemoryIsolationError`
        instead (see :class:`VectorBlock`).
        """
        warnings.warn(
            f"{type(self).__name__} has no isolated_copy() override; it is shared "
            "across candidate evaluations, which may bleed state. Override "
            "isolated_copy() to share external handles but reset in-process state.",
            stacklevel=2,
        )
        return self

    def save(self, path: Path) -> None:
        """Persist block state. Default: no-op."""

    def load(self, path: Path) -> None:
        """Load block state. Default: no-op."""

    def last_render_report(self) -> BlockRenderReport | None:
        """Describe what the most recent :meth:`render` contributed, for tracing.

        Optional. The default returns ``None`` so custom/third-party blocks need
        no changes and the tracing layer never reflects on private state. Shipped
        blocks override this to surface counts, scores, and bounded snippets.
        """
        return None

    def last_write_report(self) -> BlockWriteReport | None:
        """Describe what the most recent :meth:`on_message` did, for tracing.

        Optional — defaults to ``None`` (see :meth:`last_render_report`).
        """
        return None


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

    def isolated_copy(self) -> MemoryBlock:
        # Stateless — a fresh instance with the same text is fully isolated.
        return StaticBlock(self.text, self.name)

    def last_render_report(self) -> BlockRenderReport | None:
        if not self.text:
            return BlockRenderReport(self.name, type(self).__name__, rendered_count=0)
        return BlockRenderReport(
            self.name, type(self).__name__, rendered_count=1, snippets=[_snippet(self.text)]
        )

    def last_write_report(self) -> BlockWriteReport | None:
        return BlockWriteReport(self.name, type(self).__name__, action="noop")


# ---------------------------------------------------------------------------
# FewShotBlock
# ---------------------------------------------------------------------------


class FewShotBlock(MemoryBlock):
    """Inject few-shot input/output exemplars as a single SystemMessage.

    Mirrors :class:`StaticBlock` (read-only, stateless) but renders a list of
    ``{"input", "output"}`` demos as worked examples ahead of the conversation —
    a shape :class:`PersistentFactBlock` (bullet-list of facts) can't express.
    Produced by ``fastaiagent.optimize``'s few-shot lever (``bootstrap_demos``).

    Example::

        FewShotBlock([{"input": "Capital of France?", "output": "Paris"}])
    """

    def __init__(
        self,
        demos: list[dict[str, Any]],
        name: str = "fewshot",
        header: str = "Here are examples of good responses:",
    ):
        self.demos = list(demos)
        self.name = name
        self.header = header

    def on_message(self, message: Message) -> None:
        return

    def render(self, query: str) -> list[Message]:
        if not self.demos:
            return []
        body = "\n\n".join(
            f"Input: {d.get('input', '')}\nResponse: {d.get('output', '')}" for d in self.demos
        )
        return [SystemMessage(f"{self.header}\n\n{body}")]

    def isolated_copy(self) -> MemoryBlock:
        # Read-only demos — a fresh instance is fully isolated.
        return FewShotBlock(self.demos, name=self.name, header=self.header)

    def last_render_report(self) -> BlockRenderReport | None:
        if not self.demos:
            return BlockRenderReport(self.name, type(self).__name__, rendered_count=0)
        return BlockRenderReport(
            self.name,
            type(self).__name__,
            rendered_count=len(self.demos),
            snippets=[
                _snippet(f"{d.get('input', '')} -> {d.get('output', '')}") for d in self.demos
            ],
        )

    def last_write_report(self) -> BlockWriteReport | None:
        return BlockWriteReport(self.name, type(self).__name__, action="noop")


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
        self._last_write: BlockWriteReport | None = None

    def isolated_copy(self) -> MemoryBlock:
        # Share the llm handle; reset in-process summary state per candidate.
        return SummaryBlock(
            llm=self.llm,
            keep_last=self.keep_last,
            summarize_every=self.summarize_every,
            max_chars=self.max_chars,
        )

    def on_message(self, message: Message) -> None:
        self._archive.append(message)
        self._messages_seen += 1
        self._last_write = BlockWriteReport(self.name, type(self).__name__, action="stored")
        # Refresh summary when we cross a threshold AND there is something
        # older than keep_last to summarize.
        if self._messages_seen < self.summarize_every:
            return
        if self._messages_seen % self.summarize_every != 0:
            return
        if len(self._archive) <= self.keep_last:
            return
        self._refresh_summary()
        self._last_write = BlockWriteReport(
            self.name,
            type(self).__name__,
            action="summarized",
            detail={"summary_triggered": True, "summary_chars": len(self._summary)},
        )

    def last_write_report(self) -> BlockWriteReport | None:
        return self._last_write

    def last_render_report(self) -> BlockRenderReport | None:
        if not self._summary:
            return BlockRenderReport(self.name, type(self).__name__, rendered_count=0)
        return BlockRenderReport(
            self.name,
            type(self).__name__,
            rendered_count=1,
            snippets=[_snippet(self._summary)],
        )

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
        dedupe_against_upstream: When ``True`` and this block runs inside a
            :class:`ComposableMemory` after other blocks, drop any recalled
            message whose content is already present in an earlier block's
            output this turn (via :class:`SharedMemoryContext`). Saves prompt
            tokens by not re-recalling what a ``StaticBlock`` / fact block
            already injected. Default ``False`` (no change).
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
        dedupe_against_upstream: bool = False,
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
        self.dedupe_against_upstream = dedupe_against_upstream
        self._last_render: BlockRenderReport | None = None
        self._last_write: BlockWriteReport | None = None

    def isolated_copy(self) -> MemoryBlock:
        # VectorBlock upserts to its external store in on_message, so neither
        # sharing the handle (candidates bleed) nor deep-copying the store
        # (cost / un-clonable remote stores) is safe. Refuse — the optimize
        # layer turns this into a clear error or an explicit opt-in.
        raise MemoryIsolationError(
            "VectorBlock writes to an external VectorStore during a run, so it "
            "can't be isolated per candidate. Remove it to optimize, or pass "
            "allow_writable_memory=True to share the store (accepting that "
            "candidate runs write to it)."
        )

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
            self._last_write = BlockWriteReport(self.name, type(self).__name__, action="noop")
            return
        try:
            embedding = self.embedder.embed([chunk.content])[0]
            self.store.add([chunk], [embedding])
            self._last_write = BlockWriteReport(
                self.name, type(self).__name__, action="embedded"
            )
        except Exception as err:
            _log.warning("VectorBlock failed to index message: %s", err)
            self._last_write = BlockWriteReport(self.name, type(self).__name__, action="noop")

    def _score_hits(self, hits: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
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

    def _retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Embed the query, search, score, and namespace-filter — no rendering.

        Returns ``(chunk, final_score)`` pairs in rank order. Degrades to ``[]``
        on any embed/search error (the agent runs without recall).
        """
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
        # Keep the (chunk, score) pairs through the namespace filter so the
        # render report can surface per-item scores without recomputation.
        return [
            (c, s)
            for c, s in scored
            if c.metadata.get("namespace", self.namespace) == self.namespace
        ]

    def _build(
        self, relevant: list[tuple[Chunk, float]], deduped_count: int | None = None
    ) -> list[Message]:
        """Turn scored chunks into a SystemMessage + record the render report."""
        self._last_render = BlockRenderReport(
            self.name,
            type(self).__name__,
            rendered_count=len(relevant),
            scores=[round(float(s), 6) for _, s in relevant] or None,
            snippets=[_snippet(c.content) for c, _ in relevant] or None,
            deduped_count=deduped_count,
        )
        if not relevant:
            return []
        body = "\n".join(f"- {c.content}" for c, _ in relevant)
        return [SystemMessage(f"Relevant prior exchanges:\n{body}")]

    def render(self, query: str) -> list[Message]:
        return self._build(self._retrieve(query))

    def render_with_context(
        self, query: str, shared: SharedMemoryContext
    ) -> list[Message]:
        relevant = self._retrieve(query)
        if not self.dedupe_against_upstream or not relevant:
            return self._build(relevant)
        upstream = _norm_for_dedupe(shared.upstream_text())
        kept: list[tuple[Chunk, float]] = []
        dropped = 0
        for chunk, score in relevant:
            needle = _norm_for_dedupe(_strip_role_tag(chunk.content))
            if needle and needle in upstream:
                dropped += 1
                continue
            kept.append((chunk, score))
        return self._build(kept, deduped_count=dropped)

    def last_render_report(self) -> BlockRenderReport | None:
        return self._last_render

    def last_write_report(self) -> BlockWriteReport | None:
        return self._last_write


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
        persist: When ``True``, newly extracted facts are also written to the
            durable ``learned_memory`` table during the run — so they survive
            across runs and can be read back by :class:`PersistentFactBlock`.
            Each persisted fact is stamped with the current trace id as
            ``source_trace_id`` (lineage). Default ``False`` (in-conversation
            only — today's behaviour, unchanged).
        scope: scope for persisted facts (``user`` / ``project`` / ``agent``).
        scope_id: identifier within the scope. **Required when ``persist=True``**
            — an empty scope_id write is ambiguous.
        project_id: project partition for persisted facts (default ``""``).
        confidence: confidence stamped on auto-persisted facts. Default ``0.6``
            (below curated ``1.0``) so machine-extracted facts sort below and
            are visibly distinguishable from human-approved ones.
        store: dependency injection for tests; defaults to a
            :class:`fastaiagent.learn.MemoryStore` against the configured local.db.
    """

    name = "facts"

    def __init__(
        self,
        llm: LLMClient,
        max_facts: int = 200,
        extract_every: int = 1,
        persist: bool = False,
        scope: str = "agent",
        scope_id: str = "",
        project_id: str = "",
        confidence: float = 0.6,
        store: object | None = None,
    ):
        if max_facts < 1:
            raise ValueError("max_facts must be >= 1")
        if extract_every < 1:
            raise ValueError("extract_every must be >= 1")
        if persist:
            if scope not in ("user", "project", "agent"):
                raise ValueError(f"scope must be one of user|project|agent, got {scope!r}")
            if not scope_id:
                raise ValueError("scope_id is required when persist=True")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in 0.0..1.0")
        self.llm = llm
        self.max_facts = max_facts
        self.extract_every = extract_every
        self.persist = persist
        self.scope = scope
        self.scope_id = scope_id
        self.project_id = project_id
        self.confidence = confidence
        self._store = store  # may be None — lazy init in _persist_facts
        self._messages_seen = 0
        self._facts: list[str] = []
        self._last_write: BlockWriteReport | None = None

    def isolated_copy(self) -> MemoryBlock:
        # With persist on, this writes an external store during a run — like
        # VectorBlock, that can't be isolated per candidate without bleeding
        # writes across candidates. Refuse (optimize turns this into a clear
        # error or an explicit opt-in).
        if self.persist:
            raise MemoryIsolationError(
                "FactExtractionBlock(persist=True) writes to the learned_memory "
                "store during a run, so it can't be isolated per candidate. Set "
                "persist=False to optimize, or pass allow_writable_memory=True."
            )
        # Share the llm handle; reset extracted-facts state per candidate.
        return FactExtractionBlock(
            llm=self.llm, max_facts=self.max_facts, extract_every=self.extract_every
        )

    def on_message(self, message: Message) -> None:
        self._last_write = BlockWriteReport(self.name, type(self).__name__, action="noop")
        if message.role not in (MessageRole.user, MessageRole.assistant):
            return
        content = (message.content or "").strip()
        if not content:
            return
        self._messages_seen += 1
        if self._messages_seen % self.extract_every != 0:
            return
        new_facts = self._extract(content)
        added_facts: list[str] = []
        for fact in new_facts:
            if fact and fact not in self._facts:
                self._facts.append(fact)
                added_facts.append(fact)
        # Enforce cap — drop the oldest facts first.
        if len(self._facts) > self.max_facts:
            self._facts = self._facts[-self.max_facts :]
        detail: dict[str, Any] = {
            "facts_extracted": len(added_facts),
            "total_facts": len(self._facts),
        }
        if self.persist and added_facts:
            detail["persisted"] = self._persist_facts(added_facts)
        self._last_write = BlockWriteReport(
            self.name,
            type(self).__name__,
            action="extracted_facts",
            detail=detail,
        )

    def _resolve_store(self):
        if self._store is None:
            from fastaiagent.learn.store import MemoryStore

            self._store = MemoryStore()
        return self._store

    @staticmethod
    def _current_trace_id() -> str | None:
        """Hex trace id of the active span, so persisted facts link to the run."""
        try:
            from opentelemetry import trace as _otel_trace

            ctx = _otel_trace.get_current_span().get_span_context()
            if ctx and ctx.trace_id:
                return format(ctx.trace_id, "032x")
        except Exception:
            return None
        return None

    def _persist_facts(self, facts: list[str]) -> int:
        """Write newly-extracted facts to ``learned_memory``. Returns count added.

        Idempotent (the store's UNIQUE constraint dedupes). Never raises — a
        persist failure logs and the run continues, matching block isolation.
        """
        from fastaiagent.learn import Fact

        trace_id = self._current_trace_id()
        written = 0
        try:
            store = self._resolve_store()
            for fact in facts:
                store.add(
                    Fact(
                        scope=self.scope,  # type: ignore[arg-type]
                        scope_id=self.scope_id,
                        fact=fact,
                        source_trace_id=trace_id,
                        confidence=self.confidence,
                        project_id=self.project_id,
                    )
                )
                written += 1
        except Exception as err:
            _log.warning("FactExtractionBlock persist failed: %s", err)
        return written

    def last_write_report(self) -> BlockWriteReport | None:
        return self._last_write

    def last_render_report(self) -> BlockRenderReport | None:
        if not self._facts:
            return BlockRenderReport(self.name, type(self).__name__, rendered_count=0)
        return BlockRenderReport(
            self.name,
            type(self).__name__,
            rendered_count=len(self._facts),
            snippets=[_snippet(f) for f in self._facts],
        )

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
            raise ValueError(f"scope must be one of user|project|agent, got {scope!r}")
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

    def isolated_copy(self) -> MemoryBlock:
        # Read-only at run time (render() only calls list_active). Share the
        # store handle; reset the per-instance cache.
        return PersistentFactBlock(
            scope=self.scope,
            scope_id=self.scope_id,
            project_id=self.project_id,
            max_facts=self.max_facts,
            store=self._store,
            refresh_every=self.refresh_every,
            recency_weight=self.recency_weight,
            importance_weight=self.importance_weight,
            recency_half_life_seconds=self.recency_half_life_seconds,
        )

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
                sim_w * 1.0 + self.recency_weight * recency + self.importance_weight * importance
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
        return [SystemMessage(f"Learned facts ({scope_label}):\n{bullets}")]

    def last_render_report(self) -> BlockRenderReport | None:
        cached = self._cached or []
        return BlockRenderReport(
            self.name,
            type(self).__name__,
            rendered_count=len(cached),
            snippets=[_snippet(f) for f in cached] or None,
        )

    def last_write_report(self) -> BlockWriteReport | None:
        return BlockWriteReport(self.name, type(self).__name__, action="noop")


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

    def isolated_copy(self) -> MemoryBlock:
        # Read-only at run time (render() only GETs from the plane). Share
        # config; reset the per-instance cache.
        return PlaneFactBlock(
            self.agent_id,
            category=self.category,
            max_facts=self.max_facts,
            query_conditioned=self.query_conditioned,
            score_threshold=self.score_threshold,
            refresh_every=self.refresh_every,
            timeout=self.timeout,
        )

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
            f["content"] for f in data.get("facts", []) if isinstance(f, dict) and f.get("content")
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

    def last_render_report(self) -> BlockRenderReport | None:
        cached = self._cached or []
        return BlockRenderReport(
            self.name,
            type(self).__name__,
            rendered_count=len(cached),
            snippets=[_snippet(f) for f in cached] or None,
        )

    def last_write_report(self) -> BlockWriteReport | None:
        return BlockWriteReport(self.name, type(self).__name__, action="noop")
