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
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

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
    :meth:`render`, the query is embedded and the top-``top_k`` most similar
    past messages are returned as SystemMessage fragments.

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
    """

    name = "vector"

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        top_k: int = 5,
        namespace: str = "default",
        min_content_chars: int = 10,
    ):
        from fastaiagent.kb.embedding import get_default_embedder

        self.store = store
        self.embedder: Embedder = embedder or get_default_embedder()
        self.top_k = top_k
        self.namespace = namespace
        self.min_content_chars = min_content_chars

    def _make_chunk(self, message: Message) -> Chunk | None:
        from fastaiagent.kb.chunking import Chunk

        content = (message.content or "").strip()
        if len(content) < self.min_content_chars:
            return None
        text = f"[{message.role.value}] {content}"
        return Chunk(
            id=str(uuid.uuid4()),
            content=text,
            metadata={"namespace": self.namespace, "role": message.role.value},
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
        relevant = [
            c for c, _ in hits
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
