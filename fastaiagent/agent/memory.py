"""Agent conversation memory.

Two public classes:

- :class:`AgentMemory`       — simple sliding-window message history (unchanged
  since 0.1.x; the default memory for ``Agent``).
- :class:`ComposableMemory`  — a ``AgentMemory`` augmented with one or more
  :class:`fastaiagent.agent.memory_blocks.MemoryBlock` instances that render
  SystemMessage fragments (static facts, summaries, semantic recall, extracted
  facts, …) ahead of the primary history. Shipped in 0.4.0.

Both expose the same public surface — ``add``, ``get_context``, ``clear``,
``save``, ``load``, ``messages``, ``__len__``, ``__bool__`` — so
``Agent(memory=...)`` accepts either without code changes.

``get_context`` takes an optional ``query`` argument. ``AgentMemory`` ignores
it (plain sliding window). ``ComposableMemory`` passes it to each block's
``render(query)`` method — useful for query-conditioned blocks like
:class:`fastaiagent.agent.memory_blocks.VectorBlock`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastaiagent.llm.message import Message

if TYPE_CHECKING:
    from fastaiagent.agent.memory_blocks import MemoryBlock


class AgentMemory:
    """Manages conversation history for an agent.

    Supports token-window truncation and persistence.
    """

    def __init__(self, max_messages: int | None = None):
        self._messages: list[Message] = []
        self.max_messages = max_messages

    def add(self, message: Message) -> None:
        """Add a message to memory."""
        self._messages.append(message)
        if self.max_messages and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def get_context(
        self,
        query: str = "",
        max_messages: int | None = None,
    ) -> list[Message]:
        """Get conversation history, optionally truncated.

        ``query`` is accepted for signature compatibility with
        :class:`ComposableMemory` and is ignored by this simple implementation.
        """
        limit = max_messages or self.max_messages
        if limit:
            return list(self._messages[-limit:])
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def save(self, path: str | Path) -> None:
        """Save memory to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.model_dump(mode="json") for m in self._messages]
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load memory from a JSON file."""
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self._messages = [Message.model_validate(m) for m in data]

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return True  # Memory object is always truthy, even when empty


class ComposableMemory:
    """A sliding-window primary memory augmented with long-term memory blocks.

    Every ``add(msg)`` call is broadcast to each block via ``block.on_message``.
    Every ``get_context(query)`` call renders each block in declaration order
    (yielding SystemMessage fragments that pin facts, summaries, recalled
    snippets) and then appends the primary sliding window.

    Example::

        from fastaiagent import Agent
        from fastaiagent.agent.memory import ComposableMemory, AgentMemory
        from fastaiagent.agent.memory_blocks import (
            StaticBlock, SummaryBlock, VectorBlock,
        )

        memory = ComposableMemory(
            blocks=[
                StaticBlock("The user is Upendra. They prefer terse answers."),
                SummaryBlock(llm=llm, keep_last=10, summarize_every=5),
                VectorBlock(store=vector_store, top_k=5),
            ],
            primary=AgentMemory(max_messages=20),
        )

        agent = Agent(name="assistant", llm=llm, memory=memory)

    Backward compat: if you were on 0.3.x with ``AgentMemory``, that code
    keeps working — ``ComposableMemory`` is purely additive.
    """

    def __init__(
        self,
        blocks: list[MemoryBlock] | None = None,
        primary: AgentMemory | None = None,
    ):
        self.blocks: list[MemoryBlock] = list(blocks) if blocks else []
        self.primary: AgentMemory = primary or AgentMemory(max_messages=20)

    def add(self, message: Message) -> None:
        """Add a message to the primary window and broadcast to all blocks."""
        self.primary.add(message)
        for block in self.blocks:
            try:
                block.on_message(message)
            except Exception:
                # Individual block failures must not break the run.
                import logging

                logging.getLogger(__name__).warning(
                    "Memory block %r raised in on_message; skipping.",
                    getattr(block, "name", type(block).__name__),
                    exc_info=True,
                )

    def get_context(
        self,
        query: str = "",
        max_messages: int | None = None,
    ) -> list[Message]:
        """Render each block, then append the primary sliding window.

        Block output comes first — the SystemMessage fragments act as pinned
        context the LLM sees before the literal conversation history.
        """
        out: list[Message] = []
        for block in self.blocks:
            try:
                out.extend(block.render(query))
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Memory block %r raised in render; skipping.",
                    getattr(block, "name", type(block).__name__),
                    exc_info=True,
                )
        out.extend(self.primary.get_context(max_messages=max_messages))
        return out

    def clear(self) -> None:
        """Clear the primary window. Blocks are NOT reset — call ``reset_blocks()``."""
        self.primary.clear()

    def reset_blocks(self) -> None:
        """Force every block back to empty state, preserving ``self.blocks``.

        Useful between test cases or when starting a new conversation where
        prior facts and summaries should not leak across sessions.
        """
        for block in self.blocks:
            # Blocks with mutable state should implement a ``reset`` method;
            # we call it best-effort. Blocks without it are left as-is.
            fn = getattr(block, "reset", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Memory block %r raised in reset; skipping.",
                        getattr(block, "name", type(block).__name__),
                        exc_info=True,
                    )

    def save(self, path: str | Path) -> None:
        """Persist the primary window and every block under ``path/``.

        ``path`` is treated as a directory. Primary messages go to
        ``path/primary.json``; each block saves to ``path/blocks/{name}.json``.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.primary.save(root / "primary.json")
        for block in self.blocks:
            try:
                block.save(root / "blocks" / f"{block.name or type(block).__name__}.json")
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Memory block %r failed to save; skipping.",
                    block.name,
                    exc_info=True,
                )

    def load(self, path: str | Path) -> None:
        """Restore the primary window and each block from disk.

        The caller must have constructed the same set of blocks (in the same
        order) — blocks are matched to files by ``block.name``. Unknown files
        are ignored; missing files leave the corresponding block at its
        default state.
        """
        root = Path(path)
        primary_path = root / "primary.json"
        if primary_path.exists():
            self.primary.load(primary_path)
        blocks_dir = root / "blocks"
        if not blocks_dir.exists():
            return
        for block in self.blocks:
            file = blocks_dir / f"{block.name or type(block).__name__}.json"
            if file.exists():
                try:
                    block.load(file)
                except Exception:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Memory block %r failed to load from %s; skipping.",
                        block.name,
                        file,
                        exc_info=True,
                    )

    @property
    def messages(self) -> list[Message]:
        """The primary window's messages (not including block-rendered output)."""
        return self.primary.messages

    def __len__(self) -> int:
        return len(self.primary)

    def __bool__(self) -> bool:
        return True
