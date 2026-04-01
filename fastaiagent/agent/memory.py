"""Agent conversation memory."""

from __future__ import annotations

import json
from pathlib import Path

from fastaiagent.llm.message import Message


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

    def get_context(self, max_messages: int | None = None) -> list[Message]:
        """Get conversation history, optionally truncated."""
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
