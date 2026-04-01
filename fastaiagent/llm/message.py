"""Message types for LLM conversations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ToolCallFunction(BaseModel):
    """Function call within a tool call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class Message(BaseModel):
    """A message in a conversation."""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message dict."""
        msg: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            msg["content"] = self.content
        if self.name is not None:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai_format() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> Message:
        """Create from an OpenAI-compatible message dict."""
        import json

        tool_calls = None
        if raw_tc := data.get("tool_calls"):
            tool_calls = []
            for tc in raw_tc:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                tool_calls.append(ToolCall(id=tc["id"], name=func["name"], arguments=args))
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            name=data.get("name"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
        )


# --- Factory functions ---


def SystemMessage(content: str) -> Message:  # noqa: N802
    """Create a system message."""
    return Message(role=MessageRole.system, content=content)


def UserMessage(content: str) -> Message:  # noqa: N802
    """Create a user message."""
    return Message(role=MessageRole.user, content=content)


def AssistantMessage(  # noqa: N802
    content: str | None = None, tool_calls: list[ToolCall] | None = None
) -> Message:
    """Create an assistant message."""
    return Message(role=MessageRole.assistant, content=content, tool_calls=tool_calls)


def ToolMessage(content: str, tool_call_id: str) -> Message:  # noqa: N802
    """Create a tool result message."""
    return Message(role=MessageRole.tool, content=content, tool_call_id=tool_call_id)
