"""Message types for LLM conversations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from fastaiagent.multimodal.types import ContentPart


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
    """A message in a conversation.

    ``content`` is normally a ``str``. For multimodal user/tool messages it
    can be a ``list[ContentPart]`` (mix of strings, ``Image``, ``PDF``);
    provider-specific wire formatting then happens in
    :py:meth:`to_provider_dict`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: MessageRole
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def has_multimodal_content(self) -> bool:
        """Return ``True`` when ``content`` is a list of parts."""
        return isinstance(self.content, list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message dict.

        For string content this produces the legacy shape. For list content
        this delegates to :py:meth:`to_provider_dict` with ``provider="openai"``,
        which is the same shape OpenAI/Azure/Custom expect for multimodal.
        """
        if self.has_multimodal_content():
            return self.to_provider_dict("openai")
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

    def to_provider_dict(
        self,
        provider: str,
        *,
        model: str = "",
        pdf_mode: str = "auto",
        is_vision_capable: bool = True,
        max_pdf_pages: int = 20,
        max_image_size_mb: float | None = None,
    ) -> dict[str, Any]:
        """Convert to a provider-specific message dict.

        For string ``content`` this returns the legacy OpenAI-compat shape
        (with provider-specific tweaks where unavoidable). For list
        ``content`` this routes through
        :py:func:`fastaiagent.multimodal.format_multimodal_message`.
        """
        msg: dict[str, Any] = {"role": self.role.value}
        if self.name is not None:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai_format() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id

        if not self.has_multimodal_content():
            if self.content is not None:
                msg["content"] = self.content
            return msg

        from fastaiagent.multimodal.format import format_multimodal_message

        parts: list[ContentPart] = list(self.content or [])
        formatted = format_multimodal_message(
            parts,
            provider,
            model=model,
            pdf_mode=pdf_mode,
            is_vision_capable=is_vision_capable,
            max_pdf_pages=max_pdf_pages,
            max_image_size_mb=max_image_size_mb,
        )
        msg.update(formatted)
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


def UserMessage(content: str | list[Any]) -> Message:  # noqa: N802
    """Create a user message.

    ``content`` may be a string (legacy / text-only) or a list of
    ``ContentPart`` (text + Image + PDF) for multimodal calls.
    """
    return Message(role=MessageRole.user, content=content)


def AssistantMessage(  # noqa: N802
    content: str | None = None, tool_calls: list[ToolCall] | None = None
) -> Message:
    """Create an assistant message."""
    return Message(role=MessageRole.assistant, content=content, tool_calls=tool_calls)


def ToolMessage(content: str | list[Any], tool_call_id: str) -> Message:  # noqa: N802
    """Create a tool result message.

    ``content`` may be a string or a list of ``ContentPart`` when the tool
    returns multimodal output (e.g. a screenshot ``Image``).
    """
    return Message(role=MessageRole.tool, content=content, tool_call_id=tool_call_id)
