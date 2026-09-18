"""Message types for LLM conversations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from fastaiagent.multimodal.types import ContentPart


def _summarize_parts(parts: list[Any]) -> list[dict[str, Any]]:
    """Compact, render-free summary of multimodal parts for telemetry/logs.

    Keeps text verbatim; represents ``Image``/``PDF`` by type and size only —
    never their bytes. Deliberately avoids the PDF engine so serializing a message
    for a span can't fail on an unparseable PDF or bloat the span with base64.
    """
    from fastaiagent.multimodal.file import File
    from fastaiagent.multimodal.image import Image
    from fastaiagent.multimodal.pdf import PDF

    summary: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            summary.append({"type": "text", "text": part})
        elif isinstance(part, Image):
            summary.append(
                {
                    "type": "image",
                    "media_type": part.media_type,
                    "size_bytes": len(part.data),
                    "source_url": part.source_url,
                }
            )
        elif isinstance(part, PDF):
            summary.append(
                {
                    "type": "pdf",
                    "size_bytes": len(part.data),
                    "source_path": part.source_path,
                    "source_url": part.source_url,
                }
            )
        elif isinstance(part, File):
            summary.append(
                {
                    "type": "file",
                    "mime_type": part.mime_type,
                    "size_bytes": len(part.data),
                    "filename": part.filename,
                    "file_id": part.file_id,
                }
            )
        else:
            summary.append({"type": "text", "text": str(part)})
    return summary


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
        """Convert to a compact dict for logging / telemetry / test assertions.

        For string content this produces the legacy OpenAI shape. For list
        (multimodal) content it emits a lightweight **summary** of each part —
        NOT the provider wire format. This path must never render or base64
        media: it is called to serialize request messages onto OTel spans, and
        going through :py:meth:`to_provider_dict` there would base64 every image
        and run page-rendering for PDFs (expensive, it needs the optional PDF
        engine, and it raises on PDFs that engine can't decompress) purely to
        build a log line. Actual
        provider requests are built by :py:meth:`to_provider_dict` with the real
        provider and ``pdf_mode``.
        """
        if self.has_multimodal_content():
            return {"role": self.role.value, "content": _summarize_parts(self.content or [])}
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


# --- The tool-call / tool-result invariant ---------------------------------
#
# Both OpenAI and Anthropic reject a request whose history breaks it:
#
#   * every assistant message carrying ``tool_calls`` must be followed by
#     exactly one tool result per ``tool_call_id``;
#   * every tool result must have a parent tool call.
#
# Several agent paths can end a turn between those two halves — a middleware
# ``StopAgent`` raised from inside ``wrap_tool``, a resume that answers one of
# several parallel calls, a message trimmer that slices the pair apart. The
# history is then wrong whether or not anything re-sends it, which is why the
# repair lives here rather than at each call site's convenience.

#: What a synthesized tool result says when the call never ran. Deliberately
#: plain text the model can act on: it must not read like a tool *result*, and
#: it must not invite a retry the agent has no way to service (a skipped
#: sibling is never re-dispatched — see ``docs/durability/concepts.md``).
UNANSWERED_TOOL_RESULT = "[no result: {note}]"


def tool_message_imbalance(messages: list[Message]) -> tuple[list[str], list[str]]:
    """Return ``(unanswered_tool_call_ids, orphaned_tool_result_ids)``.

    Pure inspection — nothing is modified. ``unanswered`` is in the order the
    assistant declared the calls; ``orphans`` in the order they appear.
    """
    unanswered: list[str] = []
    orphans: list[str] = []
    open_ids: list[str] = []
    answered: set[str] = set()

    for msg in messages:
        if msg.role == MessageRole.assistant and msg.tool_calls:
            unanswered.extend(i for i in open_ids if i not in answered)
            open_ids = [tc.id for tc in msg.tool_calls]
            answered = set()
        elif msg.role == MessageRole.tool:
            tool_call_id = msg.tool_call_id or ""
            if tool_call_id in open_ids and tool_call_id not in answered:
                answered.add(tool_call_id)
            else:
                orphans.append(tool_call_id)
        else:
            unanswered.extend(i for i in open_ids if i not in answered)
            open_ids = []
            answered = set()

    unanswered.extend(i for i in open_ids if i not in answered)
    return unanswered, orphans


def balance_tool_messages(messages: list[Message], *, note: str) -> list[Message]:
    """Restore the provider invariant: every assistant tool_call has exactly one
    ToolMessage, and every ToolMessage has a parent tool_call.

    Returns a new list. Two repairs, and they are not symmetric:

    * **Unanswered calls are answered**, with a synthetic result carrying
      ``note`` — the tool is *not* run. The note is the honest record that the
      call produced nothing; inventing a plausible result, or dropping the
      assistant message so the call disappears, would both hide a turn the model
      believes it took.
    * **Orphaned results are dropped.** There is nothing truthful to invent a
      parent call from, and a result with no call is the half a provider
      rejects outright.

    ``note`` should say *why* in the model's own terms ("the agent stopped
    before this tool ran"), because the model reads it.
    """
    out: list[Message] = []
    open_ids: list[str] = []
    answered: set[str] = set()

    def flush() -> None:
        nonlocal open_ids, answered
        for tool_call_id in open_ids:
            if tool_call_id not in answered:
                out.append(
                    ToolMessage(
                        content=UNANSWERED_TOOL_RESULT.format(note=note),
                        tool_call_id=tool_call_id,
                    )
                )
        open_ids = []
        answered = set()

    for msg in messages:
        if msg.role == MessageRole.assistant and msg.tool_calls:
            flush()
            out.append(msg)
            open_ids = [tc.id for tc in msg.tool_calls]
        elif msg.role == MessageRole.tool:
            tool_call_id = msg.tool_call_id or ""
            if tool_call_id in open_ids and tool_call_id not in answered:
                answered.add(tool_call_id)
                out.append(msg)
            # else: orphan — dropped.
        else:
            flush()
            out.append(msg)

    flush()
    return out
