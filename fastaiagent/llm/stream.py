"""Streaming event types for LLM responses.

These types align with the FastAIAgent Platform's StreamEvent protocol,
enabling seamless compatibility between SDK streaming and platform streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class TextDelta:
    """A chunk of text from the LLM stream."""

    text: str


@dataclass
class ToolCallStart:
    """Emitted when the LLM initiates a tool call."""

    call_id: str
    tool_name: str


@dataclass
class ToolCallEnd:
    """Emitted when a tool call's arguments are fully parsed."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token usage from the LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamDone:
    """End-of-stream marker."""

    pass


StreamEvent = Union[TextDelta, ToolCallStart, ToolCallEnd, Usage, StreamDone]
