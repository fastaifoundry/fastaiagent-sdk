"""Streaming event types for LLM responses.

These types align with the FastAIAgent Platform's StreamEvent protocol,
enabling seamless compatibility between SDK streaming and platform streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastaiagent.llm.message import ReasoningBlock


@dataclass
class TextDelta:
    """A chunk of text from the LLM stream."""

    text: str


@dataclass
class ReasoningDelta:
    """A chunk of the model's reasoning (1.91.0) — never part of the answer text.

    ``text`` is what the provider shows (Anthropic summarized thinking, Gemini
    thought summaries, DeepSeek-style ``reasoning_content``); it can be empty.
    ``block`` is set once per finished reasoning block, on the event that closes
    it: the provider's opaque block, which the agent loop sends back.
    """

    text: str = ""
    block: ReasoningBlock | None = None


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
    #: The provider's opaque token for this call (Gemini ``thoughtSignature``),
    #: carried onto :attr:`ToolCall.signature`. See there.
    signature: str | None = None


@dataclass
class Usage:
    """Token usage from the LLM response.

    ``prompt_tokens`` is the whole prompt, cached part included;
    ``cached_tokens`` / ``cache_write_tokens`` are the parts read from and
    written to the provider's prompt cache (``cache_write_1h_tokens`` the
    one-hour share of the writes), and ``reasoning_tokens`` the part of
    ``completion_tokens`` spent reasoning (1.91.0).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0


@dataclass
class StreamDone:
    """End-of-stream marker.

    ``finish_reason`` (1.89.0) is the provider's own value, as on
    :class:`LLMResponse` (OpenAI ``length``, Anthropic ``max_tokens`` mapped
    the same way as a non-streamed call); ``""`` when the provider sent none.
    A streamed reply cut off at the token limit used to look complete.
    """

    finish_reason: str = ""


@dataclass
class HandoffEvent:
    """Emitted by :class:`fastaiagent.agent.Swarm` when control passes from
    one agent to another.

    Tagged onto the stream ahead of the target agent's first TextDelta so UI
    layers can render an "agent switch" affordance.
    """

    from_agent: str
    to_agent: str
    reason: str = ""


@dataclass
class Paused:
    """The last event of a stream whose run paused (1.77.0).

    Emitted by :meth:`fastaiagent.Agent.astream` when a tool paused the run — a
    managed approval policy or an ``interrupt()`` — instead of a private exception
    escaping the generator. The pause is already checkpointed; resume it with
    ``agent.aresume(execution_id, resume_value=Resume(...))``. For a policy pause
    ``context`` carries ``tool``, ``tool_input`` and ``pending_id``, exactly like
    ``AgentResult.pending_interrupt["context"]``.
    """

    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    execution_id: str = ""
    node_id: str = ""
    agent_path: str | None = None


StreamEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallEnd
    | Usage
    | StreamDone
    | HandoffEvent
    | Paused
)
