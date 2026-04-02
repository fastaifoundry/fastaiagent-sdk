"""LLM client abstraction with multi-provider support."""

from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import (
    AssistantMessage,
    Message,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from fastaiagent.llm.stream import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "MessageRole",
    "ToolCall",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    # Streaming
    "StreamEvent",
    "TextDelta",
    "ToolCallStart",
    "ToolCallEnd",
    "Usage",
    "StreamDone",
]
