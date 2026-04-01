"""Tests for fastaiagent.llm module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fastaiagent.llm import (
    AssistantMessage,
    LLMClient,
    LLMResponse,
    Message,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

# --- Message tests ---


class TestMessage:
    def test_system_message(self):
        msg = SystemMessage("You are helpful")
        assert msg.role == MessageRole.system
        assert msg.content == "You are helpful"

    def test_user_message(self):
        msg = UserMessage("Hello")
        assert msg.role == MessageRole.user
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = AssistantMessage("Hi there")
        assert msg.role == MessageRole.assistant
        assert msg.content == "Hi there"

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="search", arguments={"query": "test"})
        msg = AssistantMessage(tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_tool_message(self):
        msg = ToolMessage("result data", tool_call_id="call_1")
        assert msg.role == MessageRole.tool
        assert msg.tool_call_id == "call_1"

    def test_to_openai_format(self):
        msg = UserMessage("Hello")
        fmt = msg.to_openai_format()
        assert fmt == {"role": "user", "content": "Hello"}

    def test_to_openai_format_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        msg = AssistantMessage(content=None, tool_calls=[tc])
        fmt = msg.to_openai_format()
        assert fmt["role"] == "assistant"
        assert len(fmt["tool_calls"]) == 1
        assert fmt["tool_calls"][0]["function"]["name"] == "search"

    def test_from_openai_format(self):
        data = {"role": "user", "content": "Hello"}
        msg = Message.from_openai_format(data)
        assert msg.role == MessageRole.user
        assert msg.content == "Hello"

    def test_from_openai_format_with_tool_calls(self):
        data = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
        }
        msg = Message.from_openai_format(data)
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].arguments == {"q": "test"}

    def test_tool_call_to_openai_format(self):
        tc = ToolCall(id="call_1", name="greet", arguments={"name": "World"})
        fmt = tc.to_openai_format()
        assert fmt["id"] == "call_1"
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "greet"
        assert json.loads(fmt["function"]["arguments"]) == {"name": "World"}


# --- LLMClient tests ---


class TestLLMClient:
    def test_construction_defaults(self):
        llm = LLMClient()
        assert llm.provider == "openai"
        assert llm.model == "gpt-4o-mini"
        assert llm.base_url == "https://api.openai.com/v1"

    def test_construction_custom(self):
        llm = LLMClient(
            provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test"
        )
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.base_url == "https://api.anthropic.com/v1"

    def test_ollama_default_url(self):
        llm = LLMClient(provider="ollama", model="llama3")
        assert llm.base_url == "http://localhost:11434"

    def test_to_dict(self):
        llm = LLMClient(provider="openai", model="gpt-4o", temperature=0.7)
        d = llm.to_dict()
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["temperature"] == 0.7
        assert "base_url" not in d  # default URL omitted

    def test_to_dict_custom_base_url(self):
        llm = LLMClient(provider="custom", base_url="https://myapi.com/v1")
        d = llm.to_dict()
        assert d["base_url"] == "https://myapi.com/v1"

    def test_from_dict_roundtrip(self):
        original = LLMClient(
            provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.5, max_tokens=1000
        )
        d = original.to_dict()
        restored = LLMClient.from_dict(d)
        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens

    def test_unsupported_provider_raises(self):
        llm = LLMClient(provider="unsupported_xyz")
        with pytest.raises(Exception, match="Unsupported provider"):
            llm._get_provider_fn()


class TestLLMClientOpenAI:
    @pytest.mark.asyncio
    async def test_openai_complete(self):
        """Test OpenAI provider with mocked HTTP."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model": "gpt-4o-mini",
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        llm = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test-key")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await llm.acomplete([UserMessage("Hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_openai_with_tool_calls(self):
        """Test OpenAI provider returns tool calls."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "weather"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                "model": "gpt-4o-mini",
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        llm = LLMClient(provider="openai", api_key="test-key")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await llm.acomplete([UserMessage("What's the weather?")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "weather"}
        assert result.finish_reason == "tool_calls"


class TestLLMClientAnthropic:
    @pytest.mark.asyncio
    async def test_anthropic_complete(self):
        """Test Anthropic provider with mocked HTTP."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "usage": {"input_tokens": 12, "output_tokens": 6},
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await llm.acomplete(
                [SystemMessage("Be helpful"), UserMessage("Hi")]
            )

        assert result.content == "Hello from Claude!"
        assert result.usage["prompt_tokens"] == 12
        assert result.finish_reason == "stop"  # normalized from end_turn

    @pytest.mark.asyncio
    async def test_anthropic_with_tool_calls(self):
        """Test Anthropic provider returns tool calls."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "test"}}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 15},
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "tool_use",
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await llm.acomplete([UserMessage("Search")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_1"
        assert result.tool_calls[0].name == "search"
        assert result.finish_reason == "tool_calls"


class TestLLMClientOllama:
    @pytest.mark.asyncio
    async def test_ollama_complete(self):
        """Test Ollama provider with mocked HTTP."""
        mock_response = httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Hello from Ollama!"},
                "prompt_eval_count": 8,
                "eval_count": 4,
            },
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        llm = LLMClient(provider="ollama", model="llama3")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await llm.acomplete([UserMessage("Hi")])

        assert result.content == "Hello from Ollama!"
        assert result.usage["prompt_tokens"] == 8
        assert result.finish_reason == "stop"
