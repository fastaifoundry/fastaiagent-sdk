"""Tests for structured output (response_format) support."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fastaiagent.llm.client import (
    LLMClient,
    LLMResponse,
    _augment_system_for_response_format,
    _ollama_format_from_response_format,
    _strip_code_fences,
)
from fastaiagent.llm.message import SystemMessage, UserMessage


# --- Helper function tests ---


class TestStripCodeFences:
    def test_no_fences(self):
        assert _strip_code_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_plain_fences(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_fences_with_whitespace(self):
        text = '  ```json\n  {"key": "value"}  \n```  '
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_multiline_json(self):
        text = '```json\n{\n  "key": "value",\n  "num": 42\n}\n```'
        result = _strip_code_fences(text)
        parsed = json.loads(result)
        assert parsed == {"key": "value", "num": 42}

    def test_no_match_returns_original(self):
        text = "Just plain text"
        assert _strip_code_fences(text) == "Just plain text"


class TestAugmentSystemForResponseFormat:
    def test_text_type_no_change(self):
        result = _augment_system_for_response_format("Be helpful.", {"type": "text"})
        assert result == "Be helpful."

    def test_json_object(self):
        result = _augment_system_for_response_format("Be helpful.", {"type": "json_object"})
        assert "valid JSON only" in result
        assert "Be helpful." in result
        assert "JSON.parse()" in result

    def test_json_schema(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "country_info",
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}},
                    "required": ["capital"],
                },
            },
        }
        result = _augment_system_for_response_format("Be helpful.", rf)
        assert "Be helpful." in result
        assert "country_info" in result
        assert '"capital"' in result
        assert "JSON.parse()" in result

    def test_none_system_text(self):
        result = _augment_system_for_response_format(None, {"type": "json_object"})
        assert "valid JSON only" in result

    def test_non_dict_response_format(self):
        result = _augment_system_for_response_format("Hello", "text")
        assert result == "Hello"


class TestOllamaFormatConversion:
    def test_text_returns_none(self):
        assert _ollama_format_from_response_format({"type": "text"}) is None

    def test_json_object(self):
        assert _ollama_format_from_response_format({"type": "json_object"}) == "json"

    def test_json_schema_with_schema(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
        result = _ollama_format_from_response_format(rf)
        assert isinstance(result, dict)
        assert result["type"] == "object"

    def test_json_schema_without_schema(self):
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        assert _ollama_format_from_response_format(rf) == "json"

    def test_non_dict_returns_none(self):
        assert _ollama_format_from_response_format("text") is None


# --- OpenAI structured output tests ---


class TestOpenAIStructuredOutput:
    @pytest.mark.asyncio
    async def test_response_format_json_object(self):
        """Test that response_format is passed to OpenAI API body."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer": 42}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "model": "gpt-4o-mini",
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        llm = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test-key")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await llm.acomplete(
                [UserMessage("What is the answer?")],
                response_format={"type": "json_object"},
            )

        assert result.content == '{"answer": 42}'
        # Verify response_format was included in the request body
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert body["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_response_format_json_schema(self):
        """Test json_schema response_format with full schema."""
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "math_result",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "number"}},
                    "required": ["result"],
                },
                "strict": True,
            },
        }
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"result": 42}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "model": "gpt-4o-mini",
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        llm = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test-key")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await llm.acomplete([UserMessage("2+2")], response_format=rf)

        assert result.content == '{"result": 42}'
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_no_response_format_omits_field(self):
        """Without response_format, it should not be in the body."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                "model": "gpt-4o-mini",
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        llm = LLMClient(provider="openai", model="gpt-4o-mini", api_key="test-key")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await llm.acomplete([UserMessage("Hi")])

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert "response_format" not in body


# --- Anthropic structured output tests ---


class TestAnthropicStructuredOutput:
    @pytest.mark.asyncio
    async def test_system_prompt_augmented_for_json_object(self):
        """Anthropic: response_format augments system prompt."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"answer": 42}'}],
                "usage": {"input_tokens": 12, "output_tokens": 6},
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await llm.acomplete(
                [SystemMessage("Be helpful"), UserMessage("What is 2+2?")],
                response_format={"type": "json_object"},
            )

        assert result.content == '{"answer": 42}'
        # Verify system prompt was augmented
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert "valid JSON only" in body["system"]
        assert "Be helpful" in body["system"]

    @pytest.mark.asyncio
    async def test_code_fences_stripped(self):
        """Anthropic: code fences in JSON response are stripped."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '```json\n{"answer": 42}\n```'}],
                "usage": {"input_tokens": 10, "output_tokens": 8},
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await llm.acomplete(
                [UserMessage("Give me JSON")],
                response_format={"type": "json_object"},
            )

        # Code fences should be stripped
        assert result.content == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_json_schema_embeds_schema_in_system(self):
        """Anthropic: json_schema embeds full schema in system prompt."""
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name", "age"],
                },
            },
        }
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"name": "Alice", "age": 30}'}],
                "usage": {"input_tokens": 15, "output_tokens": 10},
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await llm.acomplete([UserMessage("Describe Alice")], response_format=rf)

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert "'person'" in body["system"]
        assert '"name"' in body["system"]
        assert result.content == '{"name": "Alice", "age": 30}'


# --- Ollama structured output tests ---


class TestOllamaStructuredOutput:
    @pytest.mark.asyncio
    async def test_json_object_sets_format(self):
        """Ollama: json_object sets format='json'."""
        mock_response = httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"answer": 42}'},
                "prompt_eval_count": 8,
                "eval_count": 4,
            },
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        llm = LLMClient(provider="ollama", model="llama3")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await llm.acomplete(
                [UserMessage("What is 2+2?")],
                response_format={"type": "json_object"},
            )

        assert result.content == '{"answer": 42}'
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert body["format"] == "json"

    @pytest.mark.asyncio
    async def test_json_schema_sets_schema_dict(self):
        """Ollama: json_schema sets format to the schema dict."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        rf = {"type": "json_schema", "json_schema": {"name": "test", "schema": schema}}
        mock_response = httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"x": "hello"}'},
                "prompt_eval_count": 8,
                "eval_count": 4,
            },
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        llm = LLMClient(provider="ollama", model="llama3")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await llm.acomplete([UserMessage("test")], response_format=rf)

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert body["format"] == schema

    @pytest.mark.asyncio
    async def test_text_type_no_format(self):
        """Ollama: text type does not set format."""
        mock_response = httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Hello"},
                "prompt_eval_count": 8,
                "eval_count": 4,
            },
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        llm = LLMClient(provider="ollama", model="llama3")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await llm.acomplete([UserMessage("Hi")], response_format={"type": "text"})

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert "format" not in body
