"""Tests for additional LLM parameters (top_p, stop, seed, etc.)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import UserMessage


def _make_httpx_response(status_code: int, json_data: dict):
    import httpx

    return httpx.Response(
        status_code,
        content=json.dumps(json_data).encode(),
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


_SUCCESS_JSON = {
    "choices": [{"message": {"content": "OK", "role": "assistant"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    "model": "gpt-4o",
}


def _capture_body():
    """Helper to capture the request body sent to the provider."""
    captured = {}

    async def mock_post(self_or_url, *args, **kwargs):
        captured["body"] = kwargs.get("json", {})
        return _make_httpx_response(200, _SUCCESS_JSON)

    return captured, mock_post


class TestOpenAIParams:
    @pytest.mark.asyncio
    async def test_top_p_in_body(self):
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k", top_p=0.9)
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        assert captured["body"]["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_stop_in_body(self):
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k", stop=["END", "STOP"])
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        assert captured["body"]["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_seed_in_body(self):
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k", seed=42)
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        assert captured["body"]["seed"] == 42

    @pytest.mark.asyncio
    async def test_frequency_penalty_in_body(self):
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k", frequency_penalty=0.5)
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        assert captured["body"]["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_presence_penalty_in_body(self):
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k", presence_penalty=0.3)
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        assert captured["body"]["presence_penalty"] == 0.3

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_with_tools(self):
        """parallel_tool_calls only included when tools are present."""
        llm = LLMClient(
            provider="openai", model="gpt-4o", api_key="k", parallel_tool_calls=False
        )
        captured, mock_fn = _capture_body()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")], tools=tools)

        assert captured["body"]["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_without_tools(self):
        """parallel_tool_calls NOT included when no tools."""
        llm = LLMClient(
            provider="openai", model="gpt-4o", api_key="k", parallel_tool_calls=False
        )
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        assert "parallel_tool_calls" not in captured["body"]

    @pytest.mark.asyncio
    async def test_none_params_omitted(self):
        """Unset params are not included in body."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k")
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")])

        body = captured["body"]
        assert "top_p" not in body
        assert "stop" not in body
        assert "seed" not in body
        assert "frequency_penalty" not in body
        assert "presence_penalty" not in body
        assert "parallel_tool_calls" not in body


class TestPerCallOverride:
    @pytest.mark.asyncio
    async def test_kwargs_override_init(self):
        """Per-call kwargs override constructor defaults."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="k", top_p=0.9, seed=42)
        captured, mock_fn = _capture_body()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_fn):
            await llm.acomplete([UserMessage("Hi")], top_p=0.5, seed=100)

        assert captured["body"]["top_p"] == 0.5
        assert captured["body"]["seed"] == 100


class TestAnthropicParams:
    @pytest.mark.asyncio
    async def test_top_p_in_body(self):
        llm = LLMClient(
            provider="anthropic", model="claude-sonnet-4-20250514", api_key="k", top_p=0.9
        )
        captured, mock_fn = _capture_body()

        # Anthropic returns different JSON format
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
        }

        async def mock_anthropic(self_or_url, *args, **kwargs):
            captured["body"] = kwargs.get("json", {})
            return _make_httpx_response(200, anthropic_response)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_anthropic):
            await llm.acomplete([UserMessage("Hi")])

        assert captured["body"]["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_stop_as_stop_sequences(self):
        """Anthropic uses stop_sequences (list)."""
        llm = LLMClient(
            provider="anthropic", model="claude-sonnet-4-20250514", api_key="k", stop="END"
        )
        captured, mock_fn = _capture_body()

        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
        }

        async def mock_anthropic(self_or_url, *args, **kwargs):
            captured["body"] = kwargs.get("json", {})
            return _make_httpx_response(200, anthropic_response)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_anthropic):
            await llm.acomplete([UserMessage("Hi")])

        # String stop should be converted to list
        assert captured["body"]["stop_sequences"] == ["END"]
        assert "stop" not in captured["body"]

    @pytest.mark.asyncio
    async def test_unsupported_params_not_in_body(self):
        """Anthropic doesn't support seed, frequency_penalty, etc."""
        llm = LLMClient(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="k",
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        captured = {}

        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
        }

        async def mock_anthropic(self_or_url, *args, **kwargs):
            captured["body"] = kwargs.get("json", {})
            return _make_httpx_response(200, anthropic_response)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_anthropic):
            await llm.acomplete([UserMessage("Hi")])

        body = captured["body"]
        assert "seed" not in body
        assert "frequency_penalty" not in body
        assert "presence_penalty" not in body


class TestOllamaParams:
    @pytest.mark.asyncio
    async def test_params_in_options_dict(self):
        """Ollama params go into body['options'] dict."""
        llm = LLMClient(
            provider="ollama",
            model="llama3",
            top_p=0.9,
            stop=["END"],
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        captured = {}

        ollama_response = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 3,
        }

        async def mock_ollama(self_or_url, *args, **kwargs):
            captured["body"] = kwargs.get("json", {})
            return _make_httpx_response(200, ollama_response)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_ollama):
            await llm.acomplete([UserMessage("Hi")])

        options = captured["body"]["options"]
        assert options["top_p"] == 0.9
        assert options["stop"] == ["END"]
        assert options["seed"] == 42
        assert options["frequency_penalty"] == 0.5
        assert options["presence_penalty"] == 0.3


class TestParamsSerialization:
    def test_to_dict_includes_non_none(self):
        llm = LLMClient(
            provider="openai", model="gpt-4o", top_p=0.9, seed=42, stop=["END"]
        )
        d = llm.to_dict()
        assert d["top_p"] == 0.9
        assert d["seed"] == 42
        assert d["stop"] == ["END"]
        assert "frequency_penalty" not in d
        assert "presence_penalty" not in d
        assert "parallel_tool_calls" not in d

    def test_from_dict_restores_params(self):
        llm = LLMClient(
            provider="openai",
            model="gpt-4o",
            top_p=0.9,
            seed=42,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
            parallel_tool_calls=False,
        )
        d = llm.to_dict()
        llm2 = LLMClient.from_dict(d)
        assert llm2.top_p == 0.9
        assert llm2.seed == 42
        assert llm2.stop == ["END"]
        assert llm2.frequency_penalty == 0.5
        assert llm2.presence_penalty == 0.3
        assert llm2.parallel_tool_calls is False

    def test_from_dict_defaults_to_none(self):
        llm = LLMClient.from_dict({"provider": "openai", "model": "gpt-4o"})
        assert llm.top_p is None
        assert llm.stop is None
        assert llm.seed is None
        assert llm.frequency_penalty is None
        assert llm.presence_penalty is None
        assert llm.parallel_tool_calls is None
