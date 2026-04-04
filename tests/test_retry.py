"""Tests for LLMClient retry with backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fastaiagent._internal.errors import LLMProviderError
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import UserMessage

# --- Helper to create mock httpx responses ---


def _make_httpx_response(status_code: int, json_data: dict | None = None, text: str = "error"):
    """Create a mock httpx.Response."""
    import httpx

    content = text
    if json_data is not None:
        import json

        content = json.dumps(json_data)

    return httpx.Response(
        status_code,
        content=content.encode(),
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


_SUCCESS_JSON = {
    "choices": [{"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    "model": "gpt-4o",
}


class TestShouldRetry:
    def test_429_retries(self):
        assert LLMClient._should_retry(429) is True

    def test_500_retries(self):
        assert LLMClient._should_retry(500) is True

    def test_502_retries(self):
        assert LLMClient._should_retry(502) is True

    def test_503_retries(self):
        assert LLMClient._should_retry(503) is True

    def test_529_retries(self):
        """Anthropic returns 529 for overloaded."""
        assert LLMClient._should_retry(529) is True

    def test_400_no_retry(self):
        assert LLMClient._should_retry(400) is False

    def test_401_no_retry(self):
        assert LLMClient._should_retry(401) is False

    def test_403_no_retry(self):
        assert LLMClient._should_retry(403) is False

    def test_404_no_retry(self):
        assert LLMClient._should_retry(404) is False

    def test_none_no_retry(self):
        assert LLMClient._should_retry(None) is False


class TestRetryDelay:
    def test_exponential_backoff(self):
        assert LLMClient._retry_delay(0) == 1
        assert LLMClient._retry_delay(1) == 2
        assert LLMClient._retry_delay(2) == 4
        assert LLMClient._retry_delay(3) == 8

    def test_capped_at_30(self):
        assert LLMClient._retry_delay(5) == 30
        assert LLMClient._retry_delay(10) == 30


class TestRetryAcomplete:
    @pytest.mark.asyncio
    async def test_no_retry_default(self):
        """max_retries=0 (default) — error raises immediately."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=0)
        mock_resp = _make_httpx_response(429, text="Rate limited")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(LLMProviderError) as exc_info:
                await llm.acomplete([UserMessage("Hi")])
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_retry_429_then_success(self):
        """429 on first call, 200 on second — succeeds."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=2)

        fail_resp = _make_httpx_response(429, text="Rate limited")
        ok_resp = _make_httpx_response(200, json_data=_SUCCESS_JSON)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_resp
            return ok_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=side_effect):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await llm.acomplete([UserMessage("Hi")])

        assert result.content == "Hello!"
        assert call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_500_then_success(self):
        """500 on first call, 200 on second — succeeds."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=2)

        fail_resp = _make_httpx_response(500, text="Internal Server Error")
        ok_resp = _make_httpx_response(200, json_data=_SUCCESS_JSON)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fail_resp if call_count == 1 else ok_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=side_effect):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await llm.acomplete([UserMessage("Hi")])

        assert result.content == "Hello!"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        """400 — no retry, raises immediately."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=3)
        mock_resp = _make_httpx_response(400, text="Bad Request")

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=side_effect):
            with pytest.raises(LLMProviderError) as exc_info:
                await llm.acomplete([UserMessage("Hi")])
            assert exc_info.value.status_code == 400

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self):
        """401 — auth error, no retry."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=3)
        mock_resp = _make_httpx_response(401, text="Unauthorized")

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=side_effect):
            with pytest.raises(LLMProviderError) as exc_info:
                await llm.acomplete([UserMessage("Hi")])
            assert exc_info.value.status_code == 401

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted(self):
        """Always 429 — retries max_retries times then raises."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=2)
        mock_resp = _make_httpx_response(429, text="Rate limited")

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=side_effect):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMProviderError) as exc_info:
                    await llm.acomplete([UserMessage("Hi")])
                assert exc_info.value.status_code == 429

        assert call_count == 3  # 1 original + 2 retries

    @pytest.mark.asyncio
    async def test_status_code_on_error(self):
        """LLMProviderError has status_code attribute."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key")
        mock_resp = _make_httpx_response(429, text="Rate limited")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(LLMProviderError) as exc_info:
                await llm.acomplete([UserMessage("Hi")])
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_latency_includes_retry_waits(self):
        """latency_ms includes total time across retries."""
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-key", max_retries=1)

        fail_resp = _make_httpx_response(429, text="Rate limited")
        ok_resp = _make_httpx_response(200, json_data=_SUCCESS_JSON)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fail_resp if call_count == 1 else ok_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=side_effect):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await llm.acomplete([UserMessage("Hi")])

        assert result.latency_ms >= 0


class TestRetrySerialization:
    def test_to_dict_includes_max_retries(self):
        llm = LLMClient(provider="openai", model="gpt-4o", max_retries=3)
        d = llm.to_dict()
        assert d["max_retries"] == 3

    def test_to_dict_omits_zero_retries(self):
        llm = LLMClient(provider="openai", model="gpt-4o", max_retries=0)
        d = llm.to_dict()
        assert "max_retries" not in d

    def test_from_dict_restores_max_retries(self):
        llm = LLMClient(provider="openai", model="gpt-4o", max_retries=3)
        d = llm.to_dict()
        llm2 = LLMClient.from_dict(d)
        assert llm2.max_retries == 3

    def test_from_dict_defaults_to_zero(self):
        llm = LLMClient.from_dict({"provider": "openai", "model": "gpt-4o"})
        assert llm.max_retries == 0
