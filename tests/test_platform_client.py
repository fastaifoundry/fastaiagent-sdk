"""Tests for Phase 40: Platform API client, push module, and FastAI client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformNotFoundError,
    PlatformRateLimitError,
    PlatformTierLimitError,
)
from fastaiagent._platform.api import PlatformAPI
from fastaiagent._platform.cache import OfflineCache
from fastaiagent.agent import Agent, AgentConfig
from fastaiagent.chain import Chain, NodeType
from fastaiagent.client import FastAI
from fastaiagent.deploy.push import PushResult, push_all, push_resource
from fastaiagent.guardrail import Guardrail
from fastaiagent.llm import LLMClient
from fastaiagent.prompt import Prompt
from fastaiagent.tool import FunctionTool


# --- PlatformAPI error handling tests ---


class TestPlatformAPIErrors:
    def _make_response(self, status_code: int, json_body: dict | None = None) -> httpx.Response:
        return httpx.Response(
            status_code,
            json=json_body or {},
            request=httpx.Request("POST", "https://example.com/test"),
        )

    def test_401_raises_auth_error(self):
        api = PlatformAPI(api_key="bad-key", base_url="https://example.com")
        with pytest.raises(PlatformAuthError, match="Invalid API key"):
            api._handle_response(self._make_response(401))

    def test_403_scope_raises_auth_error(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        with pytest.raises(PlatformAuthError, match="scope"):
            api._handle_response(self._make_response(
                403, {"detail": {"error": "insufficient_scope", "detail": "API key lacks required scope: agent:write"}}
            ))

    def test_403_tier_raises_tier_error(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        with pytest.raises(PlatformTierLimitError, match="Tier limit"):
            api._handle_response(self._make_response(
                403, {"detail": "Tier limit exceeded"}
            ))

    def test_404_raises_not_found(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        with pytest.raises(PlatformNotFoundError):
            api._handle_response(self._make_response(404))

    def test_429_raises_rate_limit(self):
        resp = self._make_response(429)
        resp.headers["Retry-After"] = "30"
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        with pytest.raises(PlatformRateLimitError, match="30"):
            api._handle_response(resp)

    def test_500_raises_connection_error(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        with pytest.raises(PlatformConnectionError, match="500"):
            api._handle_response(self._make_response(500))

    def test_200_returns_json(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        result = api._handle_response(self._make_response(200, {"ok": True}))
        assert result == {"ok": True}

    def test_201_returns_json(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        result = api._handle_response(self._make_response(201, {"id": "123", "created": True}))
        assert result["id"] == "123"


class TestPlatformAPIHeaders:
    def test_headers_include_api_key(self):
        api = PlatformAPI(api_key="fa_k_test123", base_url="https://example.com")
        headers = api._headers()
        assert headers["X-API-Key"] == "fa_k_test123"
        assert "fastaiagent-sdk" in headers["User-Agent"]
        assert headers["Content-Type"] == "application/json"


# --- Push module tests ---


class TestPushResource:
    def test_push_agent(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {
            "created": ["agent:test-agent", "tool:search"],
            "updated": [],
            "errors": [],
        }

        agent = Agent(
            name="test-agent",
            system_prompt="Be helpful",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            tools=[FunctionTool(name="search", description="Search")],
        )

        result = push_resource(api, agent)
        assert isinstance(result, PushResult)
        assert result.resource_type == "agent"
        assert result.name == "test-agent"
        assert result.created is True
        assert "tool:search" in result.dependencies_pushed

        # Verify API was called with correct path
        api.post.assert_called_once()
        call_args = api.post.call_args
        assert call_args[0][0] == "/public/v1/sdk/push"

    def test_push_chain(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {
            "created": ["chain:my-pipeline"],
            "updated": [],
            "errors": [],
        }

        chain = Chain("my-pipeline")
        chain.add_node("a", name="Step A")
        chain.add_node("b", name="Step B")
        chain.connect("a", "b")

        result = push_resource(api, chain)
        assert result.resource_type == "chain"
        assert result.name == "my-pipeline"
        assert result.created is True

    def test_push_tool(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {"id": "tool-123", "name": "calc", "created": True}

        tool = FunctionTool(name="calc", description="Calculator")
        result = push_resource(api, tool)
        assert result.resource_type == "tool"
        assert result.name == "calc"
        assert result.platform_id == "tool-123"

    def test_push_guardrail(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {"id": "gr-123", "name": "no_pii", "created": True}

        gr = Guardrail(name="no_pii", description="Block PII")
        result = push_resource(api, gr)
        assert result.resource_type == "guardrail"
        assert result.name == "no_pii"

    def test_push_prompt(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {"id": "p-123", "name": "greeting", "created": True}

        prompt = Prompt(name="greeting", template="Hello {{name}}!")
        result = push_resource(api, prompt)
        assert result.resource_type == "prompt"
        assert result.name == "greeting"

    def test_push_unsupported_type_raises(self):
        api = MagicMock(spec=PlatformAPI)
        with pytest.raises(TypeError, match="Cannot push"):
            push_resource(api, "not a resource")


class TestBatchPush:
    def test_push_all(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {
            "created": ["agent:bot", "tool:search"],
            "updated": ["chain:pipeline"],
            "errors": [],
        }

        agent = Agent(name="bot", llm=LLMClient())
        chain = Chain("pipeline")
        tool = FunctionTool(name="search", description="Search")

        results = push_all(api, [agent, chain, tool])
        assert len(results) == 3
        assert any(r.name == "bot" and r.created for r in results)
        assert any(r.name == "pipeline" and not r.created for r in results)

    def test_push_all_serializes_correctly(self):
        api = MagicMock(spec=PlatformAPI)
        api.post.return_value = {"created": [], "updated": [], "errors": []}

        agent = Agent(name="a", system_prompt="test", llm=LLMClient())
        push_all(api, [agent])

        call_data = api.post.call_args[0][1]
        assert len(call_data["agents"]) == 1
        assert call_data["agents"][0]["name"] == "a"
        assert call_data["agents"][0]["system_prompt"] == "test"


# --- FastAI client tests ---


class TestFastAI:
    def test_construction(self):
        fa = FastAI(api_key="fa_k_test", target="https://staging.example.com", project="my-proj")
        assert fa.project == "my-proj"
        assert fa._api._api_key == "fa_k_test"
        assert fa._api._base_url == "https://staging.example.com"

    def test_push_delegates_to_push_resource(self):
        fa = FastAI(api_key="fa_k_test")
        with patch("fastaiagent.client.push_resource") as mock_push:
            mock_push.return_value = PushResult(resource_type="agent", name="bot", created=True)
            agent = Agent(name="bot", llm=LLMClient())
            result = fa.push(agent)
            assert result.name == "bot"
            mock_push.assert_called_once()

    def test_push_all_delegates(self):
        fa = FastAI(api_key="fa_k_test")
        with patch("fastaiagent.client.push_all") as mock_push_all:
            mock_push_all.return_value = [
                PushResult(resource_type="agent", name="a", created=True)
            ]
            results = fa.push_all([Agent(name="a", llm=LLMClient())])
            assert len(results) == 1


# --- Offline cache tests ---


class TestOfflineCache:
    def test_set_and_get(self, temp_dir):
        cache = OfflineCache(cache_dir=str(temp_dir / "cache"))
        cache.set("test-key", {"data": "hello"}, ttl_seconds=3600)
        result = cache.get("test-key")
        assert result == {"data": "hello"}

    def test_get_expired(self, temp_dir):
        cache = OfflineCache(cache_dir=str(temp_dir / "cache"))
        cache.set("test-key", {"data": "old"}, ttl_seconds=-1)  # already expired
        result = cache.get("test-key")
        assert result is None

    def test_get_missing(self, temp_dir):
        cache = OfflineCache(cache_dir=str(temp_dir / "cache"))
        assert cache.get("nonexistent") is None

    def test_buffer_push(self, temp_dir):
        cache = OfflineCache(cache_dir=str(temp_dir / "cache"))
        cache.buffer_push("agent", {"name": "test"})
        cache.buffer_push("chain", {"name": "pipeline"})
        buffered = cache.get_buffered_pushes()
        assert len(buffered) == 2

    def test_clear_buffer(self, temp_dir):
        cache = OfflineCache(cache_dir=str(temp_dir / "cache"))
        cache.buffer_push("agent", {"name": "test"})
        count = cache.clear_buffer()
        assert count == 1
        assert len(cache.get_buffered_pushes()) == 0
