"""Tests for platform API client and connection management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformNotConnectedError,
    PlatformNotFoundError,
    PlatformRateLimitError,
    PlatformTierLimitError,
)
from fastaiagent._platform.api import PlatformAPI, get_platform_api
from fastaiagent.client import _connection, connect, disconnect

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
            api._handle_response(
                self._make_response(
                    403,
                    {
                        "detail": {
                            "error": "insufficient_scope",
                            "detail": "API key lacks required scope: agent:write",
                        }
                    },
                )
            )

    def test_403_tier_raises_tier_error(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        with pytest.raises(PlatformTierLimitError, match="Tier limit"):
            api._handle_response(self._make_response(403, {"detail": "Tier limit exceeded"}))

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


class TestPlatformAPIGet:
    def test_get_calls_correct_url(self):
        api = PlatformAPI(api_key="key", base_url="https://example.com")
        mock_response = httpx.Response(
            200,
            json={"data": "test"},
            request=httpx.Request("GET", "https://example.com/public/v1/test"),
        )
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = api.get("/public/v1/test", params={"version": 3})
            assert result == {"data": "test"}
            mock_client.get.assert_called_once()


# --- Connection management tests ---


class TestConnection:
    def setup_method(self):
        """Reset connection state before each test."""
        _connection.api_key = None
        _connection.target = "https://app.fastaiagent.net"
        _connection.project = None
        _connection._platform_processor = None

    def teardown_method(self):
        """Clean up connection state after each test."""
        _connection.api_key = None
        _connection.target = "https://app.fastaiagent.net"
        _connection.project = None
        _connection._platform_processor = None

    def test_not_connected_by_default(self):
        assert _connection.is_connected is False

    def test_connect_sets_state(self):
        mock_response = httpx.Response(
            200,
            json={"ok": True},
            request=httpx.Request("GET", "https://example.com/public/v1/auth/check"),
        )
        with patch("httpx.Client") as mock_client_cls, \
             patch("fastaiagent.trace.otel.get_tracer_provider") as mock_tp, \
             patch("fastaiagent.trace.platform_export.PlatformSpanExporter"):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client
            mock_tp.return_value = MagicMock()

            connect(api_key="fa_k_test", target="https://staging.example.com", project="proj")

        assert _connection.is_connected is True
        assert _connection.api_key == "fa_k_test"
        assert _connection.target == "https://staging.example.com"
        assert _connection.project == "proj"

    def test_connect_strips_trailing_slash(self):
        mock_response = httpx.Response(
            200,
            json={"ok": True},
            request=httpx.Request("GET", "https://example.com/public/v1/auth/check"),
        )
        with patch("httpx.Client") as mock_client_cls, \
             patch("fastaiagent.trace.otel.get_tracer_provider") as mock_tp, \
             patch("fastaiagent.trace.platform_export.PlatformSpanExporter"):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client
            mock_tp.return_value = MagicMock()

            connect(api_key="fa_k_test", target="https://example.com/")

        assert _connection.target == "https://example.com"

    def test_connect_auth_failure_resets_state(self):
        mock_response = httpx.Response(
            401,
            json={},
            request=httpx.Request("GET", "https://example.com/public/v1/auth/check"),
        )
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            with pytest.raises(PlatformAuthError, match="Invalid API key"):
                connect(api_key="bad-key")

        assert _connection.is_connected is False

    def test_disconnect_clears_state(self):
        _connection.api_key = "fa_k_test"
        _connection.project = "proj"
        _connection._platform_processor = MagicMock()

        disconnect()

        assert _connection.is_connected is False
        assert _connection.api_key is None
        assert _connection.project is None

    def test_headers_property(self):
        _connection.api_key = "fa_k_test"
        headers = _connection.headers
        assert headers["X-API-Key"] == "fa_k_test"
        assert headers["Content-Type"] == "application/json"
        assert "fastaiagent-sdk" in headers["User-Agent"]


class TestGetPlatformAPI:
    def setup_method(self):
        _connection.api_key = None
        _connection.target = "https://app.fastaiagent.net"
        _connection.project = None

    def teardown_method(self):
        _connection.api_key = None

    def test_raises_when_not_connected(self):
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            get_platform_api()

    def test_returns_api_when_connected(self):
        _connection.api_key = "fa_k_test"
        _connection.target = "https://staging.example.com"
        api = get_platform_api()
        assert isinstance(api, PlatformAPI)
        assert api._api_key == "fa_k_test"
        assert api._base_url == "https://staging.example.com"


class TestModuleLevelAPI:
    def setup_method(self):
        _connection.api_key = None
        _connection.project = None

    def teardown_method(self):
        _connection.api_key = None
        _connection.project = None

    def test_is_connected_attribute(self):
        import fastaiagent as fa

        assert fa.is_connected is False
        _connection.api_key = "test"
        assert fa.is_connected is True
        _connection.api_key = None

    def test_connect_and_disconnect_exported(self):
        import fastaiagent as fa

        assert hasattr(fa, "connect")
        assert hasattr(fa, "disconnect")
        assert callable(fa.connect)
        assert callable(fa.disconnect)
