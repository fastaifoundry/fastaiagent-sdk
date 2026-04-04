"""Tests for platform service integrations via fa.connect()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from fastaiagent._internal.errors import PlatformNotConnectedError, PromptNotFoundError
from fastaiagent.client import _connection
from fastaiagent.eval.dataset import Dataset
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.prompt.registry import PromptRegistry
from fastaiagent.trace.replay import Replay
from fastaiagent.trace.storage import SpanData, TraceData


@pytest.fixture(autouse=True)
def _reset_connection():
    """Reset connection state before and after each test."""
    _connection.api_key = None
    _connection.target = "https://app.fastaiagent.net"
    _connection.project = None
    _connection._platform_processor = None
    yield
    _connection.api_key = None
    _connection.target = "https://app.fastaiagent.net"
    _connection.project = None
    _connection._platform_processor = None


def _set_connected():
    """Helper to set up a connected state."""
    _connection.api_key = "fa_k_test"
    _connection.target = "https://test.example.com"
    _connection.project = "test-project"


def _mock_api_get(return_value):
    """Create a mock for PlatformAPI.get."""
    return patch("fastaiagent._platform.api.PlatformAPI.get", return_value=return_value)


def _mock_api_post(return_value=None):
    """Create a mock for PlatformAPI.post."""
    return patch(
        "fastaiagent._platform.api.PlatformAPI.post", return_value=return_value or {"ok": True}
    )


# --- TraceData.publish() ---


class TestTraceDataPublish:
    def test_publish_raises_when_not_connected(self):
        trace = TraceData(trace_id="abc123")
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            trace.publish()

    def test_publish_sends_to_platform(self):
        _set_connected()
        trace = TraceData(
            trace_id="abc123",
            name="test-trace",
            spans=[SpanData(span_id="s1", trace_id="abc123", name="root")],
        )
        with _mock_api_post() as mock_post:
            trace.publish()
            mock_post.assert_called_once()
            call_data = mock_post.call_args[0][1]
            assert call_data["project"] == "test-project"
            assert len(call_data["spans"]) == 1


# --- Replay.from_platform() ---


class TestReplayFromPlatform:
    def test_raises_when_not_connected(self):
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            Replay.from_platform("tr-abc123")

    def test_fetches_trace_and_creates_replay(self):
        _set_connected()
        platform_data = {
            "trace_id": "tr-abc123",
            "name": "test-trace",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T00:00:01Z",
            "status": "OK",
            "metadata": {},
            "spans": [
                {
                    "span_id": "s1",
                    "trace_id": "tr-abc123",
                    "name": "agent.run",
                    "start_time": "2025-01-01T00:00:00Z",
                    "end_time": "2025-01-01T00:00:01Z",
                    "status": "OK",
                }
            ],
        }
        with _mock_api_get(platform_data):
            replay = Replay.from_platform("tr-abc123")
            assert replay._trace.trace_id == "tr-abc123"
            assert len(replay._trace.spans) == 1
            assert replay._trace.spans[0].name == "agent.run"


# --- PromptRegistry ---


class TestPromptRegistryPlatform:
    def test_get_local_source(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greeting", "Hello {{name}}")
        prompt = reg.get("greeting", source="local")
        assert "Hello" in prompt.template

    def test_get_platform_source_raises_when_not_connected(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        with pytest.raises(PlatformNotConnectedError):
            reg.get("test", source="platform")

    def test_get_platform_source_fetches_from_platform(self, temp_dir):
        _set_connected()
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        platform_data = {
            "slug": "support-prompt",
            "content": "You are a helpful agent.",
            "variables": [],
            "version": 3,
            "metadata": {},
        }
        with _mock_api_get(platform_data):
            prompt = reg.get("support-prompt", source="platform")
            assert prompt.template == "You are a helpful agent."
            assert prompt.version == 3

    def test_get_platform_source_not_found_raises(self, temp_dir):
        _set_connected()
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        with patch(
            "fastaiagent._platform.api.PlatformAPI.get",
            side_effect=Exception("not found"),
        ):
            with pytest.raises(PromptNotFoundError, match="not found on platform"):
                reg.get("nonexistent", source="platform")

    def test_get_auto_uses_platform_when_connected(self, temp_dir):
        _set_connected()
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        platform_data = {
            "slug": "auto-prompt",
            "content": "Platform prompt.",
            "variables": [],
            "version": 1,
            "metadata": {},
        }
        with _mock_api_get(platform_data):
            prompt = reg.get("auto-prompt", source="auto")
            assert prompt.template == "Platform prompt."

    def test_ttl_cache_hit(self, temp_dir):
        _set_connected()
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        platform_data = {
            "slug": "cached",
            "content": "Cached prompt.",
            "variables": [],
            "version": 1,
            "metadata": {},
        }
        with _mock_api_get(platform_data) as mock_get:
            reg.get("cached", source="platform")
            reg.get("cached", source="platform")
            # Second call should use cache, not call API again
            assert mock_get.call_count == 1

    def test_refresh_clears_cache(self, temp_dir):
        _set_connected()
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        platform_data = {
            "slug": "cached",
            "content": "Cached prompt.",
            "variables": [],
            "version": 1,
            "metadata": {},
        }
        with _mock_api_get(platform_data) as mock_get:
            reg.get("cached", source="platform")
            reg.refresh("cached")
            reg.get("cached", source="platform")
            assert mock_get.call_count == 2

    def test_publish_raises_when_not_connected(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        with pytest.raises(PlatformNotConnectedError):
            reg.publish("slug", "content")

    def test_publish_posts_to_platform(self, temp_dir):
        _set_connected()
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        with _mock_api_post() as mock_post:
            reg.publish("support-prompt", "Hello {{name}}", variables=["name"])
            mock_post.assert_called_once()
            call_data = mock_post.call_args[0][1]
            assert call_data["slug"] == "support-prompt"
            assert call_data["content"] == "Hello {{name}}"
            assert call_data["variables"] == ["name"]


# --- Dataset ---


class TestDatasetPlatform:
    def test_from_platform_raises_when_not_connected(self):
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            Dataset.from_platform("golden-test-set")

    def test_from_platform_fetches_dataset(self):
        _set_connected()
        platform_data = {
            "items": [
                {"input": "What is 2+2?", "expected": "4"},
                {"input": "What is 3+3?", "expected": "6"},
            ]
        }
        with _mock_api_get(platform_data):
            ds = Dataset.from_platform("golden-test-set")
            assert len(ds) == 2
            assert ds[0]["input"] == "What is 2+2?"

    def test_publish_raises_when_not_connected(self):
        ds = Dataset.from_list([{"input": "test"}])
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            ds.publish("test-set")

    def test_publish_posts_to_platform(self):
        _set_connected()
        ds = Dataset.from_list([{"input": "test", "expected": "result"}])
        with _mock_api_post() as mock_post:
            ds.publish("regression-tests")
            mock_post.assert_called_once()
            call_data = mock_post.call_args[0][1]
            assert call_data["name"] == "regression-tests"
            assert len(call_data["items"]) == 1


# --- EvalResults ---


class TestEvalResultsPublish:
    def test_publish_raises_when_not_connected(self):
        results = EvalResults()
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            results.publish()

    def test_publish_posts_to_platform(self):
        _set_connected()
        results = EvalResults()
        results.add("accuracy", ScorerResult(score=0.9, passed=True, reason="correct"))
        results.add("accuracy", ScorerResult(score=0.8, passed=True, reason="mostly correct"))

        with _mock_api_post() as mock_post:
            results.publish(run_name="v2.1-rc")
            mock_post.assert_called_once()
            call_data = mock_post.call_args[0][1]
            assert call_data["run_name"] == "v2.1-rc"
            assert "accuracy" in call_data["scores"]
            assert len(call_data["scores"]["accuracy"]) == 2


# --- Scorer ---


class TestScorerFromPlatform:
    def test_from_platform_raises_when_not_connected(self):
        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            Scorer.from_platform("correctness-judge")

    def test_from_platform_returns_llm_judge(self):
        _set_connected()
        platform_data = {
            "criteria": "helpfulness",
            "prompt_template": "Rate the helpfulness: {input} -> {output}",
            "scale": "0-1",
        }
        with _mock_api_get(platform_data):
            scorer = Scorer.from_platform("helpfulness-judge")
            from fastaiagent.eval.llm_judge import LLMJudge

            assert isinstance(scorer, LLMJudge)
            assert scorer.criteria == "helpfulness"
            assert scorer.scale == "0-1"


# --- PlatformSpanExporter ---


class TestPlatformSpanExporter:
    def test_export_returns_success_when_not_connected(self):
        from fastaiagent.trace.platform_export import PlatformSpanExporter

        from opentelemetry.sdk.trace.export import SpanExportResult

        exporter = PlatformSpanExporter()
        result = exporter.export([])
        assert result == SpanExportResult.SUCCESS

    def test_export_posts_spans(self):
        _set_connected()
        from fastaiagent.trace.platform_export import PlatformSpanExporter

        from opentelemetry.sdk.trace.export import SpanExportResult

        exporter = PlatformSpanExporter()

        # Create a mock span
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.trace_id = 0x1234
        mock_ctx.span_id = 0x5678
        mock_span.get_span_context.return_value = mock_ctx
        mock_span.parent = None
        mock_span.name = "test-span"
        mock_span.start_time = 1000000000
        mock_span.end_time = 2000000000
        mock_span.attributes = {"key": "value"}
        mock_span.events = []
        mock_span.status = MagicMock()
        mock_span.status.status_code = MagicMock()
        mock_span.status.status_code.name = "OK"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = exporter.export([mock_span])
            assert result == SpanExportResult.SUCCESS
            mock_client.post.assert_called_once()

    def test_export_swallows_errors(self):
        _set_connected()
        from fastaiagent.trace.platform_export import PlatformSpanExporter

        from opentelemetry.sdk.trace.export import SpanExportResult

        exporter = PlatformSpanExporter()

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.trace_id = 0x1234
        mock_ctx.span_id = 0x5678
        mock_span.get_span_context.return_value = mock_ctx
        mock_span.parent = None
        mock_span.name = "test"
        mock_span.start_time = 1000000000
        mock_span.end_time = 2000000000
        mock_span.attributes = {}
        mock_span.events = []
        mock_span.status = MagicMock()
        mock_span.status.status_code = MagicMock()
        mock_span.status.status_code.name = "OK"

        with patch("httpx.Client", side_effect=Exception("network error")):
            result = exporter.export([mock_span])
            # Should still return SUCCESS — local SQLite has the data
            assert result == SpanExportResult.SUCCESS
