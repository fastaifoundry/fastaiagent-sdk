"""A RESTTool never sends the redaction placeholder as a header (1.89.0, D15).

``RESTTool.to_dict()`` replaces every header value with ``[redacted]`` so a
credential never reaches a trace or the plane. A tool rebuilt from that
definition (``Tool.from_dict``: replay, the runner) used to send the literal
``Authorization: [redacted]`` to the real endpoint. It now refuses, without a
request. A real socket server proves no request was made.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastaiagent.tool.base import Tool
from fastaiagent.tool.rest import RESTTool


@pytest.fixture(autouse=True)
def _allow_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")


def test_a_rebuilt_tool_refuses_without_calling_out(capture_server: Any) -> None:
    original = RESTTool(
        name="refund",
        url=f"{capture_server.url}/refunds",
        method="POST",
        headers={"Authorization": "Bearer real-secret"},
    )
    rebuilt = Tool.from_dict(original.to_dict())

    result = rebuilt.execute({"order": "7"})

    assert result.error and "redaction placeholder" in result.error
    assert "Authorization" in result.error
    assert capture_server.requests == [], "the endpoint was called with a placeholder header"


def test_a_tool_with_real_headers_still_sends(capture_server: Any) -> None:
    live = RESTTool(
        name="refund",
        url=f"{capture_server.url}/refunds",
        method="POST",
        headers={"Authorization": "Bearer real-secret"},
    )
    result = live.execute({"order": "7"})
    assert result.error is None, result.error
    assert len(capture_server.requests) == 1
    assert capture_server.requests[0]["headers"].get("Authorization") == "Bearer real-secret"
