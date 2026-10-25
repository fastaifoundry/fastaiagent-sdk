"""Every call to the plane announces ``X-FAA-Protocol: v1.11`` (1.94.0, S7).

The plane reads the header for telemetry and feature-gating. The SDK never sent
it, though the plane's S7 brief assumed it had since wire v1.10. Asserted on the
requests a real localhost server receives from each of the SDK's three plane
clients: the connection (traces, prompts, …), ``PlatformAPI`` and the runner
channel.
"""

from __future__ import annotations

import asyncio
import json

from fastaiagent._internal.wire import PLANE_WIRE_PROTOCOL
from fastaiagent.client import _connection
from fastaiagent.trace.platform_export import PlatformSpanExporter
from fastaiagent.trace.storage import TraceStore
from tests.test_platform_buffering import _seed_span


def _protocol(request: dict) -> str | None:
    headers = {k.lower(): v for k, v in request["headers"].items()}
    return headers.get("x-faa-protocol")


def test_the_wire_version_is_v1_11() -> None:
    assert PLANE_WIRE_PROTOCOL == "v1.11"


def test_trace_export_sends_the_header(isolated_local_db, capture_server) -> None:
    _connection.api_key = "fa_k_test"
    _connection.target = capture_server.url
    _connection.project_id = "platform-uuid-1"
    try:
        store = TraceStore()
        _seed_span(store, "s1")
        PlatformSpanExporter().export([])
        (request,) = capture_server.ingest_requests
        assert _protocol(request) == "v1.11"
        store.close()
    finally:
        _connection.api_key = None
        _connection.target = "https://app.fastaiagent.net"
        _connection.project_id = None


def test_platform_api_sends_the_header(capture_server) -> None:
    from fastaiagent._platform.api import PlatformAPI

    capture_server.set_response_sequence([(200, json.dumps({"ok": True}))])
    PlatformAPI(api_key="fa_k_test", base_url=capture_server.url).get("/public/v1/anything")
    assert _protocol(capture_server.requests[-1]) == "v1.11"


def test_the_runner_channel_sends_the_header(capture_server) -> None:
    from fastaiagent.runner.channel import RunnerChannel

    capture_server.set_response_sequence(
        [
            (200, json.dumps({"runner_id": "r1", "runner_token": "t1"})),
            (200, json.dumps({"ok": True, "ttl_seconds": 30})),
        ]
    )
    channel = RunnerChannel(capture_server.url, "fa_k_test")

    async def run() -> None:
        await channel.register(capabilities=[])
        await channel.heartbeat(status="idle", active_jobs=0)

    asyncio.run(run())
    register, heartbeat = capture_server.requests[-2:]
    assert _protocol(register) == "v1.11" and _protocol(heartbeat) == "v1.11"
