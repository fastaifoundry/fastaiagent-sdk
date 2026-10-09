"""FastAPI construction for the SDK's own servers (the Local UI, ``agent serve``).

FastAPI 0.143 traces every request by default and sends the spans to the
*global* OpenTelemetry tracer provider — which is fastaiagent's own once the SDK
has traced anything. The Local UI therefore recorded each of its own API calls
into ``local.db`` as a ``GET`` trace, interleaved with the user's agent runs.

The SDK's servers opt out of FastAPI's built-in telemetry. Detected from the
installed FastAPI's signature, so a FastAPI without the ``telemetry`` setting is
constructed exactly as before. A user's *own* FastAPI app is never touched:
whether its requests are traced is the user's choice.
"""

from __future__ import annotations

import inspect
from typing import Any

#: Every FastAPI telemetry signal off.
_TELEMETRY_OFF: dict[str, Any] = {
    "tracing": False,
    "metrics": False,
    "logs": False,
    "operation_spans": False,
}


def sdk_app_kwargs() -> dict[str, Any]:
    """Extra ``FastAPI(...)`` kwargs for an SDK-owned server: telemetry off when supported."""
    from fastapi import FastAPI

    try:
        params = inspect.signature(FastAPI.__init__).parameters
    except (TypeError, ValueError):
        return {}
    if "telemetry" not in params:
        return {}
    return {"telemetry": dict(_TELEMETRY_OFF)}
