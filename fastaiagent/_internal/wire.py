"""The plane wire version this SDK understands (1.94.0).

Sent as ``X-FAA-Protocol`` on every call to the plane. The plane uses it for
telemetry and feature-gating; new keys on the wire are additive and reach every
client, so the header announces what the SDK reads rather than unlocking it.

v1.11 — the monthly trace limit: ``/public/v1/traces/ingest`` may answer
``limit_reached`` with ``resets_at``, and trace export pauses until then (S7).
"""

from __future__ import annotations

PLANE_WIRE_PROTOCOL = "v1.11"
PROTOCOL_HEADER = "X-FAA-Protocol"
