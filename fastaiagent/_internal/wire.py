"""The plane wire version this SDK understands (1.102.0).

Sent as ``X-FAA-Protocol`` on every call to the plane. The plane uses it for
telemetry and feature-gating; new keys on the wire are additive and reach every
client, so the header announces what the SDK reads rather than unlocking it.

v1.11 — the monthly trace limit: ``/public/v1/traces/ingest`` may answer
``limit_reached`` with ``resets_at``, and trace export pauses until then (S7).

v1.12 — headless Agent Studio: the ``/public/v1/agent-studio`` routes that
``fastaiagent.studio`` and ``fastaiagent studio`` call (plane 0.20.0). A plane
older than that has no such routes; the Studio client says so instead of
returning a bare 404. A plane that does not know v1.12 serves its own latest
minor (it clamps a newer client down), so the header is safe to send to it.
"""

from __future__ import annotations

PLANE_WIRE_PROTOCOL = "v1.12"
PROTOCOL_HEADER = "X-FAA-Protocol"
