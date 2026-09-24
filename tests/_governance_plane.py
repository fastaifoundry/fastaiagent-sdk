"""A local HTTP stand-in for the plane's governance surface, shared by the tests.

It implements the frozen wire the SDK talks to for a policy-gated tool call —
``/policy``, ``/policy/decide``, ``/runs/{id}/pending`` and ``/hitl/events`` —
and records everything it receives, so a test asserts on what actually crossed
the wire.

Like the real plane since 2026-09-23, it **never decides** a pending run: a
posted pause stays ``pending`` forever, and its id is ``pr-<run_id>``. It still
counts ``GET /runs/{id}/pending`` so a test can pin that the SDK never polls.
``fail_pending_post`` makes registration fail, for the ``pending_id: null`` case.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The platform agent id every governed test agent is enrolled under.
AGENT_ID = "agent-banker-1"
#: The one tool the cached approval policy covers.
GATED_TOOL = "transfer_funds"


class GovPlane:
    """What the stand-in received, plus the one knob a test may turn."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.decide_calls: list[dict[str, Any]] = []
        self.pending_posts: list[dict[str, Any]] = []
        self.pending_polls = 0
        # Keyed by event_id: the SDK's drain re-sends until it sees a 2xx, and
        # two drains can overlap, so the same event may arrive more than once.
        self.hitl_events: dict[str, dict[str, Any]] = {}
        # True: ``POST /runs/{id}/pending`` answers 500, so the pause has no
        # pending run on the plane.
        self.fail_pending_post = False
        # What ``/policy/decide`` answers for the gated tool: ``require_approval``
        # (the default), ``deny``, ``allow``, or ``error`` for a 500.
        self.gated_decision = "require_approval"

    def ledger(self, run_id: str) -> list[dict[str, Any]]:
        """The HITL events received for one run, oldest first."""
        with self.lock:
            rows = [e for e in self.hitl_events.values() if e.get("run_id") == run_id]
        return sorted(rows, key=lambda e: (e.get("occurred_at") or "", e["event_type"]))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any, **k: Any) -> None:
        pass

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0) or 0)
        data: dict[str, Any] = json.loads(self.rfile.read(n) or b"{}")
        return data

    def _run_id(self) -> str:
        # /public/v1/runs/{run_id}/pending
        return self.path.split("/runs/", 1)[1].rsplit("/pending", 1)[0]

    def do_GET(self) -> None:  # noqa: N802
        st: GovPlane = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/auth/check":
            self._json(
                200,
                {
                    "ok": True,
                    "domain_id": "dom-1",
                    "project_id": "proj-1",
                    "scopes": ["policy:read", "policy:decide", "run:write", "run:read"],
                },
            )
        elif self.path == "/public/v1/policy":
            self._json(
                200,
                {
                    "version": "gov-test-v1",
                    "guardrail_rules": [],
                    "approval_policies": [
                        {
                            "id": "ap-1",
                            "name": "transfer-approval",
                            "agent_id": None,
                            "tool_pattern": GATED_TOOL,
                            "condition_type": "always",
                            "condition_config": None,
                            "timeout_minutes": 60,
                        }
                    ],
                },
            )
        elif self.path.endswith("/pending"):
            run_id = self._run_id()
            with st.lock:
                st.pending_polls += 1
            self._json(200, {"pending_id": f"pr-{run_id}", "status": "pending"})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        st: GovPlane = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/policy/decide":
            body = self._body()
            with st.lock:
                st.decide_calls.append(body)
                verdict = st.gated_decision
            if body.get("tool_name") == GATED_TOOL and verdict == "error":
                self._json(500, {"detail": "decide failed (test)"})
            elif body.get("tool_name") == GATED_TOOL and verdict == "deny":
                self._json(
                    200,
                    {
                        "decision": "deny",
                        "approval_request_id": None,
                        "reason": "transfers are blocked for this agent",
                    },
                )
            elif body.get("tool_name") == GATED_TOOL and verdict == "require_approval":
                self._json(
                    200,
                    {
                        "decision": "require_approval",
                        "approval_request_id": "apr-1",
                        "reason": "Matched approval policy 'transfer-approval'",
                    },
                )
            else:
                self._json(200, {"decision": "allow", "approval_request_id": None, "reason": None})
        elif self.path.endswith("/pending"):
            run_id = self._run_id()
            with st.lock:
                st.pending_posts.append({"run_id": run_id, "body": self._body()})
                failing = st.fail_pending_post
            if failing:
                self._json(500, {"detail": "registration failed (test)"})
            else:
                self._json(201, {"pending_id": f"pr-{run_id}", "status": "pending"})
        elif self.path == "/public/v1/hitl/events":
            events = self._body().get("events", [])
            with st.lock:
                for e in events:
                    st.hitl_events[e["event_id"]] = e
            self._json(201, {"ingested": len(events), "rejected": 0, "rejections": []})
        elif self.path == "/public/v1/traces/ingest":
            self._json(201, {"ingested": len(self._body().get("spans", []))})
        else:
            self._json(404, {"detail": "not found"})


@contextmanager
def serve() -> Iterator[tuple[GovPlane, str]]:
    """Run the stand-in on an ephemeral localhost port."""
    state = GovPlane()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield state, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def reset_connection() -> None:
    """Undo ``connect()`` so the next test starts disconnected with no policy."""
    import fastaiagent
    from fastaiagent.client import _connection
    from fastaiagent.trace import otel
    from fastaiagent.trace.hitl_export import get_hitl_exporter

    try:
        fastaiagent.disconnect()
    except Exception:
        pass
    otel.reset()
    # The drain caches the store it opened; the next test has its own local.db.
    get_hitl_exporter().shutdown()
    for attr, val in (
        ("api_key", None),
        ("target", "https://app.fastaiagent.net"),
        ("project", None),
        ("project_id", None),
        ("domain_id", None),
        ("policy_cache", None),
        ("governance_fail_mode", "open"),
        ("_platform_processor", None),
    ):
        setattr(_connection, attr, val)
