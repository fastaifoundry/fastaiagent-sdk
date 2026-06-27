"""Example 88 — Connected governance: enrollment + opt-in fail-closed (WS4).

When ``connect()``-ed to an Enterprise control plane, the SDK does two governance
things beyond the managed approval gate (see example 84):

1. **Enrollment** — it attests its presence + posture to the plane
   (``POST /public/v1/governance/enroll``) with a stable per-install
   ``instance_id``, so the console coverage dashboard can tell *enrolled &
   reporting* agents from **dark** ones. It's fire-and-forget — never blocks or
   raises ``connect()``; the plane upserts on ``(domain, project, instance_id)``.
2. **Opt-in fail-closed** — ``governance_fail_mode="closed"`` makes a governed
   agent refuse tool calls when governance can't be confirmed (the plane was
   unreachable at connect, so no policy is cached), instead of running
   ungoverned. The default stays ``"open"`` (non-breaking).

This demo enrolls (showing the idempotent upsert) and illustrates the fail-closed
gate decision. See docs/platform/connected-governance.md for the coverage view.

Usage:
    export FASTAIAGENT_API_KEY=fa_k_...          # a connected_state_plane domain key
    export FASTAIAGENT_TARGET=http://localhost:20001
    python examples/88_connected_governance.py

Expected output (snapshot — real run against a local plane on :20001):
    connected to http://localhost:20001 (governance_fail_mode=closed)
    enrolled: instance_id=<hex> fail_mode=closed
      first_seen=2026-06-27T...Z last_seen=2026-06-27T...Z
    re-enroll (upsert): first_seen preserved=True last_seen refreshed=True
    fail-closed gate: refused -> 'Refused: fail-closed mode — governance unavailable for this run'
    (default 'open' would allow the same call: allowed=True)
    done — posture attested; coverage dashboard now shows this instance.
"""

from __future__ import annotations

import os

import fastaiagent as fa
from fastaiagent import governance
from fastaiagent._internal.async_utils import run_sync
from fastaiagent.client import _connection


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    if not api_key:
        print("Skipping: set FASTAIAGENT_API_KEY + FASTAIAGENT_TARGET (connected_state_plane key)")
        return 1

    # Opt in to fail-closed so the attested posture is observable.
    fa.connect(api_key=api_key, target=target, governance_fail_mode="closed")
    print(f"connected to {target} (governance_fail_mode={_connection.governance_fail_mode})")
    try:
        # 1) Enroll (connect() already kicked one in the background; call it
        #    explicitly here so we can print the response).
        first = governance.enroll()
        if first is None:
            print(
                "Skipping: enroll returned nothing — domain may not be entitled "
                "(connected_state_plane) or the plane is unreachable."
            )
            return 1
        print(f"enrolled: instance_id={first['instance_id']} fail_mode={first['fail_mode']}")
        print(f"  first_seen={first['first_seen_at']} last_seen={first['last_seen_at']}")

        # 2) Re-enroll — proves the idempotent upsert (same instance_id).
        second = governance.enroll()
        print(
            f"re-enroll (upsert): first_seen preserved="
            f"{second['first_seen_at'] == first['first_seen_at']} "
            f"last_seen refreshed={second['last_seen_at'] >= first['last_seen_at']}"
        )

        # 3) Illustrate the fail-closed gate decision. Simulate "plane was
        #    unreachable at connect" by clearing the cached policy: with
        #    fail_mode='closed' a governed agent's tool call is REFUSED rather
        #    than run ungoverned. (The default 'open' would allow it.)
        _connection.policy_cache = None
        refusal = run_sync(
            governance.gate_tool_call(
                "transfer_funds", {"amount": 500}, agent_id="demo-agent", run_id="demo-run"
            )
        )
        print(f"fail-closed gate: refused -> {refusal!r}")

        _connection.governance_fail_mode = "open"
        allowed = run_sync(
            governance.gate_tool_call(
                "transfer_funds", {"amount": 500}, agent_id="demo-agent", run_id="demo-run"
            )
        )
        print(f"(default 'open' would allow the same call: allowed={allowed is None})")
        print("done — posture attested; coverage dashboard now shows this instance.")
        return 0
    finally:
        fa.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
