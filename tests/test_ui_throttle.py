"""Unit tests for the login throttler.

The login route is exercised end-to-end in ``tests/test_ui_server.py``;
this file isolates the throttler so the time-window logic is testable
without spinning a FastAPI app per case.
"""

from __future__ import annotations

import pytest

from fastaiagent.ui.throttle import LoginThrottler


class _ManualClock:
    """Drop-in replacement for ``time.monotonic`` used inside the throttler."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clocked_throttler(monkeypatch) -> tuple[LoginThrottler, _ManualClock]:
    t = LoginThrottler()
    clock = _ManualClock()
    monkeypatch.setattr(t, "_now", clock)
    return t, clock


class TestLoginThrottler:
    def test_zero_when_no_history(self, clocked_throttler):
        t, _ = clocked_throttler
        assert t.check("alice|1.2.3.4") == 0.0

    def test_below_threshold_no_lockout(self, clocked_throttler):
        t, _ = clocked_throttler
        for _ in range(4):
            cooldown = t.record_failure("alice|1.2.3.4")
            assert cooldown == 0.0
        assert t.check("alice|1.2.3.4") == 0.0

    def test_threshold_triggers_cooldown(self, clocked_throttler):
        t, _ = clocked_throttler
        # 5th failure within the window arms the cool-down.
        for _ in range(4):
            t.record_failure("alice|1.2.3.4")
        cooldown = t.record_failure("alice|1.2.3.4")
        assert cooldown >= 60.0
        assert t.check("alice|1.2.3.4") > 0

    def test_cooldown_doubles_with_extra_failures(self, clocked_throttler):
        t, _ = clocked_throttler
        for _ in range(5):
            t.record_failure("alice|1.2.3.4")
        first = t.check("alice|1.2.3.4")
        t.record_failure("alice|1.2.3.4")
        second = t.check("alice|1.2.3.4")
        assert second > first
        assert second <= 3600.0  # capped at 1 hour

    def test_window_eviction_resets_count(self, clocked_throttler):
        t, clock = clocked_throttler
        for _ in range(4):
            t.record_failure("alice|1.2.3.4")
        # Advance past the 5 min sliding window — the existing failures
        # roll out, so we are back to "no failures recorded".
        clock.advance(301.0)
        # Recording again should not arm a cool-down on this single new fail.
        cooldown = t.record_failure("alice|1.2.3.4")
        assert cooldown == 0.0

    def test_success_clears_state(self, clocked_throttler):
        t, _ = clocked_throttler
        for _ in range(5):
            t.record_failure("alice|1.2.3.4")
        assert t.check("alice|1.2.3.4") > 0
        t.record_success("alice|1.2.3.4")
        assert t.check("alice|1.2.3.4") == 0.0

    def test_keys_isolated(self, clocked_throttler):
        t, _ = clocked_throttler
        for _ in range(5):
            t.record_failure("alice|1.2.3.4")
        # Different IP for the same user is its own bucket.
        assert t.check("alice|9.9.9.9") == 0.0


class TestClientThrottleIp:
    """security_audit_2 N12 — throttle keys use the real peer IP, not a
    spoofable X-Forwarded-For, unless a trusted proxy is declared."""

    @staticmethod
    def _req(xff=None, peer="203.0.113.9"):
        from types import SimpleNamespace

        headers = {"x-forwarded-for": xff} if xff else {}
        client = SimpleNamespace(host=peer) if peer is not None else None
        return SimpleNamespace(headers=headers, client=client)

    def test_default_ignores_forwarded_for(self, monkeypatch):
        from fastaiagent.ui.throttle import client_throttle_ip

        monkeypatch.delenv("FASTAIAGENT_UI_TRUST_PROXY", raising=False)
        # Spoofed header is ignored; the real peer wins.
        assert client_throttle_ip(self._req(xff="1.2.3.4", peer="203.0.113.9")) == "203.0.113.9"
        assert client_throttle_ip(self._req(xff="9.9.9.9", peer="203.0.113.9")) == "203.0.113.9"

    def test_trust_proxy_honors_first_hop(self, monkeypatch):
        from fastaiagent.ui.throttle import client_throttle_ip

        monkeypatch.setenv("FASTAIAGENT_UI_TRUST_PROXY", "1")
        assert client_throttle_ip(self._req(xff="1.2.3.4, 10.0.0.1", peer="10.0.0.1")) == "1.2.3.4"
        # No header even under trust-proxy → peer.
        assert client_throttle_ip(self._req(xff=None, peer="10.0.0.1")) == "10.0.0.1"

    def test_missing_client_is_unknown(self, monkeypatch):
        from fastaiagent.ui.throttle import client_throttle_ip

        monkeypatch.delenv("FASTAIAGENT_UI_TRUST_PROXY", raising=False)
        assert client_throttle_ip(self._req(peer=None)) == "unknown"
