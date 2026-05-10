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
