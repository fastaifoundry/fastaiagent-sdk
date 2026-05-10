"""In-memory login throttler.

Used by ``/api/auth/login`` to slow down brute-force guesses without
locking out a legitimate user who is just fat-fingering their own
password. Holds state in process memory; the local UI is single-process
single-user, so a Redis/DB-backed limiter would be overkill.

Policy
------
* Track failed attempts per ``(client_ip, username)`` in a sliding 5 min
  window.
* After 5 failures within the window, return HTTP 429 for a 60 s
  cool-down.
* Each *additional* failure during cool-down doubles the cool-down,
  capped at 1 hour.
* Any successful login clears state for that key.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

# Sliding-window size and base thresholds. Tuned for "humans typing
# passwords occasionally make mistakes" not "bots". A bot hitting 5×
# wrong-password in 5 min is the only scenario that triggers a lockout.
_WINDOW_SECONDS: float = 300.0
_MAX_FAILS_PER_WINDOW: int = 5
_BASE_COOLDOWN_SECONDS: float = 60.0
_MAX_COOLDOWN_SECONDS: float = 3600.0


@dataclass
class _ThrottleState:
    failures: deque[float] = field(default_factory=deque)
    locked_until: float = 0.0


class LoginThrottler:
    """Thread-safe sliding-window throttler keyed by an opaque string."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, _ThrottleState] = {}

    def _now(self) -> float:
        return time.monotonic()

    def check(self, key: str) -> float:
        """Return seconds remaining on the lockout, or ``0.0`` if allowed.

        Never mutates state. Callers should still call
        :meth:`record_failure` if the password check fails after this
        returned ``0.0``.
        """
        with self._lock:
            st = self._state.get(key)
            if st is None:
                return 0.0
            now = self._now()
            self._evict(st, now)
            return max(0.0, st.locked_until - now)

    def record_failure(self, key: str) -> float:
        """Record one failure. Return the new cool-down remaining (``0.0``
        if still under the threshold).
        """
        with self._lock:
            st = self._state.setdefault(key, _ThrottleState())
            now = self._now()
            self._evict(st, now)
            st.failures.append(now)
            if len(st.failures) >= _MAX_FAILS_PER_WINDOW:
                excess = len(st.failures) - _MAX_FAILS_PER_WINDOW
                cooldown = min(
                    _BASE_COOLDOWN_SECONDS * (2 ** excess),
                    _MAX_COOLDOWN_SECONDS,
                )
                # Push the lockout out; never shorten an existing one.
                st.locked_until = max(st.locked_until, now + cooldown)
                return st.locked_until - now
            return 0.0

    def record_success(self, key: str) -> None:
        """Clear all throttling state for ``key``."""
        with self._lock:
            self._state.pop(key, None)

    def reset(self) -> None:
        """Test helper — clear every key."""
        with self._lock:
            self._state.clear()

    @staticmethod
    def _evict(st: _ThrottleState, now: float) -> None:
        cutoff = now - _WINDOW_SECONDS
        while st.failures and st.failures[0] < cutoff:
            st.failures.popleft()


_default = LoginThrottler()


def get_default_throttler() -> LoginThrottler:
    """Module-level singleton used by the login route."""
    return _default


__all__ = ["LoginThrottler", "get_default_throttler"]
