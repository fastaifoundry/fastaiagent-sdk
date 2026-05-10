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


# ---------------------------------------------------------------------------
# Generic per-key rate limiter (security_review_1.md M5).
#
# Uses the same sliding-window primitive as ``LoginThrottler`` but tracks
# *every* request (not just failures) so we can cap how many LLM calls a
# session may make per minute. The Local UI is single-user, but the LLM
# spend matters (a runaway agent or a debugging fat-finger can burn dollars
# in seconds), so a polite ceiling is cheap insurance.
# ---------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, *, limit: int, window_seconds: float):
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._limit = limit
        self._window = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def _now(self) -> float:
        return time.monotonic()

    def try_acquire(self, key: str) -> tuple[bool, float]:
        """Record a hit if the key is under the limit.

        Returns ``(allowed, retry_after_seconds)``. When ``allowed`` is
        ``True`` the caller proceeds with the request. When ``False`` the
        caller should return HTTP 429 with the ``retry_after_seconds``
        suggested in a ``Retry-After`` header.
        """
        with self._lock:
            now = self._now()
            window_start = now - self._window
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self._limit:
                # Oldest hit will roll out at hits[0] + window.
                retry_after = hits[0] + self._window - now
                return False, max(retry_after, 0.0)
            hits.append(now)
            return True, 0.0

    def reset(self) -> None:
        """Test helper — clear every key."""
        with self._lock:
            self._hits.clear()


# 30 LLM calls / minute per session. Tuned for a human iterating in the
# playground (way more than enough) but tight enough that a runaway loop
# is bounded to ≤ $a-few before the user notices.
_default_llm_limiter = RateLimiter(limit=30, window_seconds=60.0)


def get_llm_rate_limiter() -> RateLimiter:
    """Module-level singleton used by the playground LLM endpoints."""
    return _default_llm_limiter


__all__ = [
    "LoginThrottler",
    "RateLimiter",
    "get_default_throttler",
    "get_llm_rate_limiter",
]
