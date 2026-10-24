"""Cancel a run once it has reached a known point, not after a fixed delay.

A fixed delay raced the run on slow CI runners (Windows): the cancel arrived
before the run had registered, or before its first model call.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


def cancel_when(started: threading.Event, cancel: Callable[[str], Any], execution_id: str) -> None:
    """Call ``cancel(execution_id)`` from another thread once ``started`` is set."""

    def go() -> None:
        if started.wait(timeout=10):
            cancel(execution_id)

    threading.Thread(target=go, daemon=True).start()
