"""Auto-tracing for CrewAI."""

from __future__ import annotations

_enabled = False


def enable() -> None:
    """Enable auto-tracing for CrewAI."""
    global _enabled
    if _enabled:
        return

    try:
        import crewai  # noqa: F401
    except ImportError:
        raise ImportError("CrewAI is required. Install with: pip install fastaiagent[crewai]")

    _enabled = True


def disable() -> None:
    """Disable auto-tracing for CrewAI."""
    global _enabled
    _enabled = False
