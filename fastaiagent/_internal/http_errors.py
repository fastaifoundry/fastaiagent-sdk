"""What counts as a transient transport failure on an HTTP call (1.91.0).

``httpx`` turns ``httpcore``'s network errors into its own (``httpx.ConnectError``
and the rest, all ``httpx.TransportError``) through a map of ``httpcore``
classes it builds once. When something re-imports ``httpcore`` in the same
process — notebook autoreload, a test runner, a plugin loader — the classes it
then raises are new objects that map does not hold, and the raw
``httpcore.ConnectError`` escapes ``httpx`` unmapped. Measured in CI on
2026-10-21: a refused connection reached ``LLMClient`` as ``httpcore.ConnectError``
(its class a different object from the one in httpx's map), so ``max_retries``
and ``fallbacks=`` never applied, and the live fallback test failed.

So the check is by **name**, not by class object: a class imported here before a
reload would be stale in exactly the case this exists for. An error is a
transport failure when it is an ``httpx.TransportError`` or a ``TimeoutError``,
or when any class in its MRO is ``httpcore``'s ``NetworkError``,
``TimeoutException`` or ``ProtocolError`` — whichever import of ``httpcore``
raised it.
"""

from __future__ import annotations

import httpx

#: ``httpcore``'s base classes for a network, timeout or protocol failure.
_HTTPCORE_BASES = frozenset({"NetworkError", "TimeoutException", "ProtocolError"})


def is_transport_error(error: BaseException) -> bool:
    """A transient network failure, as ``httpx`` or as raw ``httpcore`` raises it."""
    if isinstance(error, (httpx.TransportError, TimeoutError)):
        return True
    return any(
        cls.__module__.split(".")[0] == "httpcore" and cls.__name__ in _HTTPCORE_BASES
        for cls in type(error).__mro__
    )
