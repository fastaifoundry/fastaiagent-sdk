"""Keeping an auto-traced provider span open for the life of its stream (1.89.0).

The OpenAI and Anthropic auto-tracers wrap ``create``. With ``stream=True``,
``create`` returns as soon as the response headers arrive, and the stream is
read afterwards. The span used to close when ``create`` returned, so it ended
OK with no usage and no content, and an error raised while the caller iterated
(a dropped connection, a provider ``error`` event part-way) happened outside
any span: the failure was invisible in the trace.

Now the span stays open until the stream is exhausted, fails, or is closed:

* **exhausted**: usage, content, tool calls and finish reason are recorded
  from the chunks (OpenAI sends usage only with
  ``stream_options={"include_usage": True}``; Anthropic sends it on
  ``message_start`` / ``message_delta``), and the span ends OK;
* **failed**: the span ends ERROR with the exception recorded, its text scrubbed
  of credentials (the P9 rule: an exception message is copied into span events,
  the status description, and from there to every exporter), and the exception
  is re-raised unchanged;
* **closed or abandoned** before the end: the span ends without an error, with
  what was read so far.

The provider's own ``Stream`` / ``AsyncStream`` object is returned, not a
proxy, so ``isinstance`` checks and the SDKs' stream helpers keep working: only
its private ``_iterator`` is wrapped. A stream object without one (a future SDK
shape) ends the span when ``create`` returns, as before.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterable
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class StreamObserver(Protocol):
    def observe(self, chunk: Any) -> None: ...

    def apply(self, span: Any) -> None: ...


def client_secrets(resource: Any) -> list[str]:
    """Credentials held by the SDK client behind a resource (``self._client``)."""
    client = getattr(resource, "_client", None)
    out: list[str] = []
    for attr in ("api_key", "auth_token"):
        value = getattr(client, attr, None)
        if isinstance(value, str) and value:
            out.append(value)
    return out


def mark_error_scrubbed(span: Any, exc: BaseException, secrets: Iterable[str] = ()) -> None:
    """Record ``exc`` on ``span`` and set it ERROR, with credentials scrubbed.

    ``span.record_exception`` copies ``str(exc)`` and the formatted traceback into
    the event verbatim, and a provider error can echo the key it rejected. The
    event's message and stack trace are replaced with scrubbed copies, and the
    status description is scrubbed too. Never raises.
    """
    try:
        if not span.is_recording():
            return
        from opentelemetry.trace import Status, StatusCode

        from fastaiagent._internal.scrub import known_credential_values, scrub_secrets

        known = [*known_credential_values(), *secrets]
        message = scrub_secrets(str(exc), known)
        stack = scrub_secrets(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), known
        )
        span.record_exception(
            exc, attributes={"exception.message": message, "exception.stacktrace": stack}
        )
        span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {message}"))
    except Exception:  # pragma: no cover - bookkeeping never breaks the caller
        logger.debug("Could not mark auto-traced span as errored", exc_info=True)


class _SpanEnder:
    """Ends the span once, applying what the observer saw."""

    def __init__(self, span: Any, observer: StreamObserver, secrets: list[str]) -> None:
        self._span = span
        self._observer = observer
        self._secrets = secrets
        self._done = False

    def observe(self, chunk: Any) -> None:
        try:
            self._observer.observe(chunk)
        except Exception:  # pragma: no cover - an odd chunk never breaks the stream
            logger.debug("Could not read a streamed chunk for the trace", exc_info=True)

    def finish(self, exc: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._observer.apply(self._span)
        except Exception:  # pragma: no cover
            logger.debug("Could not record streamed response on span", exc_info=True)
        if exc is not None:
            mark_error_scrubbed(self._span, exc, self._secrets)
        try:
            self._span.end()
        except Exception:  # pragma: no cover
            logger.debug("Could not end auto-traced stream span", exc_info=True)


class _TracedIterator:
    def __init__(self, inner: Any, ender: _SpanEnder) -> None:
        self._inner = inner
        self._ender = ender

    def __iter__(self) -> _TracedIterator:
        return self

    def __next__(self) -> Any:
        try:
            item = self._inner.__next__()
        except StopIteration:
            self._ender.finish()
            raise
        except Exception as exc:
            self._ender.finish(exc)
            raise
        except BaseException:
            # GeneratorExit / KeyboardInterrupt: an abandoned stream, not a failure.
            self._ender.finish()
            raise
        self._ender.observe(item)
        return item

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._ender.finish()

    def __del__(self) -> None:
        self._ender.finish()


class _TracedAsyncIterator:
    def __init__(self, inner: Any, ender: _SpanEnder) -> None:
        self._inner = inner
        self._ender = ender

    def __aiter__(self) -> _TracedAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._inner.__anext__()
        except StopAsyncIteration:
            self._ender.finish()
            raise
        except Exception as exc:
            self._ender.finish(exc)
            raise
        except BaseException:
            # CancelledError: a cancelled stream is not a failure.
            self._ender.finish()
            raise
        self._ender.observe(item)
        return item

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        try:
            if aclose is not None:
                await aclose()
        finally:
            self._ender.finish()

    def __del__(self) -> None:
        self._ender.finish()


def wrap_stream(
    stream: Any, span: Any, observer: StreamObserver, secrets: list[str], *, is_async: bool
) -> bool:
    """Make ``span`` live as long as ``stream``; ``False`` when it cannot be wrapped.

    Wraps the stream's ``_iterator`` (what its ``__iter__`` / ``__next__`` read)
    and its ``close``, so the span ends on exhaustion, error or close. The
    caller ends the span itself when this returns ``False``.
    """
    inner = getattr(stream, "_iterator", None)
    if inner is None:
        return False
    ender = _SpanEnder(span, observer, secrets)
    try:
        if is_async:
            stream._iterator = _TracedAsyncIterator(inner, ender)
        else:
            stream._iterator = _TracedIterator(inner, ender)
    except Exception:
        logger.debug("Could not wrap the provider stream for tracing", exc_info=True)
        return False
    original_close = getattr(stream, "close", None)
    if original_close is not None:
        try:
            if is_async:

                async def aclose() -> Any:
                    try:
                        return await original_close()
                    finally:
                        ender.finish()

                stream.close = aclose
            else:

                def close() -> Any:
                    try:
                        return original_close()
                    finally:
                        ender.finish()

                stream.close = close
        except Exception:  # pragma: no cover - a slotted stream keeps its close
            logger.debug("Could not wrap the provider stream's close", exc_info=True)
    return True


def use_span(span: Any) -> Any:
    """Make ``span`` current for the ``create`` call without ending it or
    recording exceptions (those are recorded scrubbed by the caller)."""
    from opentelemetry import trace

    return trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )
