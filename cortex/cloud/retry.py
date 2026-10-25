"""When and how a failed model request is sent again."""

from __future__ import annotations

import math
from typing import Any, Callable, Generator, List, Optional, TypeVar, cast

import anthropic
import openai

from cortex.tooling.types import TurnInterrupt

Outcome = TypeVar("Outcome")
RetryCallback = Callable[[int, int, str], None]


# Errors a server reports for a request that may succeed when sent again: by
# HTTP status, or by the error type or code it sends in the body of a stream
# that already started (with status 200).
_TRANSIENT_STATUS = frozenset({408, 409, 429})
_TRANSIENT_TYPES = frozenset(
    {"overloaded_error", "api_error", "rate_limit_error", "server_error", "rate_limit_exceeded"}
)


class StreamCutOffError(RuntimeError):
    """The response stream ended before the reply was complete."""


class ServerError(RuntimeError):
    """An error the server reported inside a response stream."""

    def __init__(self, message: str, code: object) -> None:
        super().__init__(message)
        self.code = code


def is_transient(exc: BaseException) -> bool:
    """Whether a failed request may succeed when sent again: the connection
    failed, timed out or was cut off, or the server was rate limiting,
    overloaded or failing."""
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return True
    if isinstance(exc, StreamCutOffError) or _is_transport_error(exc):
        return True
    if isinstance(exc, ServerError):
        return _transient_code(exc.code)
    if isinstance(exc, (openai.APIError, anthropic.APIError)):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status >= 400:
            return _transient_code(status)
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        if isinstance(error, dict):
            return _transient_code(error.get("type")) or _transient_code(error.get("code"))
    return False


def _transient_code(code: object) -> bool:
    if isinstance(code, int):
        return code in _TRANSIENT_STATUS or code >= 500
    return code in _TRANSIENT_TYPES


def _is_transport_error(exc: BaseException) -> bool:
    """A connection error the SDK passed on without wrapping it: the Anthropic
    SDK does so for errors while it reads a stream."""
    return any(
        cls.__name__ == "TransportError" and cls.__module__.split(".")[0] in ("httpx", "httpx2")
        for cls in type(exc).__mro__
    )


def retry_delay(attempt: int, exc: BaseException) -> float:
    """Seconds to wait before retry number `attempt`: what the server asks for
    in Retry-After or retry-after-ms (at most 60), otherwise 1, 2, 4, ... up
    to 30."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            seconds = float(str(headers.get(name))) * scale
        except ValueError:
            continue
        if math.isfinite(seconds):
            return min(max(seconds, 0.0), 60.0)
    return min(2.0 ** (attempt - 1), 30.0)


def with_retries(
    request: Callable[[], Generator[Any, None, Outcome]],
    *,
    retries: int,
    interrupt: TurnInterrupt,
    on_retry: Optional[RetryCallback] = None,
) -> Generator[Any, None, Outcome]:
    """Send a request and yield its events, sending it again after a transient
    failure, up to `retries` times. `request` starts the request and yields
    the events of its response; its return value is the request's outcome. A
    request that already yielded events is not sent again: they have reached
    the user."""
    attempt = 0
    while True:
        emitted = [False]
        try:
            return (yield from _forward(request(), emitted))
        except Exception as exc:
            if emitted[0] or attempt >= retries or interrupt.is_set() or not is_transient(exc):
                raise
            attempt += 1
            if on_retry is not None:
                on_retry(attempt, retries + 1, str(exc))
            interrupt.wait(retry_delay(attempt, exc))
            interrupt.raise_if_set()


def _forward(
    events: Generator[Any, None, Outcome], emitted: List[bool]
) -> Generator[Any, None, Outcome]:
    try:
        while True:
            try:
                event = next(events)
            except StopIteration as stop:
                return cast(Outcome, stop.value)
            emitted[0] = True
            yield event
    finally:
        events.close()
