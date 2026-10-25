import threading
import time

import anthropic
import httpx2 as httpx
import openai
import pytest

from cortex.cloud.retry import (
    ServerError,
    StreamCutOffError,
    is_transient,
    retry_delay,
    with_retries,
)
from cortex.tooling.types import TurnInterrupt, TurnInterruptedError

_REQUEST = httpx.Request("POST", "https://example.test/v1/chat/completions")


def _status_error(status, retry_after=None, sdk=openai, headers=None, body=None):
    headers = dict(headers or {})
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    response = httpx.Response(status, headers=headers, request=_REQUEST)
    return sdk.APIStatusError(f"status {status}", response=response, body=body)


@pytest.mark.parametrize("sdk", [openai, anthropic])
def test_transient_failures_are_connection_problems_and_overloaded_servers(sdk):
    assert is_transient(sdk.APIConnectionError(request=_REQUEST))
    assert is_transient(sdk.APITimeoutError(request=_REQUEST))
    for status in (408, 409, 429, 500, 502, 503, 529):
        assert is_transient(_status_error(status, sdk=sdk)), status
    for status in (400, 401, 403, 404, 413, 422):
        assert not is_transient(_status_error(status, sdk=sdk)), status
    assert is_transient(StreamCutOffError("cut"))
    assert is_transient(httpx.ReadError("reset", request=_REQUEST))
    assert not is_transient(ValueError("bad"))


def test_errors_reported_inside_a_stream_are_classified_by_their_body():
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
    invalid = {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
    assert is_transient(_status_error(200, sdk=anthropic, body=overloaded))
    assert not is_transient(_status_error(200, sdk=anthropic, body=invalid))
    assert is_transient(openai.APIError("x", request=_REQUEST, body={"code": 502}))
    assert not is_transient(openai.APIError("x", request=_REQUEST, body={"code": 400}))
    assert is_transient(ServerError("boom", "server_error"))
    assert not is_transient(ServerError("bad prompt", "invalid_prompt"))


def test_delay_follows_retry_after_then_backs_off():
    assert retry_delay(1, _status_error(429, retry_after=7)) == 7
    assert retry_delay(1, _status_error(429, retry_after=600)) == 60
    assert [retry_delay(n, _status_error(503)) for n in (1, 2, 3, 6, 9)] == [1, 2, 4, 30, 30]
    assert retry_delay(2, StreamCutOffError("cut")) == 2
    assert retry_delay(1, _status_error(429, headers={"retry-after-ms": "1500"})) == 1.5
    assert retry_delay(1, _status_error(429, retry_after="nan")) == 1
    assert retry_delay(1, _status_error(429, retry_after="-5")) == 0


def _flaky(failures, events=("a",), outcome="done"):
    """A request that fails with each of `failures` in turn, then succeeds."""
    attempts = []

    def request():
        attempts.append(1)
        if len(attempts) <= len(failures):
            failure = failures[len(attempts) - 1]
            if isinstance(failure, tuple):
                yield failure[0]
                raise failure[1]
            raise failure
        for event in events:
            yield event
        return outcome

    return request, attempts


def _run(request, retries, interrupt=None, retried=None):
    def generator():
        return (yield from with_retries(
            request,
            retries=retries,
            interrupt=interrupt or TurnInterrupt(),
            on_retry=lambda *args: retried.append(args) if retried is not None else None,
        ))

    events = []
    gen = generator()
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


def test_transient_failure_is_sent_again(monkeypatch):
    monkeypatch.setattr("cortex.cloud.retry.retry_delay", lambda attempt, exc: 0)
    request, attempts = _flaky([_status_error(503), StreamCutOffError("cut")])
    retried = []

    events, outcome = _run(request, retries=2, retried=retried)

    assert (events, outcome, len(attempts)) == (["a"], "done", 3)
    assert [args[:2] for args in retried] == [(1, 3), (2, 3)]


def test_retries_stop_at_the_budget(monkeypatch):
    monkeypatch.setattr("cortex.cloud.retry.retry_delay", lambda attempt, exc: 0)
    request, attempts = _flaky([_status_error(503)] * 3)

    with pytest.raises(openai.APIStatusError):
        _run(request, retries=2)
    assert len(attempts) == 3


def test_rejected_request_is_not_sent_again():
    request, attempts = _flaky([_status_error(400)])

    with pytest.raises(openai.APIStatusError):
        _run(request, retries=5)
    assert len(attempts) == 1


def test_request_that_already_streamed_is_not_sent_again():
    request, attempts = _flaky([("partial", StreamCutOffError("cut"))])

    with pytest.raises(StreamCutOffError):
        _run(request, retries=5)
    assert len(attempts) == 1


def test_interrupt_ends_the_wait_before_a_retry():
    interrupt = TurnInterrupt()
    request, attempts = _flaky([_status_error(429, retry_after=30)])
    threading.Timer(0.5, interrupt.set).start()
    started = time.time()

    with pytest.raises(TurnInterruptedError):
        _run(request, retries=2, interrupt=interrupt)
    assert time.time() - started < 5
    assert len(attempts) == 1
