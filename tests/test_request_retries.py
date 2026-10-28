"""Each real SDK client sends a request again after a connection reset or a
server error reported inside the stream, and gives the server's message once
its retries run out."""

from __future__ import annotations

import pytest

from cortex.tooling.types import TextDeltaEvent
from tests.provider_server import ProviderServer, make_client

_KINDS = ["chat", "anthropic", "responses"]
_MESSAGES = {"chat": "upstream failed", "anthropic": "Overloaded", "responses": "boom"}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr("cortex.cloud.retry.retry_delay", lambda attempt, exc: 0)


def _reply(kind, script, retries):
    server = ProviderServer(kind, script)
    client = make_client(kind, server.base_url)
    try:
        events = list(
            client.stream_events(
                model_id="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0,
                top_p=1,
                retries=retries,
            )
        )
    finally:
        server.close()
    text = "".join(event.delta for event in events if isinstance(event, TextDeltaEvent))
    return text, server


@pytest.mark.parametrize("failure", ["reset", "error"])
@pytest.mark.parametrize("kind", _KINDS)
def test_request_is_sent_again_after_a_transient_failure(kind, failure):
    text, server = _reply(kind, [{failure: True}, {}], retries=1)

    assert text == "late"
    assert len(server.requests) == 2


@pytest.mark.parametrize("kind", _KINDS)
def test_server_error_message_reaches_the_user_when_retries_run_out(kind):
    with pytest.raises(Exception) as failure:
        _reply(kind, [{"error": True}] * 2, retries=1)

    assert _MESSAGES[kind] in str(failure.value)
