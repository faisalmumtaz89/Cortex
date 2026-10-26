from types import SimpleNamespace

import pytest

from cortex.cloud.router import CloudRouter
from cortex.cloud.types import CloudModelRef, CloudProvider
from cortex.tooling.types import TextDeltaEvent, TurnInterrupt, TurnInterruptedError


class _DummyCredentials:
    def __init__(self, key=None, source=None):
        self.key = key
        self.source = source

    def get_api_key_with_source(self, _provider):
        return self.key, self.source


def _make_router(credentials):
    config = SimpleNamespace(
        cloud=SimpleNamespace(
            cloud_timeout_seconds=30,
            cloud_max_retries=3,
        )
    )
    return CloudRouter(config=config, credential_store=credentials)


def _events(router, **kwargs):
    return list(
        router.stream_events(
            model_ref=CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.5"),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0.2,
            top_p=1.0,
            **kwargs,
        )
    )


def test_get_auth_status_reports_source():
    router = _make_router(_DummyCredentials(key="abc", source="env"))
    is_auth, source = router.get_auth_status(CloudProvider.OPENAI)
    assert is_auth is True
    assert source == "env"


def test_stream_raises_when_credentials_missing():
    router = _make_router(_DummyCredentials(key=None, source=None))

    with pytest.raises(RuntimeError) as exc:
        _events(router)

    assert "No API key configured" in str(exc.value)


def test_client_gets_the_retry_budget_and_the_turn_interrupt():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))
    seen = {}

    class _Client:
        def stream_events(self, **kwargs):
            seen.update(kwargs)
            yield TextDeltaEvent(delta="hello")

    router._build_client = lambda _provider, _api_key: _Client()
    interrupt = TurnInterrupt()

    events = _events(router, interrupt=interrupt)

    assert [event.delta for event in events] == ["hello"]
    assert seen["retries"] == 3
    assert seen["interrupt"] is interrupt


def test_failure_names_the_model_and_key_source():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))

    class _Client:
        def stream_events(self, **_kwargs):
            raise ValueError("boom")
            yield

    router._build_client = lambda _provider, _api_key: _Client()

    with pytest.raises(RuntimeError) as exc:
        _events(router)

    assert str(exc.value) == (
        "Cloud generation failed for openai:gpt-5.5 (key saved with /login): boom"
    )


def test_failure_after_an_interrupt_reports_the_interrupt():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))
    interrupt = TurnInterrupt()

    class _Client:
        def stream_events(self, **_kwargs):
            interrupt.set()
            raise ConnectionError("socket shut down")
            yield

    router._build_client = lambda _provider, _api_key: _Client()

    with pytest.raises(TurnInterruptedError):
        _events(router, interrupt=interrupt)
