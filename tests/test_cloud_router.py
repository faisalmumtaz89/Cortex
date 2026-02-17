import time
from types import SimpleNamespace

import pytest

from cortex.cloud.router import CloudRouter
from cortex.cloud.types import CloudModelRef, CloudProvider


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
            cloud_max_retries=1,
        )
    )
    return CloudRouter(config=config, credential_store=credentials)


def test_get_auth_status_reports_source():
    router = _make_router(_DummyCredentials(key="abc", source="env"))
    is_auth, source = router.get_auth_status(CloudProvider.OPENAI)
    assert is_auth is True
    assert source == "env"


def test_stream_retries_once_when_no_output_emitted():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))
    attempts = {"count": 0}

    class _FlakyClient:
        def validate_key(self):
            return True, "ok"

        def stream(self, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary failure")
            yield "hello"
            yield " world"

    def _factory(_provider, _api_key):
        return _FlakyClient()

    router._build_client = _factory

    model_ref = CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")
    output = "".join(
        router.stream(
            model_ref=model_ref,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0.2,
            top_p=1.0,
        )
    )

    assert attempts["count"] == 2
    assert output == "hello world"


def test_stream_raises_when_credentials_missing():
    router = _make_router(_DummyCredentials(key=None, source=None))
    model_ref = CloudModelRef(provider=CloudProvider.ANTHROPIC, model_id="claude-sonnet-4-5")

    with pytest.raises(RuntimeError) as exc:
        list(
            router.stream(
                model_ref=model_ref,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=32,
                temperature=0.2,
                top_p=1.0,
            )
        )

    assert "No API key configured" in str(exc.value)


def test_stream_times_out_when_client_stalls():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))
    router.config.cloud.cloud_timeout_seconds = 1
    router.config.cloud.cloud_max_retries = 0

    class _HangingClient:
        def validate_key(self):
            return True, "ok"

        def stream(self, **_kwargs):
            while True:
                time.sleep(3600)
                yield ""

    router._build_client = lambda _provider, _api_key: _HangingClient()

    model_ref = CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")
    with pytest.raises(RuntimeError) as exc:
        list(
            router.stream(
                model_ref=model_ref,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=32,
                temperature=0.2,
                top_p=1.0,
            )
        )

    assert "Cloud generation failed" in str(exc.value)


def test_stream_uses_non_stream_fallback_after_timeout():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))
    router.config.cloud.cloud_timeout_seconds = 1
    router.config.cloud.cloud_max_retries = 0

    class _TimeoutThenFallbackClient:
        def validate_key(self):
            return True, "ok"

        def stream(self, **_kwargs):
            while True:
                time.sleep(3600)
                yield ""

        def generate_once(self, **_kwargs):
            return "fallback response"

    router._build_client = lambda _provider, _api_key: _TimeoutThenFallbackClient()

    model_ref = CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")
    output = "".join(
        router.stream(
            model_ref=model_ref,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=32,
            temperature=0.2,
            top_p=1.0,
        )
    )
    assert output == "fallback response"


def test_stream_does_not_retry_after_timeout_without_output():
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))
    router.config.cloud.cloud_timeout_seconds = 1
    router.config.cloud.cloud_max_retries = 2
    attempts = {"count": 0}

    class _TimeoutClient:
        def validate_key(self):
            return True, "ok"

        def stream(self, **_kwargs):
            attempts["count"] += 1
            while True:
                time.sleep(3600)
                yield ""

    router._build_client = lambda _provider, _api_key: _TimeoutClient()
    model_ref = CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")

    with pytest.raises(RuntimeError):
        list(
            router.stream(
                model_ref=model_ref,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=32,
                temperature=0.2,
                top_p=1.0,
            )
        )

    assert attempts["count"] == 1
