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
        "Cloud generation failed for openai:gpt-5.5 (the saved key): boom"
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


def test_openai_and_anthropic_never_take_a_base_url_from_the_environment(monkeypatch):
    """The SDKs read OPENAI_BASE_URL and ANTHROPIC_BASE_URL when no host is
    given: one left on a runner would send the key to that host, and the reply
    would still be labelled verified. The clients always name the official host."""
    from cortex.cloud.clients.anthropic_client import AnthropicClient
    from cortex.cloud.clients.openai_client import OpenAIClient

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    openai = OpenAIClient(api_key="sk-test")
    anthropic = AnthropicClient(api_key="sk-ant-test")
    assert str(openai.client.base_url) == "https://api.openai.com/v1/"
    assert openai.endpoint == "https://api.openai.com/v1"
    assert str(anthropic.client.base_url) == "https://api.anthropic.com"
    assert anthropic.endpoint == "https://api.anthropic.com"


def _status_error(status: int, message: str) -> Exception:
    import httpx2 as httpx
    import openai

    response = httpx.Response(status, request=httpx.Request("POST", "http://h/v1"))
    return openai.APIStatusError(message, response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "named", "said"),
    [
        (_status_error(401, "Bad key."), "http://h/v1",
         "Bad key. The URL in use, http://***@h/v1, is set in X."),
        (_status_error(403, "No"), "http://u:p@h/v1",
         "No. Check the model name and that your key may use it, and that http://***@h/v1 is "
         "where the API's paths start (often /v1). That URL is set in X."),
        # Azure sends requests to a longer URL than the endpoint that is set.
        (_status_error(404, "No"), "http://u:p@h/v1/openai/v1/",
         "No. Check the model name and that your key may use it, and that "
         "http://***@h/v1/openai/v1/ is where the API's paths start (often /v1). "
         "The URL in use, http://***@h/v1, is set in X."),
        # The server's message keeps the way it ends.
        (_status_error(401, "Really?"), "", "Really? The URL in use, http://***@h/v1, is set in X."),
        (_status_error(401, "Wait...\n"), "", "Wait... The URL in use, http://***@h/v1, is set in X."),
        (_status_error(401, "Denied"), "", "Denied. The URL in use, http://***@h/v1, is set in X."),
        (_status_error(400, "Bad request"), "http://u:p@h/v1", "Bad request"),
        (_status_error(429, "Slow down"), "http://u:p@h/v1", "Slow down"),
    ],
)
def test_a_failure_the_url_may_explain_says_where_the_url_is_set(exc, named, said):
    from cortex.cloud.router import _describe_failure

    described = _describe_failure(
        exc, named, custom_url=True, url="http://u:p@h/v1", origin="set in X")
    assert described == said


def test_a_timeout_and_a_connection_with_no_cause_say_where_the_url_is_set():
    import httpx2 as httpx
    import openai

    from cortex.cloud.router import _describe_failure

    request = httpx.Request("POST", "http://h/v1")
    for exc, said in (
        (openai.APITimeoutError(request=request),
         "timed out waiting for http://h/v1. That URL is set in X."),
        (openai.APIConnectionError(request=request),
         "could not reach http://h/v1. That URL is set in X."),
    ):
        assert _describe_failure(
            exc, "http://h/v1", custom_url=True, url="http://h/v1", origin="set in X") == said
        # At OpenAI or Anthropic there is no URL of the user's to name.
        assert _describe_failure(exc, "http://h/v1") == said.split(". That URL")[0]


@pytest.mark.parametrize(
    ("status", "said"),
    [
        (401, "Bad key."),
        (404, "No. Check the model name and that your key may use it."),
    ],
)
def test_a_failure_at_openai_or_anthropic_names_no_url_of_the_users(status, said):
    router = _make_router(_DummyCredentials(key="abc", source="keychain"))

    class _Client:
        base_url = "https://api.openai.com/v1"

        def stream_events(self, **_kwargs):
            raise _status_error(status, "Bad key." if status == 401 else "No")
            yield

    router._build_client = lambda _provider, _api_key: _Client()

    with pytest.raises(RuntimeError) as exc:
        _events(router)

    assert str(exc.value) == f"Cloud generation failed for openai:gpt-5.5 (the saved key): {said}"


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://u:sekret@h:9/v1?api_key=K&region=eu", "https://***@h:9/v1?api_key=***&region=eu"),
        ("ws://u:sekret@h/v1", "ws://***@h/v1"),
        ("HTTP://u:sekret@h/v1", "http://***@h/v1"),
        # No scheme, as in the error that says the URL needs one.
        ("u:sekret@h:9/v1?api_key=K", "***@h:9/v1?api_key=***"),
        ("sekret@h/v1", "***@h/v1"),
        ("u:sekret@h?x=1", "***@h?x=1"),
        ("u:sek@ret@h#x", "***@h#x"),
        ("u:sekret@h/v1?next=http://x", "***@h/v1?next=http://x"),
        # A scheme mistyped, or with something around it.
        ("//u:sekret@h/v1", "//***@h/v1"),
        ("https:/u:sekret@h/v1", "https:///***@h/v1"),
        ("https//u:sekret@h/v1", "https//***@h/v1"),
        ("://u:sekret@h", "://***@h"),
        ("1http://u:sekret@h", "1http://***@h"),
        ('"https://u:sekret@h"', '"https://***@h"'),
        ("https ://u:sekret@h", "https ://***@h"),
        (" http://u:sekret@h/v1", "http://***@h/v1"),
        ("http:\t//u:sekret@h", "http://***@h"),
        ("\x01https://u:sekret@h", "https://***@h"),
        ("a@b/u:sekret@h/v1", "***@b/***@h/v1"),
        # A scheme with no host after it.
        ("https:///u:sekret@h/v1", "https:///***@h/v1"),
        # A URL that cannot be parsed.
        ("http://u:sekret@[::1", "http://***@[::1"),
        ("u:sekret@[::1", "***@[::1"),
        # An @ that is no credential: after the host of a URL with a scheme,
        # or after a ? or #.
        ("https://h/users/a@b", "https://h/users/a@b"),
        ("a+b://h/x@y", "a+b://h/x@y"),
        ("h/v1?email=a@b", "h/v1?email=a@b"),
        ("h#a@b", "h#a@b"),
        # Nothing to mask.
        ("h:9/v1", "h:9/v1"),
        ("not-a-url", "not-a-url"),
        ("/v1", "/v1"),
        ("", ""),
    ],
)
def test_a_shown_url_masks_its_credentials_with_or_without_a_scheme(url, shown):
    from cortex.cloud.router import shown_url

    assert shown_url(url) == shown


@pytest.mark.parametrize(
    ("key", "value", "shown"),
    [
        ("cloud_openai_compatible_base_url", "u:sekret@h/v1", "***@h/v1"),
        ("cloud_azure_endpoint", "sekret@h", "***@h"),
        ("cloud_default_openai_model", "ft:gpt@org", "ft:gpt@org"),  # not a URL
        ("cloud_default_openai_model", "http://u:sekret@h/v1", "http://***@h/v1"),
        ("temperature", 0.3, "0.3"),
        ("auto_update_check", False, "false"),
    ],
)
def test_status_reads_only_a_url_setting_as_a_url_without_a_scheme(key, value, shown):
    from cortex.app.command_service import _setting_text

    assert _setting_text(key, value) == shown
