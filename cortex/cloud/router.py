"""Cloud inference routing and provider dispatch."""

from __future__ import annotations

import logging
import os
import re
from http import HTTPStatus
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import anthropic
import openai

from cortex.cloud.clients import AnthropicClient, ChatCompletionsClient, OpenAIClient
from cortex.cloud.credentials import ENV_KEY_MAP, CloudCredentialStore
from cortex.cloud.retry import RetryCallback
from cortex.cloud.types import CloudModelRef, CloudProvider
from cortex.tooling.types import TurnInterrupt, TurnInterruptedError

logger = logging.getLogger(__name__)

# A provider's URL settings: its environment variable, its config.yaml key,
# and the state key /login saves it under.
_URL_SETTINGS = {
    CloudProvider.AZURE: ("AZURE_OPENAI_ENDPOINT", "cloud_azure_endpoint", "azure_endpoint"),
    CloudProvider.OPENAI_COMPATIBLE: (
        "OPENAI_COMPATIBLE_BASE_URL",
        "cloud_openai_compatible_base_url",
        "openai_compatible_base_url",
    ),
}
URL_CONFIG_KEYS = frozenset(setting[1] for setting in _URL_SETTINGS.values())
# How url_in_effect names the source of a URL that /login, cortex login or
# a version before 2.0.12 saved.
_SAVED = "saved in ~/.cortex/state.yaml"
# Providers whose /login also takes the URL requests go to.
LOGIN_URL_ARGUMENT = {
    CloudProvider.AZURE: "<endpoint>",
    CloudProvider.OPENAI_COMPATIBLE: "<base_url>",
}


class CloudRouter:
    """Route generation requests to cloud providers."""

    def __init__(self, config, credential_store: Optional[CloudCredentialStore] = None):
        self.config = config
        self.credential_store = credential_store or CloudCredentialStore()

    def _timeout_seconds(self) -> int:
        """How long a model request may wait for data before it fails."""
        cloud_cfg = getattr(self.config, "cloud", None)
        return max(1, int(getattr(cloud_cfg, "cloud_timeout_seconds", 60)))

    def _max_retries(self) -> int:
        cloud_cfg = getattr(self.config, "cloud", None)
        retries = int(getattr(cloud_cfg, "cloud_max_retries", 2))
        return max(0, retries)

    def url_in_effect(self, provider: CloudProvider) -> Tuple[str, str]:
        """The Azure endpoint or OpenAI-compatible base URL requests go to, and
        where it is set: its environment variable, else ~/.cortex/config.yaml,
        else saved state (("", "") if none)."""
        if provider not in _URL_SETTINGS:
            return "", ""
        env_name = _URL_SETTINGS[provider][0]
        url = os.environ.get(env_name, "").strip().rstrip("/")
        return (url, env_name) if url else self._saved_url(provider)

    def _saved_url(self, provider: CloudProvider) -> Tuple[str, str]:
        _env_name, config_key, state_key = _URL_SETTINGS[provider]
        cloud_cfg = getattr(self.config, "cloud", None)
        url = str(getattr(cloud_cfg, config_key, "") or "").strip()
        if url:
            # Set there, it is in effect even when it is only slashes.
            setting_source = getattr(self.config, "setting_source", None)
            source = setting_source(config_key) if callable(setting_source) else ""
            return (url.rstrip("/"), source or "config.yaml") if url.rstrip("/") else ("", "")
        get_state = getattr(self.config, "get_state_value", None)
        if callable(get_state):
            url = str(get_state(state_key, "") or "").strip().rstrip("/")
        return (url, _SAVED) if url else ("", "")

    def saved_endpoint_url(self, provider: CloudProvider) -> str:
        """The Azure endpoint or OpenAI-compatible base URL kept across sessions ("" if none)."""
        return self._saved_url(provider)[0] if provider in _URL_SETTINGS else ""

    def _azure_endpoint(self) -> str:
        """Azure resource endpoint."""
        return self.url_in_effect(CloudProvider.AZURE)[0]

    def openai_compatible_base_url(self) -> str:
        """Base URL of the OpenAI-compatible endpoint."""
        return self.url_in_effect(CloudProvider.OPENAI_COMPATIBLE)[0]

    def _build_client(self, provider: CloudProvider, api_key: str):
        timeout_seconds = self._timeout_seconds()
        if provider == CloudProvider.OPENAI:
            return OpenAIClient(api_key=api_key, timeout_seconds=timeout_seconds)
        if provider == CloudProvider.AZURE:
            endpoint = self._azure_endpoint()
            if not endpoint:
                raise RuntimeError(
                    "Azure OpenAI endpoint not configured. Add it with /login azure "
                    "<api_key> <endpoint>, or set AZURE_OPENAI_ENDPOINT."
                )
            # Azure's OpenAI-compatible v1 surface accepts Bearer auth, so the
            # standard OpenAI client works with a rebased URL.
            return OpenAIClient(
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                base_url=f"{endpoint}/openai/v1/",
            )
        if provider == CloudProvider.OPENAI_COMPATIBLE:
            base_url = self.openai_compatible_base_url()
            if not base_url:
                raise RuntimeError(
                    "OpenAI-compatible base URL not configured. Add it with /login "
                    "openai-compatible <api_key> <base_url>, or set OPENAI_COMPATIBLE_BASE_URL."
                )
            return ChatCompletionsClient(
                base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds
            )
        if provider == CloudProvider.ANTHROPIC:
            return AnthropicClient(api_key=api_key, timeout_seconds=timeout_seconds)
        raise ValueError(f"Unsupported cloud provider: {provider}")

    def get_auth_status(self, provider: CloudProvider) -> Tuple[bool, Optional[str]]:
        """Get auth status and active credential source for provider."""
        key, source = self.credential_store.get_api_key_with_source(provider)
        return bool(key), source

    def stream_events(
        self,
        *,
        model_ref: CloudModelRef,
        messages: Iterable[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools=None,
        tool_choice: str = "auto",
        tool_executor=None,
        max_tool_iterations: int = 100,
        on_retry: Optional[RetryCallback] = None,
        interrupt: Optional[TurnInterrupt] = None,
    ):
        """Yield normalized events from selected cloud provider."""
        script_path = os.environ.get("CORTEX_SCRIPTED_MODEL", "").strip()
        if script_path:
            # Deterministic scripted backend for end-to-end runtime validation.
            from cortex.cloud.clients.scripted_client import ScriptedClient

            client = ScriptedClient(script_path)
            yield from client.stream_events(
                model_id=model_ref.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                tool_choice=tool_choice,
                tool_executor=tool_executor,
                max_tool_iterations=max_tool_iterations,
            )
            return

        api_key, source = self.credential_store.get_api_key_with_source(model_ref.provider)
        if not api_key:
            env_name = ENV_KEY_MAP[model_ref.provider]
            raise RuntimeError(
                f"No API key configured for {model_ref.provider.value}. "
                f"Run /login {model_ref.provider.value} or set {env_name}."
            )

        client = self._build_client(model_ref.provider, api_key)
        try:
            yield from client.stream_events(
                model_id=model_ref.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                tool_choice=tool_choice,
                tool_executor=tool_executor,
                max_tool_iterations=max_tool_iterations,
                interrupt=interrupt,
                retries=self._max_retries(),
                on_retry=on_retry,
            )
        except TurnInterruptedError:
            raise
        except Exception as exc:
            if interrupt is not None and interrupt.is_set():
                raise TurnInterruptedError() from exc
            # Which key was sent: an old variable can shadow the key /login saved.
            key_label = {
                "env": f" (key from {ENV_KEY_MAP[model_ref.provider]})",
                "keychain": " (the saved key)",
                "file": " (the saved key)",
            }.get(str(source), "")
            logger.error("Cloud generation failed for %s: %s", model_ref.selector, exc)
            endpoint = str(getattr(client, "endpoint", "") or getattr(client, "base_url", ""))
            custom = model_ref.provider in (CloudProvider.AZURE, CloudProvider.OPENAI_COMPATIBLE)
            url, url_source = self.url_in_effect(model_ref.provider)
            failure = _describe_failure(
                exc, endpoint, custom_url=custom,
                url=url, origin=url_origin(model_ref.provider, url_source) if url else "",
            )
            raise RuntimeError(
                f"Cloud generation failed for {model_ref.selector}{key_label}: {failure}"
            ) from exc


def url_origin(provider: CloudProvider, source: str) -> str:
    """Where a provider's URL is set, as url_in_effect names it, worded for a
    message; for a saved one, also how to replace it."""
    if source != _SAVED:
        return f"set in {source}"
    argument = LOGIN_URL_ARGUMENT[provider]
    return (
        f"{_SAVED}; to change it: /login {provider.value} <api_key> {argument} in the app, "
        f"or `cortex login {provider.value} {argument}`"
    )


# A query parameter whose name has one of these in it holds a credential.
SECRET_PARAMETER = re.compile(r"key|token|secret|sig|pass|pwd|auth|credential|code", re.IGNORECASE)


_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


def _shown_parameter(parameter: str) -> str:
    name, equals, value = parameter.partition("=")
    return f"{name}=***" if equals and value and SECRET_PARAMETER.search(name) else parameter


def shown_url(url: str) -> str:
    """A URL as messages show it: credentials in it (user:password@, or a
    query parameter such as api_key=) masked."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return _without_userinfo(url)
    if "@" in parts.netloc:
        parts = parts._replace(netloc="***@" + parts.netloc.rsplit("@", 1)[1])
    query = "&".join(_shown_parameter(parameter) for parameter in parts.query.split("&"))
    shown = urlunsplit(parts._replace(query=query))
    host_read = bool(parts.netloc) and _SCHEME.match(url)
    return shown if host_read else _without_userinfo(shown)


def _without_userinfo(shown: str) -> str:
    """Text shown for a URL that does not start with scheme:// and a host
    (the URL the "must start with https:// or http://" error is about) or
    that cannot be parsed: what comes before an @ is masked in each part
    between slashes, up to a ? or #, so a user:password@ does not show. It
    only ever masks more of what is shown."""
    end = re.search(r"[?#]", shown)
    address = shown[: end.start()] if end else shown
    return re.sub(r"[^/]*@", "***@", address) + shown[len(address):]


def _describe_failure(
    exc: Exception, endpoint: str = "", *, custom_url: bool = False, url: str = "", origin: str = ""
) -> str:
    """The error, with what to check: for a connection that failed, the host
    and why; else the reason the server gave in its status line when the
    message lacks it (a gateway that answers "403 model not allowed: ..."
    with no body). When the URL may be what is wrong (no connection, or the
    server answered 401, 403 or 404) and it is a URL of the user's (`url`,
    set where `origin` says), also where it is set."""
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        timed_out = isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError))
        cause = exc.__cause__ or exc.__context__
        why = f" ({cause})" if cause is not None and str(cause) and not timed_out else ""
        verb = "timed out waiting for" if timed_out else "could not reach"
        failure = f"{verb} {shown_url(endpoint) or 'the provider'}{why}"
        return _where_set(failure, url, origin, named=endpoint) or failure
    text = str(exc)
    response = getattr(exc, "response", None)
    reason = str(getattr(response, "reason_phrase", "") or "").strip()
    try:
        standard = HTTPStatus(int(getattr(response, "status_code", 0))).phrase
    except ValueError:
        standard = ""
    if reason and reason != standard and reason.lower() not in text.lower():
        text = f"{text} ({reason})"
    status = getattr(response, "status_code", None)
    if status in (403, 404):
        # The usual cause: a model name the provider does not know, or one
        # this key may not use; with a URL of your own, also the URL.
        text = f"{_ended(text)} Check the model name and that your key may use it"
        if custom_url and endpoint:
            text += f", and that {shown_url(endpoint)} is where the API's paths start (often /v1)"
        return _where_set(text, url, origin, named=endpoint) or f"{text}."
    if status == 401:
        return _where_set(text, url, origin) or text
    return text


def _where_set(failure: str, url: str, origin: str, *, named: str = "") -> str:
    """The failure, followed by where the user's URL is set ("" when it is
    not a URL of theirs). `named` is the URL the failure has shown: Azure
    requests go to a longer one than the endpoint that is set."""
    if not origin:
        return ""
    which = "That URL" if named == url else f"The URL in use, {shown_url(url)},"
    return f"{_ended(failure)} {which} is {origin}."


def _ended(sentence: str) -> str:
    """A server's message ends as it ended it, or with a full stop."""
    sentence = sentence.rstrip()
    return sentence if sentence.endswith((".", "?", "!")) else f"{sentence}."
