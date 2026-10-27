"""Cloud inference routing and provider dispatch."""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Dict, Iterable, Optional, Tuple

from cortex.cloud.clients import AnthropicClient, ChatCompletionsClient, OpenAIClient
from cortex.cloud.credentials import ENV_KEY_MAP, CloudCredentialStore
from cortex.cloud.retry import RetryCallback
from cortex.cloud.types import CloudModelRef, CloudProvider
from cortex.tooling.types import TurnInterrupt, TurnInterruptedError

logger = logging.getLogger(__name__)


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

    def _configured_url(self, env_name: str, config_key: str, state_key: str) -> str:
        """A provider URL from the environment, then ~/.cortex/config.yaml, then persisted state."""
        url = os.environ.get(env_name, "").strip()
        if not url:
            cloud_cfg = getattr(self.config, "cloud", None)
            url = str(getattr(cloud_cfg, config_key, "") or "").strip()
        if not url:
            get_state = getattr(self.config, "get_state_value", None)
            if callable(get_state):
                url = str(get_state(state_key, "") or "").strip()
        return url.rstrip("/")

    def _azure_endpoint(self) -> str:
        """Azure resource endpoint."""
        return self._configured_url("AZURE_OPENAI_ENDPOINT", "cloud_azure_endpoint", "azure_endpoint")

    def openai_compatible_base_url(self) -> str:
        """Base URL of the OpenAI-compatible endpoint."""
        return self._configured_url(
            "OPENAI_COMPATIBLE_BASE_URL",
            "cloud_openai_compatible_base_url",
            "openai_compatible_base_url",
        )

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
            raise RuntimeError(
                f"Cloud generation failed for {model_ref.selector}{key_label}: "
                f"{_describe_failure(exc)}"
            ) from exc


def _describe_failure(exc: Exception) -> str:
    """The error, with the reason the server gave in its status line when the
    message lacks it (a gateway that answers "403 model not allowed: ..."
    with no body)."""
    text = str(exc)
    response = getattr(exc, "response", None)
    reason = str(getattr(response, "reason_phrase", "") or "").strip()
    try:
        standard = HTTPStatus(int(getattr(response, "status_code", 0))).phrase
    except ValueError:
        standard = ""
    if reason and reason != standard and reason.lower() not in text.lower():
        text = f"{text} ({reason})"
    if getattr(response, "status_code", None) in (403, 404):
        # The usual cause: a model name the provider does not know, or one
        # this key may not use.
        text = f"{text}. Check the model name and that your key may use it."
    return text
