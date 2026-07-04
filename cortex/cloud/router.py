"""Cloud inference routing and provider dispatch."""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from typing import Callable, Dict, Iterable, Optional, Tuple

from cortex.cloud.clients import AnthropicClient, LumenClient, OpenAIClient
from cortex.cloud.credentials import ENV_KEY_MAP, CloudCredentialStore
from cortex.cloud.types import CloudModelRef, CloudProvider
from cortex.tooling.types import FinishEvent, TextDeltaEvent

logger = logging.getLogger(__name__)


class CloudRouter:
    """Route generation requests to cloud providers."""

    def __init__(self, config, credential_store: Optional[CloudCredentialStore] = None):
        self.config = config
        self.credential_store = credential_store or CloudCredentialStore()
        # Set by the worker: returns the managed Lumen server's base URL
        # (None when the server is not running).
        self.lumen_base_url: Optional[Callable[[], Optional[str]]] = None

    def _timeout_seconds(self) -> int:
        cloud_cfg = getattr(self.config, "cloud", None)
        tools_cfg = getattr(self.config, "tools", None)
        cloud_timeout = int(getattr(cloud_cfg, "cloud_timeout_seconds", 60))
        tools_idle = int(getattr(tools_cfg, "tools_idle_timeout_seconds", cloud_timeout))
        return max(1, tools_idle)

    def _max_retries(self) -> int:
        cloud_cfg = getattr(self.config, "cloud", None)
        retries = int(getattr(cloud_cfg, "cloud_max_retries", 2))
        return max(0, retries)

    def _azure_endpoint(self) -> str:
        """Azure resource endpoint: env, then config, then persisted state."""
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        if not endpoint:
            cloud_cfg = getattr(self.config, "cloud", None)
            endpoint = str(getattr(cloud_cfg, "cloud_azure_endpoint", "") or "").strip()
        if not endpoint:
            get_state = getattr(self.config, "get_state_value", None)
            if callable(get_state):
                endpoint = str(get_state("azure_endpoint", "") or "").strip()
        return endpoint.rstrip("/")

    def _build_client(self, provider: CloudProvider, api_key: str):
        timeout_seconds = self._timeout_seconds()
        if provider == CloudProvider.OPENAI:
            return OpenAIClient(api_key=api_key, timeout_seconds=timeout_seconds)
        if provider == CloudProvider.AZURE:
            endpoint = self._azure_endpoint()
            if not endpoint:
                raise RuntimeError(
                    "Azure OpenAI endpoint not configured. Set AZURE_OPENAI_ENDPOINT "
                    "or cloud_azure_endpoint in config.yaml."
                )
            # Azure's OpenAI-compatible v1 surface accepts Bearer auth, so the
            # standard OpenAI client works with a rebased URL.
            return OpenAIClient(
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                base_url=f"{endpoint}/openai/v1/",
            )
        if provider == CloudProvider.LUMEN:
            base_url = self.lumen_base_url() if self.lumen_base_url else None
            if not base_url:
                raise RuntimeError(
                    "Lumen server is not running. Select a local model with /model first."
                )
            # Lumen speaks Chat Completions (not the Responses API), so it gets
            # its own client. Generation timeout is generous: local decode of a
            # long answer legitimately takes minutes on big prompts.
            return LumenClient(base_url=base_url, timeout_seconds=max(600, timeout_seconds))
        if provider == CloudProvider.ANTHROPIC:
            return AnthropicClient(api_key=api_key, timeout_seconds=timeout_seconds)
        raise ValueError(f"Unsupported cloud provider: {provider}")

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Return whether a failed cloud attempt should be retried."""
        if isinstance(exc, TimeoutError):
            return False

        lowered = str(exc).lower()
        non_retryable_markers = (
            "invalid api key",
            "authentication",
            "unauthorized",
            "forbidden",
            "permission denied",
            "model not found",
            "unknown model",
            "does not exist",
            "status=401",
            "status=403",
            "status=404",
        )
        return not any(marker in lowered for marker in non_retryable_markers)

    def _stream_with_idle_timeout(
        self,
        stream_iterable,
        idle_timeout_seconds: int,
        on_wait: Optional[Callable[[int], None]] = None,
    ):
        """Yield from possibly-blocking event stream with idle timeout watchdog."""
        event_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue(maxsize=256)
        stop_signal = threading.Event()
        done_sentinel = ("done", None)

        def _producer() -> None:
            try:
                for event in stream_iterable:
                    if stop_signal.is_set():
                        break
                    event_queue.put(("event", event))
            except BaseException as exc:  # pragma: no cover - exercised by integration tests
                event_queue.put(("error", exc))
            finally:
                event_queue.put(done_sentinel)

        worker = threading.Thread(target=_producer, daemon=True)
        worker.start()

        poll_seconds = 1
        waited_seconds = 0

        while True:
            try:
                event_type, payload = event_queue.get(timeout=poll_seconds)
            except queue.Empty as exc:
                waited_seconds += poll_seconds
                if on_wait is not None:
                    on_wait(waited_seconds)
                if waited_seconds < idle_timeout_seconds:
                    continue
                stop_signal.set()
                raise TimeoutError(
                    f"Cloud stream stalled for more than {idle_timeout_seconds}s while waiting for output."
                ) from exc

            if event_type == "event":
                waited_seconds = 0
                yield payload
                continue
            if event_type == "error":
                stop_signal.set()
                if isinstance(payload, BaseException):
                    raise payload
                raise RuntimeError(f"Unexpected stream error payload: {payload!r}")
            if (event_type, payload) == done_sentinel:
                break

    def get_auth_status(self, provider: CloudProvider) -> Tuple[bool, Optional[str]]:
        """Get auth status and active credential source for provider."""
        if provider == CloudProvider.LUMEN:
            return True, "local"
        key, source = self.credential_store.get_api_key_with_source(provider)
        return bool(key), source

    def validate_api_key(self, provider: CloudProvider, api_key: str) -> Tuple[bool, str]:
        """Validate an API key by creating a provider client and running a cheap check."""
        try:
            client = self._build_client(provider, api_key)
        except Exception as exc:
            return False, str(exc)
        valid, message = client.validate_key()
        return bool(valid), str(message)

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
        max_tool_iterations: int = 25,
        on_wait: Optional[Callable[[int, int, int], None]] = None,
        on_retry: Optional[Callable[[int, int, str], None]] = None,
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

        api_key: Optional[str]
        source: Optional[str]
        if model_ref.provider == CloudProvider.LUMEN:
            # Managed local server — no credentials involved.
            api_key, source = "lumen", "local"
        else:
            api_key, source = self.credential_store.get_api_key_with_source(model_ref.provider)
        if not api_key:
            env_name = ENV_KEY_MAP[model_ref.provider]
            raise RuntimeError(
                f"No API key configured for {model_ref.provider.value}. "
                f"Run /login {model_ref.provider.value} or set {env_name}."
            )

        request_id = uuid.uuid4().hex[:10]
        retries = self._max_retries()
        last_error: Optional[Exception] = None
        total_attempts = retries + 1
        per_attempt_timeout = max(5, int(self._timeout_seconds()))

        for attempt in range(retries + 1):
            attempt_num = attempt + 1
            client = self._build_client(model_ref.provider, api_key)
            emitted = False
            interactive_tools = bool(tools) and tool_executor is not None
            logger.info(
                "Cloud stream attempt=%s/%s request_id=%s provider=%s model=%s timeout=%ss interactive_tools=%s",
                attempt_num,
                total_attempts,
                request_id,
                model_ref.provider.value,
                model_ref.model_id,
                per_attempt_timeout,
                interactive_tools,
            )
            try:
                if hasattr(client, "stream_events"):
                    client_events = client.stream_events(
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
                else:
                    # Backward compatibility for older test doubles.
                    client_events = (
                        TextDeltaEvent(delta=str(delta))
                        for delta in client.stream(
                            model_id=model_ref.model_id,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    )

                # Tool-capable cloud turns can require interactive permission prompts.
                # Keep those on the main thread to avoid TTY races and prompt corruption.
                if interactive_tools:
                    for event in client_events:
                        emitted = True
                        yield event
                else:
                    wait_callback = None
                    if on_wait is not None:
                        def wait_callback(waited: int, attempt_value: int = attempt_num) -> None:
                            on_wait(waited, attempt_value, total_attempts)

                    for event in self._stream_with_idle_timeout(
                        client_events,
                        per_attempt_timeout,
                        on_wait=wait_callback,
                    ):
                        emitted = True
                        yield event

                logger.info(
                    "Cloud stream completed request_id=%s attempt=%s/%s provider=%s model=%s emitted=%s",
                    request_id,
                    attempt_num,
                    total_attempts,
                    model_ref.provider.value,
                    model_ref.model_id,
                    emitted,
                )
                return
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable_error(exc)
                logger.warning(
                    "Cloud stream failed request_id=%s attempt=%s/%s provider=%s model=%s retryable=%s error=%s",
                    request_id,
                    attempt_num,
                    total_attempts,
                    model_ref.provider.value,
                    model_ref.model_id,
                    retryable,
                    exc,
                )

                # Fallback to single-shot text completion when stream produced nothing.
                if not emitted and hasattr(client, "generate_once"):
                    try:
                        fallback_text = client.generate_once(
                            model_id=model_ref.model_id,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        if fallback_text:
                            yield TextDeltaEvent(delta=fallback_text)
                            yield FinishEvent(
                                reason="stop",
                                provenance=getattr(client, "last_provenance", None),
                            )
                            return
                    except Exception as fallback_exc:
                        last_error = fallback_exc
                        logger.warning(
                            "Cloud fallback failed request_id=%s attempt=%s/%s provider=%s model=%s error=%s",
                            request_id,
                            attempt_num,
                            total_attempts,
                            model_ref.provider.value,
                            model_ref.model_id,
                            fallback_exc,
                        )

                # Retry only if no output emitted and attempts remain.
                if emitted or attempt >= retries or not retryable:
                    break
                if on_retry is not None:
                    on_retry(attempt_num, total_attempts, str(last_error or exc))

        source_label = f" (using {source})" if source else ""
        failure = RuntimeError(
            f"Cloud generation failed for {model_ref.selector}{source_label}: {last_error}"
        )
        logger.error(
            "Cloud generation failed request_id=%s provider=%s model=%s error=%s",
            request_id,
            model_ref.provider.value,
            model_ref.model_id,
            last_error,
        )
        raise failure

    def stream(
        self,
        *,
        model_ref: CloudModelRef,
        messages: Iterable[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        on_wait: Optional[Callable[[int, int, int], None]] = None,
        on_retry: Optional[Callable[[int, int, str], None]] = None,
    ):
        """Backward-compatible text-only delta stream."""
        for event in self.stream_events(
            model_ref=model_ref,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=None,
            tool_choice="auto",
            tool_executor=None,
            on_wait=on_wait,
            on_retry=on_retry,
        ):
            if isinstance(event, TextDeltaEvent) and event.delta:
                yield event.delta
