from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cortex.app.command_service import CommandService
from cortex.app.model_service import ModelService
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from tests.lumen_fakes import FakeLumenRuntime, catalog


class _FakeConfig:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self.last_used = ""

    def set_state_value(self, key: str, value: str) -> None:
        self.state[key] = value

    def update_last_used_model(self, name: str) -> None:
        self.last_used = name


class _FakeCloudRouter:
    def __init__(self, *, authenticated: bool = False) -> None:
        self.authenticated = authenticated

    def get_auth_status(self, _provider):
        return self.authenticated, "fake"


class _FakeCredentialStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []

    def get_auth_summary(self, provider):
        return {"authenticated": False, "source": None}

    def save_api_key(self, provider, api_key):
        self.saved.append((provider.value, api_key))
        return True, ""

    def delete_api_key(self, provider):
        return True, "deleted"


def _build(
    *,
    lumen: FakeLumenRuntime | None = None,
    cloud_authenticated: bool = False,
) -> tuple[CommandService, ModelService, FakeLumenRuntime]:
    runtime = lumen or FakeLumenRuntime()
    model_service = ModelService(
        config=_FakeConfig(),
        lumen_runtime=runtime,
        gpu_validator=SimpleNamespace(),
        cloud_router=_FakeCloudRouter(authenticated=cloud_authenticated),
        credential_store=_FakeCredentialStore(),
        cloud_catalog=SimpleNamespace(list_models=lambda: []),
    )
    service = CommandService(
        model_service=model_service,
        clear_session=lambda _session_id: {"ok": True, "message": "cleared"},
        save_session=lambda _session_id: {"ok": True, "path": "/tmp/x.json"},
        lumen_runtime=runtime,
    )
    return service, model_service, runtime


# ---- /download -------------------------------------------------------------


def test_download_requires_args() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/download")
    assert result["ok"] is False
    assert "Usage: /download <model[:quant]>" in str(result["message"])


def test_download_rejects_extra_arguments() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/download a b")
    assert result["ok"] is False
    assert "Usage: /download" in str(result["message"])


def test_download_rejects_invalid_shell_syntax() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command='/download "unterminated')
    assert result["ok"] is False
    assert "Invalid download arguments" in str(result["message"])


def test_download_unknown_model_lists_supported() -> None:
    service, _, runtime = _build()
    result = service.execute(session_id="s1", command="/download llama-3")
    assert result["ok"] is False
    assert "qwen3-5-9b" in str(result["message"])
    assert runtime.pull_calls == []


def test_download_already_cached_loads_the_model() -> None:
    service, model_service, runtime = _build()
    result = service.execute(session_id="s1", command="/download qwen3-5-9b:q4_0")
    assert result["ok"] is True
    assert "Already downloaded: local · qwen3-5-9b:q4_0 — now active." in str(result["message"])
    assert runtime.pull_calls == []
    assert runtime.ensure_calls == ["qwen3-5-9b:q4_0"]
    assert model_service.active_target.local_model == "qwen3-5-9b:q4_0"


def test_download_pulls_and_streams_progress_lines() -> None:
    service, _, runtime = _build()
    events: list[dict] = []

    result = service.execute(
        session_id="s1",
        command="/download qwen3-6-27b:q4_0",
        progress_callback=events.append,
    )

    assert result["ok"] is True, result
    assert runtime.pull_calls == ["qwen3-6-27b:q4_0"]
    # Downloading IS selecting: the model auto-loads and becomes active.
    assert "local · qwen3-6-27b:q4_0 ready — now active." in str(result["message"])
    assert runtime.ensure_calls == ["qwen3-6-27b:q4_0"]
    # Progress lines are forwarded in the schema the worker forwards to the
    # TUI: pull lines as kind "download" (bytes measured by the caller), and a
    # final "model-load" transition when the chained GPU load begins.
    assert events, "expected forwarded progress payloads"
    for event in events[:-1]:
        assert event["kind"] == "download"
        assert event["repo_id"] == "qwen3-6-27b:q4_0"
        assert isinstance(event["phase"], str) and event["phase"]
        assert "bytes_downloaded" not in event
    assert events[-1]["kind"] == "model-load"
    assert events[-1]["phase"] == "loading"


def test_download_failure_is_summarized_and_does_not_activate() -> None:
    service, model_service, runtime = _build(lumen=FakeLumenRuntime(pull_ok=False))
    result = service.execute(session_id="s1", command="/download qwen3-6-27b:q4_0")
    assert result["ok"] is False
    assert "Download failed" in str(result["message"])
    assert runtime.ensure_calls == []
    assert model_service.active_target.local_model is None


def test_download_load_failure_after_pull_is_reported() -> None:
    runtime = FakeLumenRuntime(ensure_ok=False, ensure_message="boot failed")
    service, model_service, _ = _build(lumen=runtime)
    result = service.execute(session_id="s1", command="/download qwen3-6-27b:q4_0")
    assert result["ok"] is False
    assert "Downloaded local · qwen3-6-27b:q4_0, but loading failed" in str(result["message"])
    assert model_service.active_target.local_model is None


def test_preflight_download_resolves_bare_name_to_cached_quant() -> None:
    service, _, _ = _build()
    preflight = service.preflight_download("qwen3-5-9b")
    assert preflight["ok"] is True
    assert preflight["selector"] == "qwen3-5-9b:q4_0"
    assert preflight["cached"] is True


# ---- /benchmark -------------------------------------------------------------


def test_benchmark_rejects_cloud_backend() -> None:
    service, model_service, _ = _build(cloud_authenticated=True)
    model_service.active_target = ActiveModelTarget.cloud(
        CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.1")
    )
    result = service.execute(session_id="s1", command="/benchmark")
    assert result["ok"] is False
    assert "local models only" in str(result["message"])


def test_benchmark_requires_loaded_model() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/benchmark")
    assert result["ok"] is False
    assert "No local model loaded" in str(result["message"])


def test_benchmark_rejects_invalid_tokens() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/benchmark --tokens abc")
    assert result["ok"] is False
    assert "positive integer" in str(result["message"])


def test_benchmark_rejects_out_of_range_tokens() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/benchmark 9999")
    assert result["ok"] is False
    assert "between 1 and 8192" in str(result["message"])


def test_benchmark_measures_through_lumen_endpoint(monkeypatch) -> None:
    service, model_service, runtime = _build()
    assert model_service.select_local_model("qwen3-5-9b:q4_0")["ok"] is True

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"usage": {"completion_tokens": 50, "prompt_tokens": 5, "total_tokens": 55}}
            ).encode("utf-8")

    captured: dict = {}

    def _fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = service.execute(session_id="s1", command="/benchmark 50")

    assert result["ok"] is True, result
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["max_tokens"] == 50
    assert captured["body"]["model"] == "qwen3-5-9b"
    benchmark = result["benchmark"]
    assert benchmark["tokens_generated"] == 50
    assert "tok/s" in str(result["message"])


# ---- /model ------------------------------------------------------------------


def test_model_list_shows_lumen_tags() -> None:
    service, model_service, _ = _build()
    model_service.select_local_model("qwen3-5-9b:q4_0")

    result = service.execute(session_id="s1", command="/model list")

    message = str(result["message"])
    assert "Local models (Lumen):" in message
    assert "- qwen3-5-9b:q4_0 (active, loaded)" in message
    assert "- qwen3-5-9b:q8_0 (not downloaded)" in message


def test_model_routes_cloud_selector_to_cloud() -> None:
    service, _, _ = _build(cloud_authenticated=True)
    result = service.execute(session_id="s1", command="/model openai:gpt-5.1")
    assert result["ok"] is True
    assert "openai:gpt-5.1" in str(result["message"])


def test_model_routes_lumen_selector_with_colon_to_local() -> None:
    service, model_service, runtime = _build()
    result = service.execute(session_id="s1", command="/model qwen3-5-9b:q4_0")
    assert result["ok"] is True, result
    assert runtime.ensure_calls == ["qwen3-5-9b:q4_0"]
    assert model_service.active_target.local_model == "qwen3-5-9b:q4_0"


def test_model_rejects_empty_cloud_selector() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/model openai:")
    assert result["ok"] is False
    assert "Usage: /model <provider:model>" in str(result["message"])


def test_model_uncached_selector_sync_fallback_carries_marker() -> None:
    # The worker intercepts this path in the TUI and auto-downloads; the sync
    # result (direct callers) carries the structured marker, no /download hint.
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/model qwen3-6-27b:q4_0")
    assert result["ok"] is False
    assert result["not_cached"] is True
    assert "/download" not in str(result["message"])


# ---- /setup --------------------------------------------------------------------


def test_setup_loads_first_cached_local_model() -> None:
    service, model_service, runtime = _build()
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is True, result
    assert runtime.ensure_calls == ["qwen3-5-9b:q4_0"]
    assert model_service.active_target.local_model == "qwen3-5-9b:q4_0"


def test_setup_reports_complete_when_model_already_active() -> None:
    service, model_service, runtime = _build()
    model_service.select_local_model("qwen3-5-9b:q4_0")
    runtime.ensure_calls.clear()

    result = service.execute(session_id="s1", command="/setup")

    assert result["ok"] is True
    assert "Setup complete" in str(result["message"])
    assert runtime.ensure_calls == []


def test_setup_guides_to_download_when_nothing_cached() -> None:
    service, _, _ = _build(
        lumen=FakeLumenRuntime(models=catalog(("qwen3-5-9b", "Q4_0", False)))
    )
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is False
    # One-flow guidance: the picker downloads and loads; never a /download dead end.
    assert "Pick one with /model" in str(result["message"])
    assert "/download" not in str(result["message"])


# ---- /login ---------------------------------------------------------------------


def test_login_rejects_lumen_provider() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/login lumen key")
    assert result["ok"] is False
    assert "managed local engine" in str(result["message"])


def test_login_rejects_unknown_provider() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/login huggingface")
    assert result["ok"] is False
    assert "/login azure" in str(result["message"])


def test_login_provider_key_save_sets_default_success_message() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/login openai sk-test")
    assert result["ok"] is True
    assert result["message"] == "Saved openai API key."


def test_login_status_includes_formatted_message() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/login openai")
    assert result["ok"] is True
    assert "openai" in str(result["message"]).lower()


# ---- dispatch basics --------------------------------------------------------------


def test_execute_rejects_empty_command() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="   ")
    assert result["ok"] is False
    assert "Command cannot be empty" in str(result["message"])


def test_execute_rejects_non_slash_command() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="hello")
    assert result["ok"] is False
    assert "Not a slash command" in str(result["message"])


@pytest.mark.parametrize("command", ["/quit", "/exit"])
def test_execute_quit_commands_return_exit(command: str) -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command=command)
    assert result == {"ok": True, "exit": True}


def test_execute_save_sets_default_message_from_path() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/save")
    assert result["ok"] is True
    assert result["message"] == "Saved conversation: /tmp/x.json."


def test_execute_unknown_command_returns_error() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/nonsense")
    assert result["ok"] is False
    assert "Unknown command" in str(result["message"])


def test_help_lists_current_commands_without_template() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/help")
    message = str(result["message"])
    assert "/template" not in message
    assert "/download" in message and "/benchmark" in message


def test_template_command_is_gone() -> None:
    service, _, _ = _build()
    result = service.execute(session_id="s1", command="/template status")
    assert result["ok"] is False
    assert "Unknown command" in str(result["message"])


def test_cancelled_download_reports_cancelled_not_failed() -> None:
    """A user-initiated cancel is not a failure (consensus-confirmed wording
    fix: it used to resolve as 'Download failed: Download cancelled.')."""
    service, _models, runtime = _build()
    runtime.pull = lambda selector, *, on_line=None, cancel_requested=None: (
        False,
        "Download cancelled.",
    )
    service.preflight_download = lambda args: {
        "ok": True,
        "selector": "qwen3-6-27b:q4_0",
        "cached": False,
    }
    result = service.execute(session_id="s1", command="/download qwen3-6-27b:q4_0")
    assert result["ok"] is False
    assert result["message"] == "Download cancelled."
    assert "failed" not in str(result["message"]).lower()
