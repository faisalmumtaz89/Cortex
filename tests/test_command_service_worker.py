from __future__ import annotations

from types import SimpleNamespace

import pytest

from cortex.app.command_service import CommandService
from cortex.app.model_service import ModelService
from cortex.cloud.types import CloudModelRef, CloudProvider


class _FakeConfig:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}

    def set_state_value(self, key: str, value: str) -> None:
        self.state[key] = value


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


def _build(*, cloud_authenticated: bool = False) -> tuple[CommandService, ModelService]:
    catalog = [
        CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5.5"),
        CloudModelRef(provider=CloudProvider.ANTHROPIC, model_id="claude-fable-5"),
    ]
    model_service = ModelService(
        config=_FakeConfig(),
        cloud_router=_FakeCloudRouter(authenticated=cloud_authenticated),
        credential_store=_FakeCredentialStore(),
        cloud_catalog=SimpleNamespace(list_models=lambda: list(catalog)),
    )
    service = CommandService(
        model_service=model_service,
        clear_session=lambda _session_id: {"ok": True, "message": "cleared"},
        save_session=lambda _session_id: {"ok": True, "path": "/tmp/x.json"},
    )
    return service, model_service


# ---- /model ------------------------------------------------------------------


def test_model_routes_cloud_selector_to_cloud() -> None:
    service, model_service = _build(cloud_authenticated=True)
    result = service.execute(session_id="s1", command="/model openai:gpt-5.1")
    assert result["ok"] is True
    assert result["message"] == "openai:gpt-5.1 — now active."
    assert model_service.active_target.label == "openai:gpt-5.1"


@pytest.mark.parametrize("command", ["/model openai:", "/model gpt-5.5"])
def test_model_without_provider_and_model_is_a_usage_error(command: str) -> None:
    service, model_service = _build(cloud_authenticated=True)
    result = service.execute(session_id="s1", command=command)
    assert result["ok"] is False
    assert "Usage: /model <provider>:<model>" in str(result["message"])
    assert model_service.active_target.cloud_model is None


def test_model_rejects_unknown_provider() -> None:
    service, _ = _build(cloud_authenticated=True)
    result = service.execute(session_id="s1", command="/model lumen:qwen3-5-9b")
    assert result["ok"] is False
    assert "Unsupported provider 'lumen'" in str(result["message"])


def test_model_list_shows_catalog_with_auth_and_active_tags() -> None:
    service, _ = _build(cloud_authenticated=True)
    service.execute(session_id="s1", command="/model openai:gpt-5.5")
    message = str(service.execute(session_id="s1", command="/model")["message"])
    assert message.splitlines()[0] == "Active model: openai:gpt-5.5"
    assert "- openai:gpt-5.5 (active, ready)" in message
    assert "- anthropic:claude-fable-5 (ready)" in message
    assert "Local" not in message


@pytest.mark.parametrize("command", ["/gpu", "/download qwen3-5-9b", "/benchmark", "/setup"])
def test_gpu_download_benchmark_setup_are_unknown_commands(command: str) -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command=command)
    assert result["ok"] is False
    assert "Unknown command" in str(result["message"])


# ---- /login ------------------------------------------------------------------


def test_login_rejects_unknown_provider() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/login huggingface")
    assert result["ok"] is False
    assert "/login azure" in str(result["message"])


def test_login_provider_key_save_sets_default_success_message() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/login openai sk-test")
    assert result["ok"] is True
    assert result["message"] == "Saved openai API key."


def test_login_status_includes_formatted_message() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/login openai")
    assert result["ok"] is True
    assert "openai" in str(result["message"]).lower()


# ---- dispatch basics --------------------------------------------------------------


def test_execute_rejects_empty_command() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="   ")
    assert result["ok"] is False
    assert "Command cannot be empty" in str(result["message"])


def test_execute_rejects_non_slash_command() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="hello")
    assert result["ok"] is False
    assert "Not a slash command" in str(result["message"])


@pytest.mark.parametrize("command", ["/quit", "/exit"])
def test_execute_quit_commands_return_exit(command: str) -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command=command)
    assert result == {"ok": True, "exit": True}


def test_execute_save_sets_default_message_from_path() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/save")
    assert result["ok"] is True
    assert result["message"] == "Saved conversation: /tmp/x.json."


def test_execute_unknown_command_returns_error() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/nonsense")
    assert result["ok"] is False
    assert "Unknown command" in str(result["message"])


def test_help_lists_current_commands_without_template() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/help")
    message = str(result["message"])
    assert "/template" not in message
    assert "/model" in message and "/login" in message


def test_template_command_is_gone() -> None:
    service, _ = _build()
    result = service.execute(session_id="s1", command="/template status")
    assert result["ok"] is False
    assert "Unknown command" in str(result["message"])
