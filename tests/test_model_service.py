from __future__ import annotations

from types import SimpleNamespace

from cortex.app.model_service import ModelService
from cortex.cloud.types import CloudProvider


class _FakeConfig:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}

    def set_state_value(self, key: str, value) -> None:
        self.state[key] = value

    def get_state_value(self, key: str, default=None):
        return self.state.get(key, default)


class _FakeCloudRouter:
    def __init__(self, *, openai_auth: bool = False, anthropic_auth: bool = False) -> None:
        self._auth = {
            CloudProvider.OPENAI: openai_auth,
            CloudProvider.ANTHROPIC: anthropic_auth,
        }

    def get_auth_status(self, provider: CloudProvider):
        return self._auth.get(provider, False), "fake"


def _service(
    *, openai_auth: bool = False, anthropic_auth: bool = False
) -> tuple[ModelService, _FakeConfig]:
    config = _FakeConfig()
    service = ModelService(
        config=config,
        cloud_router=_FakeCloudRouter(openai_auth=openai_auth, anthropic_auth=anthropic_auth),
        credential_store=SimpleNamespace(),
        cloud_catalog=SimpleNamespace(list_models=lambda: []),
    )
    return service, config


def test_select_cloud_model_rejects_when_provider_not_authenticated() -> None:
    service, config = _service(anthropic_auth=False)

    result = service.select_cloud_model(provider="anthropic", model_id="claude-haiku-4-5")

    assert result["ok"] is False
    assert "anthropic is not authenticated" in str(result["message"])
    assert service.active_target.cloud_model is None
    assert config.state == {}


def test_select_cloud_model_sets_active_target_and_state_when_authenticated() -> None:
    service, config = _service(openai_auth=True)

    result = service.select_cloud_model(provider="openai", model_id="gpt-5.1")

    assert result["ok"] is True
    assert result["message"] == "openai:gpt-5.1 — now active. Saved as your model once it answers."
    assert service.active_target.cloud_model is not None
    assert service.active_target.cloud_model.selector == "openai:gpt-5.1"
    service.select_cloud_model(provider="openai", model_id="gpt-5.1")  # picked twice
    # Not in the catalog: saved once it answers, and the picker keeps it then.
    assert config.state == {}
    assert "until it answers" in service.unanswered_hint(service.active_target.cloud_model)
    service.model_answered(service.active_target.cloud_model)
    assert config.state == {
        "last_used_cloud_provider": "openai",
        "last_used_cloud_model": "gpt-5.1",
        "selected_models": ["openai:gpt-5.1"],
    }
    assert service.unanswered_hint(service.active_target.cloud_model) == ""


def test_a_model_saved_before_is_not_held_until_it_answers() -> None:
    """The model a session restores was saved by an earlier one, whether or
    not the picker's list of names still has it."""
    service, config = _service(openai_auth=True)
    config.state = {"last_used_cloud_provider": "openai", "last_used_cloud_model": "gpt-5.1"}

    service.select_cloud_model(provider="openai", model_id="gpt-5.1")

    assert service.unanswered_hint(service.active_target.cloud_model) == ""
    assert config.state["selected_models"] == ["openai:gpt-5.1"]


def test_a_model_that_answered_is_saved_at_once_when_chosen_again() -> None:
    service, config = _service(openai_auth=True)
    for model_id in ("name-a", "name-b"):
        service.select_cloud_model(provider="openai", model_id=model_id)
        service.model_answered(service.active_target.cloud_model)

    service.select_cloud_model(provider="openai", model_id="name-a")

    assert service.unanswered_hint(service.active_target.cloud_model) == ""
    assert config.state["last_used_cloud_model"] == "name-a"


def test_a_model_that_answers_after_another_is_chosen_is_listed_but_not_restored() -> None:
    service, config = _service(openai_auth=True)
    service.select_cloud_model(provider="openai", model_id="name-a")
    first = service.active_target.cloud_model
    service.select_cloud_model(provider="openai", model_id="name-b")

    service.model_answered(first)

    assert config.state == {"selected_models": ["openai:name-a"]}
    assert service.unanswered_hint(first) == ""
