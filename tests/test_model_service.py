from __future__ import annotations

from types import SimpleNamespace

from cortex.app.model_service import ModelService
from cortex.cloud.types import CloudProvider


class _FakeConfig:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}

    def set_state_value(self, key: str, value: str) -> None:
        self.state[key] = value


class _FakeModelManager:
    current_model = None

    def discover_available_models(self):
        return []

    def list_models(self):
        return []


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
        model_manager=_FakeModelManager(),
        gpu_validator=SimpleNamespace(),
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
    assert service.active_target.backend == "local"
    assert config.state == {}


def test_select_cloud_model_sets_active_target_and_state_when_authenticated() -> None:
    service, config = _service(openai_auth=True)

    result = service.select_cloud_model(provider="openai", model_id="gpt-5.1")

    assert result["ok"] is True
    assert result["message"] == "Active cloud model set to openai:gpt-5.1"
    assert service.active_target.backend == "cloud"
    assert service.active_target.cloud_model is not None
    assert service.active_target.cloud_model.selector == "openai:gpt-5.1"
    assert config.state == {
        "last_used_backend": "cloud",
        "last_used_cloud_provider": "openai",
        "last_used_cloud_model": "gpt-5.1",
    }
