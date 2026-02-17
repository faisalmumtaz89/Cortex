from __future__ import annotations

from types import SimpleNamespace

from cortex.app.worker_runtime import WorkerRuntime
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider


class _FakeConfig:
    def __init__(
        self,
        *,
        state: dict[str, object] | None = None,
        cloud_enabled: bool = True,
        cloud_default_openai_model: str = "gpt-5-nano",
        cloud_default_anthropic_model: str = "claude-sonnet-4-5",
        last_used_model: str = "",
        default_model: str = "",
    ) -> None:
        self._state = state or {}
        self.cloud = SimpleNamespace(
            cloud_enabled=cloud_enabled,
            cloud_default_openai_model=cloud_default_openai_model,
            cloud_default_anthropic_model=cloud_default_anthropic_model,
        )
        self.model = SimpleNamespace(last_used_model=last_used_model, default_model=default_model)

    def get_state_value(self, key: str, default=None):
        return self._state.get(key, default)


class _FakeModelService:
    def __init__(self, *, cloud_ok: bool = True, local_ok: bool = True) -> None:
        self.cloud_ok = cloud_ok
        self.local_ok = local_ok
        self.cloud_calls: list[tuple[str, str]] = []
        self.local_calls: list[str] = []
        self.active_target = ActiveModelTarget.local(model_name=None)

    def select_cloud_model(self, *, provider: str, model_id: str):
        self.cloud_calls.append((provider, model_id))
        if self.cloud_ok:
            self.active_target = ActiveModelTarget.cloud(
                CloudModelRef(provider=CloudProvider.from_value(provider), model_id=model_id)
            )
            return {"ok": True}
        return {"ok": False, "message": "cloud restore failed"}

    def select_local_model(self, model_name_or_path: str):
        self.local_calls.append(model_name_or_path)
        if self.local_ok:
            self.active_target = ActiveModelTarget.local(model_name=model_name_or_path)
            return {"ok": True}
        return {"ok": False, "message": "local restore failed"}

    def get_active_model_label(self) -> str:
        if self.active_target.backend == "cloud" and self.active_target.cloud_model:
            return self.active_target.cloud_model.selector
        return self.active_target.local_model or "No model loaded"


class _FakeCloudRouter:
    def __init__(self, *, openai_auth: bool = False, anthropic_auth: bool = False) -> None:
        self._auth = {
            CloudProvider.OPENAI: openai_auth,
            CloudProvider.ANTHROPIC: anthropic_auth,
        }

    def get_auth_status(self, provider: CloudProvider):
        return self._auth.get(provider, False), "fake"


class _FakeCloudCatalog:
    def list_models(self):
        return [
            CloudModelRef(provider=CloudProvider.OPENAI, model_id="gpt-5-nano"),
            CloudModelRef(provider=CloudProvider.ANTHROPIC, model_id="claude-sonnet-4-5"),
        ]


def test_restore_startup_target_prefers_cloud_state() -> None:
    fake_runtime = SimpleNamespace(
        config=_FakeConfig(
            state={
                "last_used_backend": "cloud",
                "last_used_cloud_provider": "openai",
                "last_used_cloud_model": "gpt-5-nano",
            }
        ),
        model_service=_FakeModelService(),
    )

    notices = WorkerRuntime._restore_startup_target(fake_runtime)
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5-nano")]
    assert notices == ["Restored cloud model: openai:gpt-5-nano"]


def test_restore_startup_target_reports_no_model_when_state_empty() -> None:
    fake_runtime = SimpleNamespace(
        config=_FakeConfig(state={}, cloud_enabled=True, last_used_model="", default_model=""),
        model_service=_FakeModelService(),
    )

    notices = WorkerRuntime._restore_startup_target(fake_runtime)
    assert notices == ["No model loaded. Use /model to select local or cloud."]
    assert fake_runtime.model_service.active_target.backend == "local"
    assert fake_runtime.model_service.active_target.local_model is None


def test_restore_startup_target_uses_authenticated_cloud_default_when_no_state() -> None:
    fake_runtime = SimpleNamespace(
        config=_FakeConfig(
            state={},
            cloud_enabled=True,
            cloud_default_openai_model="gpt-5.1",
            last_used_model="",
            default_model="",
        ),
        model_service=_FakeModelService(),
        cloud_router=_FakeCloudRouter(openai_auth=True),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = WorkerRuntime._restore_startup_target(fake_runtime)
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5.1")]
    assert notices == ["Restored cloud model: openai:gpt-5.1"]


def test_restore_startup_target_clears_target_on_local_restore_failure() -> None:
    fake_runtime = SimpleNamespace(
        config=_FakeConfig(state={"last_used_backend": "local"}, last_used_model="missing-model"),
        model_service=_FakeModelService(local_ok=False),
    )

    notices = WorkerRuntime._restore_startup_target(fake_runtime)
    assert fake_runtime.model_service.local_calls == ["missing-model"]
    assert notices == ["local restore failed"]
    assert fake_runtime.model_service.active_target.backend == "local"
    assert fake_runtime.model_service.active_target.local_model is None
