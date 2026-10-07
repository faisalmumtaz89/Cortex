from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

import pytest

from cortex.app.worker_runtime import WorkerRuntime
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider

NO_MODEL_NOTICE = "No model loaded. Pick one with /model, or add a key with /login."


class _FakeConfig:
    def __init__(
        self,
        *,
        state: dict[str, object] | None = None,
        cloud_enabled: bool = True,
        cloud_default_openai_model: str = "gpt-5-nano",
        cloud_default_anthropic_model: str = "claude-sonnet-4-5",
    ) -> None:
        self._state = state or {}
        self.cloud = SimpleNamespace(
            cloud_enabled=cloud_enabled,
            cloud_default_openai_model=cloud_default_openai_model,
            cloud_default_anthropic_model=cloud_default_anthropic_model,
        )

    def get_state_value(self, key: str, default=None):
        return self._state.get(key, default)

    def set_state_value(self, key: str, value) -> None:
        self._state[key] = value


class _FakeModelService:
    def __init__(self, *, cloud_ok: bool = True) -> None:
        self.cloud_ok = cloud_ok
        self.cloud_calls: list[tuple[str, str]] = []
        self.active_target = ActiveModelTarget()

    def select_cloud_model(self, *, provider: str, model_id: str):
        self.cloud_calls.append((provider, model_id))
        if self.cloud_ok:
            self.active_target = ActiveModelTarget.cloud(
                CloudModelRef(provider=CloudProvider.from_value(provider), model_id=model_id)
            )
            return {"ok": True}
        return {"ok": False, "message": "cloud restore failed"}


class _FailAnthropicModelService(_FakeModelService):
    """Anthropic selection fails (no key); every other provider succeeds."""

    def select_cloud_model(self, *, provider: str, model_id: str):
        if provider == "anthropic":
            self.cloud_calls.append((provider, model_id))
            return {
                "ok": False,
                "message": "anthropic is not authenticated. Run /login anthropic <api_key>.",
            }
        return super().select_cloud_model(provider=provider, model_id=model_id)


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


@pytest.fixture
def runtime_factory():
    def _build(
        *,
        state: dict[str, object] | None = None,
        cloud_enabled: bool = True,
        cloud_default_openai_model: str = "gpt-5-nano",
        cloud_default_anthropic_model: str = "claude-sonnet-4-5",
        model_service: _FakeModelService | None = None,
        cloud_router: _FakeCloudRouter | None = None,
        cloud_catalog: _FakeCloudCatalog | None = None,
    ) -> SimpleNamespace:
        runtime = SimpleNamespace(
            config=_FakeConfig(
                state=state,
                cloud_enabled=cloud_enabled,
                cloud_default_openai_model=cloud_default_openai_model,
                cloud_default_anthropic_model=cloud_default_anthropic_model,
            ),
            model_service=model_service or _FakeModelService(),
        )

        if cloud_router is not None:
            runtime.cloud_router = cloud_router
        if cloud_catalog is not None:
            runtime.cloud_catalog = cloud_catalog

        runtime._set_runtime_state_if_supported = MethodType(
            WorkerRuntime._set_runtime_state_if_supported, runtime
        )
        runtime._restore_startup_target = MethodType(WorkerRuntime._restore_startup_target, runtime)
        return runtime

    return _build


def test_restore_startup_target_prefers_cloud_state(runtime_factory) -> None:
    fake_runtime = runtime_factory(
        state={
            "last_used_cloud_provider": "openai",
            "last_used_cloud_model": "gpt-5-nano",
        }
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5-nano")]
    assert notices == ["Restored openai:gpt-5-nano"]
    assert fake_runtime.model_service.active_target.label == "openai:gpt-5-nano"


def test_restore_startup_target_falls_back_to_authenticated_default_cloud(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={
            "last_used_cloud_provider": "anthropic",
            "last_used_cloud_model": "claude-haiku-4-5",
        },
        cloud_default_openai_model="gpt-5.1",
        model_service=_FailAnthropicModelService(),
        cloud_router=_FakeCloudRouter(openai_auth=True, anthropic_auth=False),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [
        ("anthropic", "claude-haiku-4-5"),
        ("openai", "gpt-5.1"),
    ]
    assert notices == [
        "anthropic is not authenticated. Run /login anthropic <api_key>.",
        "Restored openai:gpt-5.1",
    ]


def test_restore_startup_target_fallback_uses_catalog_when_no_configured_default(
    runtime_factory,
) -> None:
    """A failed saved selection falls back to the authenticated provider's first
    catalog model when no default is configured for that provider."""
    fake_runtime = runtime_factory(
        state={
            "last_used_cloud_provider": "anthropic",
            "last_used_cloud_model": "claude-haiku-4-5",
        },
        cloud_default_openai_model="",
        model_service=_FailAnthropicModelService(),
        cloud_router=_FakeCloudRouter(openai_auth=True, anthropic_auth=False),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [
        ("anthropic", "claude-haiku-4-5"),
        ("openai", "gpt-5-nano"),
    ]
    assert notices[-1] == "Restored openai:gpt-5-nano"
    assert fake_runtime.config._state["last_used_cloud_provider"] == "anthropic"


def test_restore_startup_target_reports_no_model_when_state_empty(runtime_factory) -> None:
    fake_runtime = runtime_factory(state={}, cloud_enabled=True)

    notices = fake_runtime._restore_startup_target()
    assert notices == [NO_MODEL_NOTICE]
    assert fake_runtime.model_service.cloud_calls == []
    assert fake_runtime.model_service.active_target.cloud_model is None
    assert fake_runtime.model_service.active_target.label == "No model loaded"


def test_restore_startup_target_uses_authenticated_cloud_default_when_no_state(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={},
        cloud_enabled=True,
        cloud_default_openai_model="gpt-5.1",
        cloud_router=_FakeCloudRouter(openai_auth=True),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5.1")]
    assert notices == ["Restored openai:gpt-5.1"]


def test_restore_startup_target_clears_stale_cloud_state_when_nothing_restores(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={
            "last_used_cloud_provider": "anthropic",
            "last_used_cloud_model": "claude-haiku-4-5",
        },
        model_service=_FakeModelService(cloud_ok=False),
        cloud_router=_FakeCloudRouter(openai_auth=False, anthropic_auth=False),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert notices == ["cloud restore failed", NO_MODEL_NOTICE]
    assert fake_runtime.model_service.active_target.cloud_model is None
    assert fake_runtime.config._state.get("last_used_cloud_provider") == ""
    assert fake_runtime.config._state.get("last_used_cloud_model") == ""


# ---- /update cortex background narration ------------------------------------


def _update_runtime(command_service) -> tuple[SimpleNamespace, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    fake = SimpleNamespace(
        command_service=command_service,
        _update_task_lock=threading.Lock(),
        _active_update_thread=None,
    )
    fake._emit_event = lambda *, session_id, event_type, payload: events.append(
        (event_type, payload)
    )
    return fake, events


def test_run_background_update_streams_phases_and_finishes() -> None:
    class _FakeCommandService:
        def execute(self, *, session_id, command, progress_callback=None):
            assert command == "/update cortex"
            progress_callback(
                {"kind": "engine-update", "repo_id": "cortex", "phase": "verifying checksum"}
            )
            return {"ok": True, "message": "Cortex 9.9.9 installed — restart Cortex to apply."}

    fake, events = _update_runtime(_FakeCommandService())
    MethodType(WorkerRuntime._run_background_update, fake)(session_id="s1")

    frames = [payload for kind, payload in events if kind == "message.updated"]
    assert [frame["progress"]["phase"] for frame in frames] == [
        "starting",
        "verifying checksum",
        "ready",
    ]
    for frame in frames:
        assert frame["progress"]["kind"] == "engine-update"
        assert frame["progress"]["repo_id"] == "cortex"
        assert frame["role"] == "system"
    # One operation = one transcript message, resolved in place.
    assert len({frame["message_id"] for frame in frames}) == 1
    assert [frame["final"] for frame in frames] == [False, False, True]
    assert frames[-1]["content"] == "Cortex 9.9.9 installed — restart Cortex to apply."
    # Background operations are not turns: session.status is never touched and
    # the final frame is the only record (no duplicate system.notice).
    assert not any(kind == "session.status" for kind, _ in events)
    assert not any(kind == "system.notice" for kind, _ in events)


def test_run_background_update_reports_failure() -> None:
    class _FakeCommandService:
        def execute(self, **_kwargs):
            return {"ok": False, "message": "Cortex update failed: download failed: offline"}

    fake, events = _update_runtime(_FakeCommandService())
    MethodType(WorkerRuntime._run_background_update, fake)(session_id="s1")

    final_frames = [p for kind, p in events if kind == "message.updated" and p.get("final")]
    assert len(final_frames) == 1
    assert final_frames[0]["progress"]["phase"] == "failed"
    assert final_frames[0]["content"] == "Cortex update failed: download failed: offline"
    assert not any(kind == "session.status" for kind, _ in events)
