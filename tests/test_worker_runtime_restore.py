from __future__ import annotations

import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

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

    def set_state_value(self, key: str, value) -> None:
        self._state[key] = value


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


@pytest.fixture
def runtime_factory():
    def _build(
        *,
        state: dict[str, object] | None = None,
        cloud_enabled: bool = True,
        cloud_default_openai_model: str = "gpt-5-nano",
        cloud_default_anthropic_model: str = "claude-sonnet-4-5",
        last_used_model: str = "",
        default_model: str = "",
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
                last_used_model=last_used_model,
                default_model=default_model,
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
        runtime._normalize_startup_state = MethodType(
            WorkerRuntime._normalize_startup_state, runtime
        )
        runtime._restore_startup_target = MethodType(WorkerRuntime._restore_startup_target, runtime)
        return runtime

    return _build


def test_restore_startup_target_prefers_cloud_state(runtime_factory) -> None:
    fake_runtime = runtime_factory(
        state={
            "last_used_backend": "cloud",
            "last_used_cloud_provider": "openai",
            "last_used_cloud_model": "gpt-5-nano",
        }
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5-nano")]
    assert notices == ["Restored cloud model: openai:gpt-5-nano"]


def test_restore_startup_target_falls_back_to_local_when_cloud_restore_fails(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={
            "last_used_backend": "cloud",
            "last_used_cloud_provider": "anthropic",
            "last_used_cloud_model": "claude-haiku-4-5",
        },
        last_used_model="Qwen--Qwen2.5-0.5B-Instruct",
        model_service=_FakeModelService(cloud_ok=False, local_ok=True),
        cloud_router=_FakeCloudRouter(openai_auth=True),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [("anthropic", "claude-haiku-4-5")]
    assert fake_runtime.model_service.local_calls == ["Qwen--Qwen2.5-0.5B-Instruct"]
    assert notices == [
        "cloud restore failed",
        "Restored local model: Qwen--Qwen2.5-0.5B-Instruct",
    ]


def test_restore_startup_target_falls_back_to_authenticated_default_cloud_when_local_unset(
    runtime_factory,
) -> None:
    class _FailAnthropicThenAllowOpenAIModelService(_FakeModelService):
        def select_cloud_model(self, *, provider: str, model_id: str):
            self.cloud_calls.append((provider, model_id))
            if provider == "anthropic":
                return {
                    "ok": False,
                    "message": "anthropic is not authenticated. Run /login anthropic <api_key>.",
                }
            self.active_target = ActiveModelTarget.cloud(
                CloudModelRef(provider=CloudProvider.from_value(provider), model_id=model_id)
            )
            return {"ok": True}

    fake_runtime = runtime_factory(
        state={
            "last_used_backend": "cloud",
            "last_used_cloud_provider": "anthropic",
            "last_used_cloud_model": "claude-haiku-4-5",
        },
        cloud_default_openai_model="gpt-5.1",
        last_used_model="",
        default_model="",
        model_service=_FailAnthropicThenAllowOpenAIModelService(),
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
        "Restored cloud model: openai:gpt-5.1",
    ]


def test_restore_startup_target_reports_no_model_when_state_empty(runtime_factory) -> None:
    fake_runtime = runtime_factory(
        state={},
        cloud_enabled=True,
        last_used_model="",
        default_model="",
    )

    notices = fake_runtime._restore_startup_target()
    assert notices == ["No model loaded. Use /model to select local or cloud."]
    assert fake_runtime.model_service.active_target.backend == "local"
    assert fake_runtime.model_service.active_target.local_model is None


def test_restore_startup_target_uses_authenticated_cloud_default_when_no_state(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={},
        cloud_enabled=True,
        cloud_default_openai_model="gpt-5.1",
        last_used_model="",
        default_model="",
        cloud_router=_FakeCloudRouter(openai_auth=True),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5.1")]
    assert notices == ["Restored cloud model: openai:gpt-5.1"]


def test_restore_startup_target_clears_target_on_local_restore_failure(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={"last_used_backend": "local"},
        last_used_model="missing-model",
        model_service=_FakeModelService(local_ok=False),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.local_calls == ["missing-model"]
    assert notices == [
        "local restore failed",
        "No model loaded. Use /model to select local or cloud.",
    ]
    assert fake_runtime.model_service.active_target.backend == "local"
    assert fake_runtime.model_service.active_target.local_model is None
    assert fake_runtime.config._state.get("last_used_backend") == "local"
    assert fake_runtime.config._state.get("last_used_model") == ""


def test_restore_startup_target_local_failure_can_fallback_to_authenticated_cloud_default(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={"last_used_backend": "local"},
        cloud_default_openai_model="gpt-5.1",
        last_used_model="missing-model",
        default_model="",
        model_service=_FakeModelService(local_ok=False, cloud_ok=True),
        cloud_router=_FakeCloudRouter(openai_auth=True, anthropic_auth=False),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert fake_runtime.model_service.local_calls == ["missing-model"]
    assert fake_runtime.model_service.cloud_calls == [("openai", "gpt-5.1")]
    assert notices == [
        "local restore failed",
        "Restored cloud model: openai:gpt-5.1",
    ]


def test_restore_startup_target_clears_stale_cloud_state_when_nothing_restores(
    runtime_factory,
) -> None:
    fake_runtime = runtime_factory(
        state={
            "last_used_backend": "cloud",
            "last_used_cloud_provider": "anthropic",
            "last_used_cloud_model": "claude-haiku-4-5",
            "last_used_model": "missing-local",
        },
        last_used_model="missing-local",
        default_model="",
        model_service=_FakeModelService(local_ok=False, cloud_ok=False),
        cloud_router=_FakeCloudRouter(openai_auth=False, anthropic_auth=False),
        cloud_catalog=_FakeCloudCatalog(),
    )

    notices = fake_runtime._restore_startup_target()
    assert notices == [
        "cloud restore failed",
        "local restore failed",
        "No model loaded. Use /model to select local or cloud.",
    ]
    assert fake_runtime.config._state.get("last_used_backend") == "local"
    assert fake_runtime.config._state.get("last_used_cloud_provider") == ""
    assert fake_runtime.config._state.get("last_used_cloud_model") == ""
    assert fake_runtime.config._state.get("last_used_model") == ""


def test_format_download_progress_message_strips_trailing_colon_and_skips_invalid_done_bytes() -> (
    None
):
    message = WorkerRuntime._format_download_progress_message(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        payload={
            "description": "Download complete:",
            "unit": "B",
            "bytes_downloaded": 0,
            "bytes_total": 12_000_000,
            "percent": 0.0,
            "done": True,
        },
    )

    assert message == "Download complete"


def test_format_download_progress_message_handles_download_complete_label_without_done_flag() -> (
    None
):
    message = WorkerRuntime._format_download_progress_message(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        payload={
            "description": "Download complete:",
            "unit": "B",
            "bytes_downloaded": 0,
            "bytes_total": 12_000_000,
            "percent": 0.0,
            "done": False,
        },
    )

    assert message == "Download complete"


def test_format_download_progress_message_keeps_file_progress_readable() -> None:
    message = WorkerRuntime._format_download_progress_message(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        payload={
            "description": "Fetching 10 files",
            "unit": "it",
            "bytes_downloaded": 4,
            "bytes_total": 10,
            "percent": 40.0,
            "done": False,
        },
    )

    assert message == "Fetching 10 files: 40.0% (4/10 files)"


def test_format_download_progress_message_supports_fractional_file_progress() -> None:
    message = WorkerRuntime._format_download_progress_message(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        payload={
            "description": "Fetching 16 files",
            "unit": "files",
            "units_downloaded": 0.8,
            "units_total": 16.0,
            "bytes_downloaded": 0,
            "bytes_total": 16,
            "percent": 5.0,
            "done": False,
        },
    )

    assert message == "Fetching 16 files: 5.0% (0.8/16 files)"


def test_build_download_progress_payload_never_reports_non_terminal_full_completion() -> None:
    payload = WorkerRuntime._build_download_progress_payload(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        phase="transferring",
        bytes_downloaded=7_300,
        bytes_total=7_340,
        allow_bytes_percent=True,
        files_completed=16.0,
        files_total=16.0,
        speed_bps=100.0,
        eta_seconds=1.0,
        elapsed_seconds=10.0,
        stalled=False,
    )

    assert payload["phase"] == "transferring"
    assert payload["percent"] == 99.455
    assert payload["eta_seconds"] == 1.0


def test_build_download_progress_payload_completed_normalizes_terminal_values() -> None:
    payload = WorkerRuntime._build_download_progress_payload(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        phase="completed",
        bytes_downloaded=7_300,
        bytes_total=7_340,
        allow_bytes_percent=True,
        files_completed=15.0,
        files_total=16.0,
        speed_bps=100.0,
        eta_seconds=1.0,
        elapsed_seconds=10.0,
        stalled=True,
    )

    assert payload["phase"] == "completed"
    assert payload["bytes_downloaded"] == 7_340
    assert payload["files_completed"] == 16.0
    assert payload["percent"] == 100.0
    assert payload["eta_seconds"] is None
    assert payload["stalled"] is False


def test_build_download_progress_payload_finalizing_reports_completion_without_eta() -> None:
    payload = WorkerRuntime._build_download_progress_payload(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        phase="finalizing",
        bytes_downloaded=7_340,
        bytes_total=7_340,
        allow_bytes_percent=True,
        files_completed=16.0,
        files_total=16.0,
        speed_bps=100.0,
        eta_seconds=1.0,
        elapsed_seconds=10.0,
        stalled=True,
    )

    assert payload["phase"] == "finalizing"
    assert payload["percent"] == 100.0
    assert payload["eta_seconds"] is None
    assert payload["stalled"] is False


def test_format_download_progress_content_finalizing_is_explicit() -> None:
    message = WorkerRuntime._format_download_progress_content(
        repo_id="hf-internal-testing/tiny-random-gpt2",
        progress={
            "phase": "finalizing",
            "bytes_downloaded": 7_340,
            "bytes_total": 7_340,
            "percent": 100.0,
            "speed_bps": 100.0,
            "eta_seconds": None,
            "files_completed": 16.0,
            "files_total": 16.0,
            "stalled": False,
        },
    )

    assert message == "Finalizing download: hf-internal-testing/tiny-random-gpt2"


def test_directory_size_bytes_counts_nested_files(tmp_path: Path) -> None:
    root = tmp_path / "download"
    (root / "nested").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"1234")
    (root / "nested" / "b.bin").write_bytes(b"12345")

    assert WorkerRuntime._directory_size_bytes(root) == 9


def test_directory_size_bytes_handles_file_target(tmp_path: Path) -> None:
    target = tmp_path / "single.bin"
    target.write_bytes(b"abcdef")

    assert WorkerRuntime._directory_size_bytes(target) == 6


def test_run_background_download_emits_heartbeat_when_progress_is_silent(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class _SilentCommandService:
        def execute(
            self, *, session_id: str, command: str, progress_callback, cancel_requested
        ):  # noqa: ANN001
            time.sleep(0.2)
            return {"ok": True, "message": "Downloaded to /tmp/fake-model"}

    fake_runtime = SimpleNamespace(
        command_service=_SilentCommandService(),
        _download_cancel_event=threading.Event(),
        _download_task_lock=threading.Lock(),
        _active_download_thread=None,
        _download_progress_sample_interval_seconds=0.05,
        _download_progress_heartbeat_seconds=0.05,
        _download_progress_stall_seconds=0.05,
        _directory_size_bytes=lambda _path: 0,
        _format_download_progress_content=WorkerRuntime._format_download_progress_content,
        _build_download_progress_payload=WorkerRuntime._build_download_progress_payload,
        _notice_message_for_result=WorkerRuntime._notice_message_for_result,
        _emit_event=lambda *, session_id, event_type, payload: events.append((event_type, payload)),
    )

    WorkerRuntime._run_background_download(
        fake_runtime,
        session_id="session-1",
        command="/download user/repo",
        repo_id="user/repo",
        download_monitor_path=str(tmp_path),
    )

    heartbeat_payloads = [
        payload for event_type, payload in events if event_type == "message.updated"
    ]
    assert any(
        str(item.get("content", "")).startswith("Preparing download:")
        for item in heartbeat_payloads
    )
    assert any(isinstance(item.get("progress"), dict) for item in heartbeat_payloads)


def test_run_background_download_heartbeat_not_blocked_by_stale_progress_callbacks(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class _NoisyStaleProgressCommandService:
        def execute(
            self, *, session_id: str, command: str, progress_callback, cancel_requested
        ):  # noqa: ANN001
            payload = {
                "description": "Fetching 16 files",
                "unit": "files",
                "units_downloaded": 0.0,
                "units_total": 16.0,
                "bytes_downloaded": 0,
                "bytes_total": 16,
                "percent": 0.0,
                "done": False,
            }
            for _ in range(12):
                progress_callback(payload)
                time.sleep(0.02)
            return {"ok": True, "message": "Downloaded to /tmp/fake-model"}

    fake_runtime = SimpleNamespace(
        command_service=_NoisyStaleProgressCommandService(),
        _download_cancel_event=threading.Event(),
        _download_task_lock=threading.Lock(),
        _active_download_thread=None,
        _download_progress_sample_interval_seconds=0.05,
        _download_progress_heartbeat_seconds=0.05,
        _download_progress_stall_seconds=0.05,
        _directory_size_bytes=lambda _path: 0,
        _format_download_progress_content=WorkerRuntime._format_download_progress_content,
        _build_download_progress_payload=WorkerRuntime._build_download_progress_payload,
        _notice_message_for_result=WorkerRuntime._notice_message_for_result,
        _emit_event=lambda *, session_id, event_type, payload: events.append((event_type, payload)),
    )

    WorkerRuntime._run_background_download(
        fake_runtime,
        session_id="session-1",
        command="/download user/repo",
        repo_id="user/repo",
        download_monitor_path=str(tmp_path),
    )

    message_updates = [payload for event_type, payload in events if event_type == "message.updated"]
    assert any(
        isinstance(payload.get("progress"), dict)
        and payload["progress"].get("files_total") == 16.0
        and payload["progress"].get("percent") == 0.0
        for payload in message_updates
    )
    assert any(
        "waiting for transfer" in str(payload.get("content", "")) for payload in message_updates
    )


def test_run_background_download_heartbeat_survives_description_noise(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class _NoisyDescriptionCommandService:
        def execute(
            self, *, session_id: str, command: str, progress_callback, cancel_requested
        ):  # noqa: ANN001
            for idx in range(20):
                progress_callback(
                    {
                        "description": f"Fetching 16 files ({idx})",
                        "unit": "files",
                        "units_downloaded": 0.0,
                        "units_total": 16.0,
                        "bytes_downloaded": 0,
                        "bytes_total": 16,
                        "percent": 0.0,
                        "done": False,
                    }
                )
                time.sleep(0.02)
            return {"ok": True, "message": "Downloaded to /tmp/fake-model"}

    fake_runtime = SimpleNamespace(
        command_service=_NoisyDescriptionCommandService(),
        _download_cancel_event=threading.Event(),
        _download_task_lock=threading.Lock(),
        _active_download_thread=None,
        _download_progress_sample_interval_seconds=0.05,
        _download_progress_heartbeat_seconds=0.05,
        _download_progress_stall_seconds=0.05,
        _directory_size_bytes=lambda _path: 0,
        _format_download_progress_content=WorkerRuntime._format_download_progress_content,
        _build_download_progress_payload=WorkerRuntime._build_download_progress_payload,
        _notice_message_for_result=WorkerRuntime._notice_message_for_result,
        _emit_event=lambda *, session_id, event_type, payload: events.append((event_type, payload)),
    )

    WorkerRuntime._run_background_download(
        fake_runtime,
        session_id="session-1",
        command="/download user/repo",
        repo_id="user/repo",
        download_monitor_path=str(tmp_path),
    )

    message_updates = [payload for event_type, payload in events if event_type == "message.updated"]
    assert any(
        "waiting for transfer" in str(payload.get("content", "")) for payload in message_updates
    )
