from __future__ import annotations

import threading
import time
from types import MethodType, SimpleNamespace

import pytest

from cortex.app.worker_runtime import WorkerRuntime
from cortex.cloud.types import ActiveModelTarget, CloudModelRef, CloudProvider
from cortex.lumen_runtime import SWITCH_IN_FLIGHT_PREFIX


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

    def restore_local_target(self, selector: str):
        self.local_calls.append(selector)
        if self.local_ok:
            self.active_target = ActiveModelTarget.local(model_name=selector)
            return selector
        return None

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

        runtime._boot_task_lock = threading.Lock()
        runtime._active_boot_thread = None
        runtime._active_boot_selector = None
        runtime._model_choice_seq = 0
        runtime._is_boot_active = MethodType(WorkerRuntime._is_boot_active, runtime)
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
    assert notices == ["Restored cloud · openai:gpt-5-nano"]


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
        "Restored local · Qwen--Qwen2.5-0.5B-Instruct (server starts on first use)",
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
        "Restored cloud · openai:gpt-5.1",
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
    assert notices == ["Restored cloud · openai:gpt-5.1"]


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
        "Failed to restore local · missing-model",
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
        "Failed to restore local · missing-model",
        "Restored cloud · openai:gpt-5.1",
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
        "Failed to restore local · missing-local",
        "No model loaded. Use /model to select local or cloud.",
    ]
    assert fake_runtime.config._state.get("last_used_backend") == "local"
    assert fake_runtime.config._state.get("last_used_cloud_provider") == ""
    assert fake_runtime.config._state.get("last_used_cloud_model") == ""
    assert fake_runtime.config._state.get("last_used_model") == ""


def test_run_background_download_streams_lines_and_finishes(runtime_factory) -> None:
    events: list[tuple[str, dict]] = []

    class _FakeCommandService:
        def execute(
            self,
            *,
            session_id,
            command,
            progress_callback=None,
            cancel_requested=None,
            activation_guard=None,
        ):
            assert command == "/download qwen3-6-27b:q4_0"
            progress_callback(
                {
                    "kind": "download",
                    "repo_id": "qwen3-6-27b:q4_0",
                    "phase": "downloading 50%",
                    "bytes_downloaded": 0,
                }
            )
            return {"ok": True, "message": "Downloaded qwen3-6-27b:q4_0."}

    fake_runtime = runtime_factory()
    fake_runtime._model_choice_seq = 0
    fake_runtime.command_service = _FakeCommandService()
    fake_runtime._download_cancel_event = threading.Event()
    fake_runtime._emit_event = lambda *, session_id, event_type, payload: events.append(
        (event_type, payload)
    )
    bound = MethodType(WorkerRuntime._run_background_download, fake_runtime)

    bound(session_id="s1", command="/download qwen3-6-27b:q4_0", repo_id="qwen3-6-27b:q4_0", choice_seq=0)

    progress_events = [p for kind, p in events if kind == "message.updated"]
    assert progress_events, "expected streamed progress frames"
    for payload in progress_events:
        progress = payload["progress"]
        assert progress["kind"] == "download"
        assert progress["repo_id"] == "qwen3-6-27b:q4_0"
        assert isinstance(progress["bytes_downloaded"], int)
    assert progress_events[-1]["final"] is True
    # The final frame carries the outcome as its content (no separate fallback).
    assert "Downloaded qwen3-6-27b:q4_0." in str(progress_events[-1]["content"])
    # No duplicate system.notice: the operation's own final frame is the record.
    assert not any(
        kind == "system.notice" and "Downloaded" in str(p.get("message", ""))
        for kind, p in events
    )
    # Background model ops are NOT turns: session.status is never touched, so
    # the generic Working… spinner cannot run for them.
    assert not any(kind == "session.status" for kind, _ in events)


def test_background_download_transitions_to_model_load_stage(runtime_factory) -> None:
    """The chained auto-load flips the SAME progress message from the download
    indicator to the GPU-load indicator (kind download → model-load)."""
    events: list[tuple[str, dict]] = []

    class _FakeCommandService:
        def execute(self, *, progress_callback=None, **_kwargs):
            progress_callback({"kind": "download", "repo_id": "m:q4_0", "phase": "Downloading: https://x"})
            progress_callback({"kind": "model-load", "repo_id": "m:q4_0", "phase": "loading"})
            return {"ok": True, "message": "local · m:q4_0 ready — now active."}

    fake_runtime = runtime_factory()
    fake_runtime._model_choice_seq = 0
    fake_runtime.command_service = _FakeCommandService()
    fake_runtime._download_cancel_event = threading.Event()
    fake_runtime._emit_event = lambda *, session_id, event_type, payload: events.append(
        (event_type, payload)
    )
    bound = MethodType(WorkerRuntime._run_background_download, fake_runtime)
    bound(session_id="s1", command="/download m:q4_0", repo_id="m:q4_0", choice_seq=0)

    frames = [p for kind, p in events if kind == "message.updated"]
    kinds = [f["progress"]["kind"] for f in frames]
    assert "download" in kinds and "model-load" in kinds
    assert kinds.index("download") < kinds.index("model-load")
    load_frames = [f for f in frames if f["progress"]["kind"] == "model-load" and not f["final"]]
    assert load_frames, "expected a live model-load frame"
    # The load stage never claims bytes and announces the transition.
    assert "bytes_downloaded" not in load_frames[0]["progress"]
    assert "loading" in str(load_frames[0]["content"]).lower()
    assert str(load_frames[0]["content"]).startswith("Loading ")
    # All frames share ONE message id (a transition, not a second message).
    assert len({f["message_id"] for f in frames}) == 1


def test_background_download_polls_real_part_bytes(runtime_factory, tmp_path, monkeypatch) -> None:
    """Bytes come from the lumen cache's .part files (pull's own bar is
    TTY-only): a growing .part must surface as bytes_downloaded > 0."""
    monkeypatch.setenv("LUMEN_CACHE_DIR", str(tmp_path))
    events: list[tuple[str, dict]] = []
    part = tmp_path / "qwen.gguf.part"

    class _FakeCommandService:
        def execute(self, *, progress_callback=None, **_kwargs):
            part.write_bytes(b"x" * 4096)
            time.sleep(1.4)  # let the 1s poller observe the bytes
            return {"ok": True, "message": "Downloaded m:q4_0."}

    fake_runtime = runtime_factory()
    fake_runtime._model_choice_seq = 0
    fake_runtime.command_service = _FakeCommandService()
    fake_runtime._download_cancel_event = threading.Event()
    fake_runtime._emit_event = lambda *, session_id, event_type, payload: events.append(
        (event_type, payload)
    )
    bound = MethodType(WorkerRuntime._run_background_download, fake_runtime)
    bound(session_id="s1", command="/download m:q4_0", repo_id="m:q4_0", choice_seq=0)

    frames = [p for kind, p in events if kind == "message.updated"]
    live_bytes = [
        f["progress"].get("bytes_downloaded", 0)
        for f in frames
        if f["progress"]["kind"] == "download"
    ]
    assert any(b >= 4096 for b in live_bytes), live_bytes


def test_run_background_download_reports_failure(runtime_factory) -> None:
    events: list[tuple[str, dict]] = []

    class _FakeCommandService:
        def execute(self, **_kwargs):
            return {"ok": False, "message": "Download failed: network unreachable"}

    fake_runtime = runtime_factory()
    fake_runtime._model_choice_seq = 0
    fake_runtime.command_service = _FakeCommandService()
    fake_runtime._download_cancel_event = threading.Event()
    fake_runtime._emit_event = lambda *, session_id, event_type, payload: events.append(
        (event_type, payload)
    )
    bound = MethodType(WorkerRuntime._run_background_download, fake_runtime)

    bound(session_id="s1", command="/download x", repo_id="x", choice_seq=0)

    final_frames = [p for kind, p in events if kind == "message.updated" and p.get("final")]
    assert final_frames and final_frames[-1]["progress"]["phase"] == "failed"
    assert not any(kind == "session.status" for kind, _ in events)


# ---- /model one-flow intercept (uncached → auto-downloading select) ----------


def _intercept_runtime(
    *,
    resolved: dict | None,
    command_result: dict | None = None,
    active_repo: str | None = None,
):
    """Minimal WorkerRuntime stand-in wired with the REAL intercept, download
    tracking, and background runner so the one-flow path runs end to end."""
    events: list[tuple[str, dict]] = []
    executed: list[str] = []

    class _Commands:
        def execute(
            self,
            *,
            session_id,
            command,
            progress_callback=None,
            cancel_requested=None,
            activation_guard=None,
        ):
            executed.append(command)
            return dict(command_result or {"ok": True, "message": "qwen3-6-27b:q4_0 ready — now active."})

    class _Models:
        def resolve_local_selector(self, selector):
            return dict(resolved) if resolved is not None else {"ok": False, "message": "unknown"}

    fake = SimpleNamespace(
        model_service=_Models(),
        command_service=_Commands(),
        lumen_runtime=SimpleNamespace(
            serving_selector=lambda: None,
            starting_selector=lambda: None,
        ),
        _download_task_lock=threading.Lock(),
        _active_download_thread=None,
        _active_download_repo_id=None,
        _download_cancel_event=threading.Event(),
        _boot_task_lock=threading.Lock(),
        _active_boot_thread=None,
        _active_boot_selector=None,
        _model_choice_seq=0,
    )
    fake._emit_event = lambda *, session_id, event_type, payload: events.append(
        (event_type, payload)
    )
    for name in (
        "_current_download_repo_id",
        "_download_busy_message",
        "_maybe_intercept_model_download",
        "_intercept_cached_model_boot",
        "_start_background_download",
        "_run_background_download",
        "_start_background_boot",
        "_run_background_boot",
        "_is_download_active",
        "_is_boot_active",
        "_current_boot_selector",
        "_booting_selector",
    ):
        setattr(fake, name, MethodType(getattr(WorkerRuntime, name), fake))

    if active_repo is not None:
        hold = threading.Event()
        thread = threading.Thread(target=hold.wait, daemon=True)
        thread.start()
        fake._active_download_thread = thread
        fake._active_download_repo_id = active_repo
        fake._hold = hold  # released by the test
    return fake, events, executed


def _uncached(selector: str) -> dict:
    return {"ok": True, "model": SimpleNamespace(cached=False, selector=selector)}


def test_model_intercept_starts_auto_loading_download(runtime_factory) -> None:
    fake, events, executed = _intercept_runtime(resolved=_uncached("qwen3-6-27b:q4_0"))

    result = fake._maybe_intercept_model_download(
        session_id="s1", selector_arg="qwen3-6-27b:q4_0"
    )

    assert result is not None and result["ok"] is True
    assert result["background"] is True
    assert "Downloading qwen3-6-27b:q4_0 — loads automatically when done" in str(result["message"])
    assert "/download" not in str(result["message"]).replace("/download cancel", "")

    fake._active_download_thread.join(timeout=10)
    assert executed == ["/download qwen3-6-27b:q4_0"]
    finals = [
        p for kind, p in events
        if kind == "message.updated" and p.get("final") is True
    ]
    assert any("ready — now active" in str(p.get("content", "")) for p in finals)
    assert not any(kind == "session.status" for kind, _ in events)


def test_model_intercept_same_selector_reports_already_downloading() -> None:
    fake, events, executed = _intercept_runtime(
        resolved=_uncached("qwen3-6-27b:q4_0"), active_repo="qwen3-6-27b:q4_0"
    )
    try:
        result = fake._maybe_intercept_model_download(
            session_id="s1", selector_arg="qwen3-6-27b:q4_0"
        )
        assert result is not None and result["ok"] is True
        assert "Already downloading qwen3-6-27b:q4_0" in str(result["message"])
        assert executed == []
    finally:
        fake._hold.set()


def test_model_intercept_different_download_blocks_new_start() -> None:
    fake, _events, executed = _intercept_runtime(
        resolved=_uncached("qwen3-6-27b:q4_0"), active_repo="qwen3-5-9b:q8_0"
    )
    try:
        result = fake._maybe_intercept_model_download(
            session_id="s1", selector_arg="qwen3-6-27b:q4_0"
        )
        assert result is not None and result["ok"] is False
        assert "Another download (qwen3-5-9b:q8_0) is in progress" in str(result["message"])
        assert executed == []
    finally:
        fake._hold.set()


def test_model_intercept_falls_through_for_serving_cloud_list_and_unknown() -> None:
    # A cached selector the server is ALREADY serving answers instantly via the
    # sync path (fall through); anything else cached boots in the background.
    serving, _e1, _x1 = _intercept_runtime(
        resolved={"ok": True, "model": SimpleNamespace(cached=True, selector="qwen3-5-9b:q4_0")}
    )
    serving.lumen_runtime = SimpleNamespace(serving_selector=lambda: "qwen3-5-9b:q4_0")
    assert serving._maybe_intercept_model_download(session_id="s", selector_arg="qwen3-5-9b:q4_0") is None

    unknown, _e2, _x2 = _intercept_runtime(resolved=None)
    assert unknown._maybe_intercept_model_download(session_id="s", selector_arg="nope") is None

    passthrough, _e3, _x3 = _intercept_runtime(resolved=_uncached("x"))
    assert passthrough._maybe_intercept_model_download(session_id="s", selector_arg="") is None
    assert passthrough._maybe_intercept_model_download(session_id="s", selector_arg="list") is None
    assert (
        passthrough._maybe_intercept_model_download(session_id="s", selector_arg="openai:gpt-5.1")
        is None
    )


def _cached(selector: str) -> dict:
    return {"ok": True, "model": SimpleNamespace(cached=True, selector=selector)}


def test_cached_model_boots_in_background_with_ready_notice() -> None:
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    activated: list[str] = []
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: None,
        ensure_server=lambda selector: (True, f"Lumen serving {selector} on port 1."),
    )
    fake.model_service.activate_local = lambda selector: (
        activated.append(selector)
        or {"ok": True, "message": f"local · {selector} ready — now active."}
    )

    result = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")

    assert result is not None and result["ok"] is True and result["background"] is True
    # Terse selection confirmation only — the transcript's live "Loading …"
    # row solely owns load state (no duplicated live text in the panel).
    assert str(result["message"]) == "local · qwen3-5-9b:q4_0 selected."
    assert "Loading" not in str(result["message"])

    fake._active_boot_thread.join(timeout=10)
    assert activated == ["qwen3-5-9b:q4_0"]
    finals = [
        p for kind, p in events
        if kind == "message.updated" and p.get("final") is True
    ]
    assert any(
        "local · qwen3-5-9b:q4_0 ready — now active." in str(p.get("content", ""))
        for p in finals
    )
    frames = [p for kind, p in events if kind == "message.updated"]
    assert frames and frames[0]["progress"]["kind"] == "model-load"
    assert frames[0]["progress"]["phase"] == "loading"
    assert frames[-1]["final"] is True and frames[-1]["progress"]["phase"] == "ready"
    assert not any(kind == "session.status" for kind, _ in events)


def test_cached_model_boot_failure_reports_log_tail() -> None:
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: None,
        ensure_server=lambda selector: (False, "lumen-server failed to become ready: boom."),
    )
    fake.model_service.activate_local = lambda selector: {"ok": True}

    result = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert result is not None and result["ok"] is True

    fake._active_boot_thread.join(timeout=10)
    finals = [
        p for kind, p in events if kind == "message.updated" and p.get("final") is True
    ]
    assert any(
        "failed to start" in str(p.get("content", "")) and "boom" in str(p.get("content", ""))
        for p in finals
    )
    frames = [p for kind, p in events if kind == "message.updated" and p.get("final")]
    assert frames and frames[-1]["progress"]["phase"] == "failed"


def test_background_boot_switch_refusal_surfaces_retry_message_verbatim() -> None:
    """A background boot refused because ANOTHER model's boot is in flight
    must narrate the runtime's 'Model is switching …' retry message verbatim
    (same contract as the turn path) — never as a hard 'failed to start'."""
    refusal = f"{SWITCH_IN_FLIGHT_PREFIX}qwen3-6-27b:q4_0 — retry in a moment."
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: None,
        ensure_server=lambda selector: (False, refusal),
    )
    fake.model_service.activate_local = lambda selector: {"ok": True}

    result = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert result is not None and result["ok"] is True

    fake._active_boot_thread.join(timeout=10)
    finals = [
        p for kind, p in events if kind == "message.updated" and p.get("final") is True
    ]
    assert finals, "background boot must resolve its progress message"
    content = str(finals[-1].get("content", ""))
    assert content == refusal
    assert "failed to start" not in content
    assert finals[-1]["progress"]["phase"] == "failed"


def test_uncached_download_blocked_while_turn_driven_boot_in_flight() -> None:
    """The uncached-selector branch (download + chained auto-load) must also
    see turn-driven boots via the runtime's starting_selector — its auto-load
    would otherwise race the in-flight boot."""
    fake, _events, _executed = _intercept_runtime(resolved=_uncached("qwen3-6-27b:q4_0"))
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: "qwen3-5-9b:q8_0",  # turn-driven boot
    )
    result = fake._maybe_intercept_model_download(
        session_id="s1", selector_arg="qwen3-6-27b:q4_0"
    )
    assert result is not None and result["ok"] is False
    assert "Still starting local · qwen3-5-9b:q8_0" in str(result["message"])
    assert "before downloading" in str(result["message"])


def test_model_switch_blocked_while_turn_driven_boot_in_flight() -> None:
    """A turn can be the boot creator (restore/crash leaves the active model
    without a live server). /model must see that in-flight boot through the
    runtime's starting_selector — not only the worker's own boot thread — and
    block the switch instead of racing it."""
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: "qwen3-6-27b:q4_0",  # turn-driven boot
    )

    result = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert result is not None and result["ok"] is False
    assert "Still starting local · qwen3-6-27b:q4_0" in str(result["message"])

    # Same selector as the turn-driven boot: dedupe, don't double-start.
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: "qwen3-5-9b:q4_0",
    )
    again = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert again is not None and again["ok"] is True
    assert "Already starting local · qwen3-5-9b:q4_0" in str(again["message"])


def test_cached_boot_dedupes_and_blocks_switches() -> None:
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    hold = threading.Event()
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: None,
        ensure_server=lambda selector: (hold.wait(10), (True, "ok"))[1],
    )
    fake.model_service.activate_local = lambda selector: {"ok": True, "message": "done"}

    first = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert first is not None and first["background"] is True

    again = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert again is not None and again["ok"] is True
    assert "Already starting local · qwen3-5-9b:q4_0" in str(again["message"])

    fake.model_service.resolve_local_selector = lambda selector: _cached("qwen3-6-27b:q4_0")
    other = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-6-27b:q4_0")
    assert other is not None and other["ok"] is False
    assert "Still starting local · qwen3-5-9b:q4_0" in str(other["message"])

    hold.set()
    fake._active_boot_thread.join(timeout=10)


def test_stale_boot_keeps_model_inactive_after_new_choice() -> None:
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    hold = threading.Event()
    activated: list[str] = []
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: None,
        ensure_server=lambda selector: (hold.wait(10), (True, "ok"))[1],
    )
    fake.model_service.activate_local = lambda selector: activated.append(selector) or {"ok": True}

    result = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert result is not None and result["background"] is True

    fake._model_choice_seq += 1  # the user picked something newer mid-boot
    hold.set()
    fake._active_boot_thread.join(timeout=10)

    assert activated == []
    finals = [
        p for kind, p in events if kind == "message.updated" and p.get("final") is True
    ]
    assert any(
        "kept inactive — you switched models" in str(p.get("content", "")) for p in finals
    )


def _passthrough_resolver(fake, *, cached: bool = True) -> None:
    """resolve_local_selector echoing the requested selector (multi-model tests)."""
    fake.model_service.resolve_local_selector = lambda s: {
        "ok": True,
        "model": SimpleNamespace(cached=cached, selector=s),
    }


def _boot_in_flight(fake):
    """Start a held cached boot for qwen3-5-9b:q4_0; returns (hold, activated)."""
    hold = threading.Event()
    activated: list[str] = []
    fake.lumen_runtime = SimpleNamespace(
        serving_selector=lambda: None,
        starting_selector=lambda: None,
        ensure_server=lambda selector: (hold.wait(10), (True, "ok"))[1],
    )
    fake.model_service.activate_local = lambda selector: activated.append(selector) or {"ok": True}
    result = fake._maybe_intercept_model_download(session_id="s1", selector_arg="qwen3-5-9b:q4_0")
    assert result is not None and result["background"] is True
    return hold, activated


def test_blocked_switch_attempt_does_not_drop_activation() -> None:
    """A rejected mid-boot switch attempt is NOT a model choice: the in-flight
    boot's promised auto-activation must still happen (consensus-confirmed
    regression: every /model command used to bump the choice seq, silently
    resolving the op as 'kept inactive — you switched models')."""
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    _passthrough_resolver(fake, cached=True)
    hold, activated = _boot_in_flight(fake)

    blocked = fake._maybe_intercept_model_download(
        session_id="s1", selector_arg="qwen3-6-27b:q8_0"
    )
    assert blocked is not None and blocked["ok"] is False
    assert "wait for it to finish" in str(blocked["message"])

    hold.set()
    fake._active_boot_thread.join(timeout=10)
    assert activated == ["qwen3-5-9b:q4_0"], "blocked attempt must not cancel activation"
    finals = [p for kind, p in events if kind == "message.updated" and p.get("final") is True]
    assert any("ready — now active" in str(p.get("content", "")) for p in finals)
    assert not any("kept inactive" in str(p.get("content", "")) for p in finals)


def test_same_selector_reselect_does_not_drop_activation() -> None:
    """Re-selecting the very model that is loading keeps its promise
    ('it will be active shortly' must come true)."""
    fake, events, _executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    _passthrough_resolver(fake, cached=True)
    hold, activated = _boot_in_flight(fake)

    again = fake._maybe_intercept_model_download(
        session_id="s1", selector_arg="qwen3-5-9b:q4_0"
    )
    assert again is not None and again["ok"] is True
    assert "Already starting" in str(again["message"])

    hold.set()
    fake._active_boot_thread.join(timeout=10)
    assert activated == ["qwen3-5-9b:q4_0"]
    finals = [p for kind, p in events if kind == "message.updated" and p.get("final") is True]
    assert not any("kept inactive" in str(p.get("content", "")) for p in finals)


def test_uncached_select_blocked_while_boot_in_flight() -> None:
    """A download must never start while a boot runs — its chained auto-load
    would tear down the in-flight server (consensus-confirmed major)."""
    fake, events, executed = _intercept_runtime(resolved=_cached("qwen3-5-9b:q4_0"))
    _passthrough_resolver(fake, cached=True)
    hold, activated = _boot_in_flight(fake)

    fake.model_service.resolve_local_selector = lambda s: {
        "ok": True,
        "model": SimpleNamespace(cached=False, selector=s),
    }
    blocked = fake._maybe_intercept_model_download(
        session_id="s1", selector_arg="qwen3-5-9b:q8_0"
    )
    assert blocked is not None and blocked["ok"] is False
    assert "before downloading another model" in str(blocked["message"])
    assert executed == [], "no /download may run during a boot"

    hold.set()
    fake._active_boot_thread.join(timeout=10)
    assert activated == ["qwen3-5-9b:q4_0"]
