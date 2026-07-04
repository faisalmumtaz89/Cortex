from __future__ import annotations

from types import SimpleNamespace

from cortex.app.model_service import ModelService
from cortex.cloud.types import CloudProvider
from tests.lumen_fakes import FakeLumenRuntime, catalog


class _FakeConfig:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self.last_used: str = ""

    def set_state_value(self, key: str, value: str) -> None:
        self.state[key] = value

    def update_last_used_model(self, name: str) -> None:
        self.last_used = name


class _FakeCloudRouter:
    def __init__(self, *, openai_auth: bool = False, anthropic_auth: bool = False) -> None:
        self._auth = {
            CloudProvider.OPENAI: openai_auth,
            CloudProvider.ANTHROPIC: anthropic_auth,
        }

    def get_auth_status(self, provider: CloudProvider):
        return self._auth.get(provider, False), "fake"


def _service(
    *,
    openai_auth: bool = False,
    anthropic_auth: bool = False,
    lumen: FakeLumenRuntime | None = None,
) -> tuple[ModelService, _FakeConfig, FakeLumenRuntime]:
    config = _FakeConfig()
    runtime = lumen or FakeLumenRuntime()
    service = ModelService(
        config=config,
        lumen_runtime=runtime,
        gpu_validator=SimpleNamespace(),
        cloud_router=_FakeCloudRouter(openai_auth=openai_auth, anthropic_auth=anthropic_auth),
        credential_store=SimpleNamespace(),
        cloud_catalog=SimpleNamespace(list_models=lambda: []),
    )
    return service, config, runtime


def test_select_cloud_model_rejects_when_provider_not_authenticated() -> None:
    service, config, _ = _service(anthropic_auth=False)

    result = service.select_cloud_model(provider="anthropic", model_id="claude-haiku-4-5")

    assert result["ok"] is False
    assert "anthropic is not authenticated" in str(result["message"])
    assert service.active_target.backend == "local"
    assert config.state == {}


def test_select_cloud_model_sets_active_target_and_state_when_authenticated() -> None:
    service, config, _ = _service(openai_auth=True)

    result = service.select_cloud_model(provider="openai", model_id="gpt-5.1")

    assert result["ok"] is True
    assert result["message"] == "cloud · openai:gpt-5.1 — now active."
    assert service.active_target.backend == "cloud"
    assert service.active_target.cloud_model is not None
    assert service.active_target.cloud_model.selector == "openai:gpt-5.1"
    assert config.state == {
        "last_used_backend": "cloud",
        "last_used_cloud_provider": "openai",
        "last_used_cloud_model": "gpt-5.1",
    }


def test_select_local_model_boots_lumen_and_persists_state() -> None:
    service, config, runtime = _service()

    result = service.select_local_model("qwen3-5-9b:q4_0")

    assert result["ok"] is True, result
    assert runtime.ensure_calls == ["qwen3-5-9b:q4_0"]
    assert service.active_target.backend == "local"
    assert service.active_target.local_model == "qwen3-5-9b:q4_0"
    assert config.state["last_used_backend"] == "local"
    assert config.last_used == "qwen3-5-9b:q4_0"


def test_select_local_bare_name_prefers_cached_quant() -> None:
    service, _, runtime = _service()

    result = service.select_local_model("qwen3-5-9b")

    # Q4_0 is the cached quant in the default fake catalog; Q8_0 is not.
    assert result["ok"] is True, result
    assert runtime.ensure_calls == ["qwen3-5-9b:q4_0"]


def test_select_local_uncached_model_returns_structured_not_cached_marker() -> None:
    service, _, runtime = _service()

    result = service.select_local_model("qwen3-6-27b:q4_0")

    # The worker intercepts this in the TUI (auto-download); the sync fallback
    # is a structured marker whose message never dead-ends into /download.
    assert result["ok"] is False
    assert result["not_cached"] is True
    assert result["selector"] == "qwen3-6-27b:q4_0"
    assert "/download" not in str(result["message"])
    assert "download" in str(result["message"]).lower()
    assert runtime.ensure_calls == []


def test_select_local_unknown_model_lists_supported_names() -> None:
    service, _, _ = _service()

    result = service.select_local_model("llama-3")

    assert result["ok"] is False
    assert "qwen3-5-9b" in str(result["message"])


def test_list_models_marks_cached_loaded_and_active() -> None:
    service, _, runtime = _service()
    assert service.select_local_model("qwen3-5-9b:q4_0")["ok"] is True

    listing = service.list_models()
    local = {item["name"]: item for item in listing["local"]}

    assert local["qwen3-5-9b:q4_0"]["cached"] is True
    assert local["qwen3-5-9b:q4_0"]["loaded"] is True
    assert local["qwen3-5-9b:q4_0"]["active"] is True
    assert local["qwen3-5-9b:q8_0"]["cached"] is False
    assert local["qwen3-5-9b:q8_0"]["active"] is False
    assert listing["active_target"]["label"] == "qwen3-5-9b:q4_0"


def test_restore_local_target_is_lazy() -> None:
    service, _, runtime = _service()

    restored = service.restore_local_target("qwen3-5-9b:q4_0")

    assert restored == "qwen3-5-9b:q4_0"
    assert service.active_target.local_model == "qwen3-5-9b:q4_0"
    assert runtime.ensure_calls == []  # server must NOT boot at startup


def test_restore_local_target_rejects_uncached() -> None:
    service, _, _ = _service(lumen=FakeLumenRuntime(models=catalog(("qwen3-5-9b", "Q4_0", False))))

    assert service.restore_local_target("qwen3-5-9b:q4_0") is None
    assert service.active_target.local_model is None


def test_status_summary_reports_lumen_server() -> None:
    service, _, _ = _service()
    service.select_local_model("qwen3-5-9b:q4_0")

    summary = service.status_summary()

    assert summary["backend"] == "local"
    assert "qwen3-5-9b:q4_0" in str(summary["lumen_server"])


def test_list_models_and_status_reflect_a_booting_server() -> None:
    service, _config, runtime = _service()
    runtime._starting = "qwen3-5-9b:q4_0"  # boot in flight, not ready yet

    listing = service.list_models()
    entry = next(m for m in listing["local"] if m["name"] == "qwen3-5-9b:q4_0")
    assert entry["loading"] is True
    assert entry["loaded"] is False

    summary = service.status_summary()
    assert "starting qwen3-5-9b:q4_0" in str(summary["lumen_server"])

    gpu = service.gpu_status()
    assert gpu["lumen_server"] == "starting"


def test_reselect_active_serving_model_reports_already_active() -> None:
    service, _config, _runtime = _service()
    first = service.select_local_model("qwen3-5-9b:q4_0")
    assert first["ok"] is True
    again = service.select_local_model("qwen3-5-9b:q4_0")
    assert again["ok"] is True
    assert "already active" in str(again["message"])
