from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cortex.app.command_service import CommandService


class _FakeModelManager:
    def __init__(self, current_model: str = "demo-model") -> None:
        self.current_model = current_model
        self.tokenizers = {current_model: object()} if current_model else {}

    def discover_available_models(self):
        if not self.current_model:
            return []
        return [{"name": self.current_model, "path": f"/tmp/{self.current_model}", "format": "mlx", "size_gb": 1.2}]


class _FakeModelService:
    def __init__(self, current_model: str = "demo-model", *, backend: str = "local") -> None:
        self.model_manager = _FakeModelManager(current_model=current_model)
        self.active_target = SimpleNamespace(backend=backend)
        self.load_calls: list[str] = []

    def select_local_model(self, model_name_or_path: str):
        self.load_calls.append(model_name_or_path)
        return {"ok": True, "message": "loaded", "active_model": "demo-model"}

    def status_summary(self):
        return {"active_model": "demo-model"}

    def gpu_status(self):
        return {"chip_name": "Apple M4"}

    def list_models(self):
        return {"local": [], "cloud": []}

    def select_cloud_model(self, provider: str, model_id: str):
        return {"ok": True, "message": f"cloud {provider}:{model_id}"}

    def auth_status(self, _provider):
        return {"authenticated": False}

    def auth_save_key(self, _provider, _key):
        return {"ok": True}


class _FakeDownloader:
    def __init__(self, *, success: bool = True, message: str = "Downloaded", path: Path | None = None):
        self.success = success
        self.message = message
        self.path = path or Path("/tmp/model.gguf")
        self.calls: list[tuple[str, str | None]] = []

    def download_model(self, repo_id: str, filename: str | None):
        self.calls.append((repo_id, filename))
        return self.success, self.message, self.path if self.success else None


class _FakeTemplateConfigManager:
    def __init__(self, config=None):
        self._config = config

    def get_model_config(self, _model_name: str):
        return self._config


class _FakeTemplateRegistry:
    def __init__(self, *, config=None, reset_ok: bool = True):
        self.config_manager = _FakeTemplateConfigManager(config=config)
        self.reset_ok = reset_ok
        self.setup_calls: list[tuple[str, bool, bool]] = []

    def setup_model(self, model_name: str, *, tokenizer, interactive: bool, force_setup: bool):
        self.setup_calls.append((model_name, interactive, force_setup))
        return SimpleNamespace(config=SimpleNamespace(name="ChatML"))

    def reset_model_config(self, _model_name: str):
        return self.reset_ok

    def list_templates(self):
        return [{"type": "chatml", "name": "ChatML"}]


class _FakeInferenceEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def benchmark(self, prompt: str = "Once upon a time", num_tokens: int = 100):
        self.calls.append((prompt, num_tokens))
        return SimpleNamespace(
            tokens_generated=num_tokens,
            time_elapsed=2.0,
            tokens_per_second=num_tokens / 2.0,
            first_token_latency=0.2,
            gpu_utilization=61.0,
            memory_used_gb=3.2,
        )


def _service(*, model_service=None, downloader=None, template_registry=None, inference_engine=None) -> CommandService:
    return CommandService(
        model_service=model_service or _FakeModelService(),
        clear_session=lambda _session_id: {"ok": True, "message": "cleared"},
        save_session=lambda _session_id: {"ok": True, "path": "/tmp/x.json"},
        model_downloader=downloader or _FakeDownloader(),
        template_registry=template_registry or _FakeTemplateRegistry(),
        inference_engine=inference_engine or _FakeInferenceEngine(),
    )


def test_download_requires_args() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/download")
    assert result["ok"] is False
    assert "Usage: /download <repo_id>" in str(result["message"])


def test_download_rejects_invalid_repo_format() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/download badrepo")
    assert result["ok"] is False
    assert "Expected: username/model-name" in str(result["message"])


def test_download_rejects_extra_arguments() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/download user/repo file.gguf extra")
    assert result["ok"] is False
    assert "Too many arguments" in str(result["message"])


def test_download_success_calls_downloader() -> None:
    downloader = _FakeDownloader(success=True, message="ok", path=Path("/tmp/demo.gguf"))
    service = _service(downloader=downloader)
    result = service.execute(session_id="s1", command="/download user/repo model.gguf")
    assert result["ok"] is True
    assert downloader.calls == [("user/repo", "model.gguf")]
    assert result["download"]["path"] == "/tmp/demo.gguf"


def test_download_with_load_selects_local_model() -> None:
    model_service = _FakeModelService()
    downloader = _FakeDownloader(success=True, message="ok", path=Path("/tmp/loaded.gguf"))
    service = _service(model_service=model_service, downloader=downloader)
    result = service.execute(session_id="s1", command="/download user/repo --load")
    assert result["ok"] is True
    assert model_service.load_calls == ["/tmp/loaded.gguf"]
    assert isinstance(result.get("load"), dict)
    assert result["load"]["ok"] is True


def test_template_status_requires_loaded_model() -> None:
    service = _service(model_service=_FakeModelService(current_model=""))
    result = service.execute(session_id="s1", command="/template status")
    assert result["ok"] is False
    assert result["message"] == "No model loaded."


def test_template_status_returns_configuration() -> None:
    config = SimpleNamespace(
        detected_type="reasoning",
        user_preference="simple",
        custom_filters=["<think>"],
        show_reasoning=False,
        confidence=0.9,
        last_updated="2026-02-13T00:00:00",
    )
    service = _service(template_registry=_FakeTemplateRegistry(config=config))
    result = service.execute(session_id="s1", command="/template status")
    assert result["ok"] is True
    assert result["template"]["detected_type"] == "reasoning"
    assert result["template"]["custom_filters"] == ["<think>"]


def test_template_reset_reports_missing_config() -> None:
    service = _service(template_registry=_FakeTemplateRegistry(reset_ok=False))
    result = service.execute(session_id="s1", command="/template reset")
    assert result["ok"] is False
    assert "No template configuration found" in str(result["message"])


def test_template_auto_runs_non_interactive_setup() -> None:
    registry = _FakeTemplateRegistry()
    service = _service(template_registry=registry)
    result = service.execute(session_id="s1", command="/template")
    assert result["ok"] is True
    assert registry.setup_calls == [("demo-model", False, True)]
    assert result["template_name"] == "ChatML"


def test_benchmark_rejects_cloud_backend() -> None:
    service = _service(model_service=_FakeModelService(backend="cloud"))
    result = service.execute(session_id="s1", command="/benchmark")
    assert result["ok"] is False
    assert "local models only" in str(result["message"])


def test_benchmark_requires_loaded_model() -> None:
    service = _service(model_service=_FakeModelService(current_model=""))
    result = service.execute(session_id="s1", command="/benchmark")
    assert result["ok"] is False
    assert result["message"] == "No model loaded."


def test_benchmark_executes_and_returns_metrics_payload() -> None:
    engine = _FakeInferenceEngine()
    service = _service(inference_engine=engine)
    result = service.execute(session_id="s1", command="/benchmark --tokens 80 --prompt 'hello'")
    assert result["ok"] is True
    assert engine.calls == [("hello", 80)]
    assert result["benchmark"]["requested_tokens"] == 80
    assert result["benchmark"]["tokens_generated"] == 80
    assert result["benchmark"]["model"] == "demo-model"
    assert "Benchmark complete" in str(result["message"])


def test_benchmark_rejects_invalid_tokens() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/benchmark --tokens nope")
    assert result["ok"] is False
    assert "positive integer" in str(result["message"])


def test_status_command_includes_formatted_message() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/status")
    assert result["ok"] is True
    assert "System status" in str(result["message"])
    assert "- Active model:" in str(result["message"])


def test_gpu_command_includes_formatted_message() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/gpu")
    assert result["ok"] is True
    assert "GPU status" in str(result["message"])
    assert "- Chip name:" in str(result["message"])


def test_login_status_includes_formatted_message() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/login openai")
    assert result["ok"] is True
    assert isinstance(result.get("auth"), dict)
    assert "Authentication status" in str(result["message"])
    assert "- Provider: openai" in str(result["message"])


@pytest.mark.parametrize(
    "command",
    [
        "/help",
        "/status",
        "/gpu",
        "/models",
        "/model",
        "/login openai",
        "/finetune status",
        "/benchmark",
        "/clear",
        "/save",
    ],
)
def test_core_slash_commands_emit_transcript_ready_message(command: str) -> None:
    service = _service(model_service=_FakeModelService(current_model=""))
    result = service.execute(session_id="s1", command=command)
    assert bool(str(result.get("message", "")).strip())


def test_finetune_help_for_worker_mode() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/finetune")
    assert result["ok"] is True
    assert "Usage: /finetune status" in str(result["message"])


def test_finetune_status_payload_shape() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/finetune status")
    assert "finetune" in result
    assert isinstance(result["finetune"], dict)
    assert "mlx_available" in result["finetune"]
    assert result["finetune"]["available_local_models"] == 1
