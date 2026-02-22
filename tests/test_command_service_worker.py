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
        return [
            {
                "name": self.current_model,
                "path": f"/tmp/{self.current_model}",
                "format": "mlx",
                "size_gb": 1.2,
            }
        ]


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
    def __init__(
        self, *, success: bool = True, message: str = "Downloaded", path: Path | None = None
    ):
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


def _service(
    *, model_service=None, downloader=None, template_registry=None, inference_engine=None
) -> CommandService:
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


def test_download_rejects_unknown_option() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/download user/repo --force")
    assert result["ok"] is False
    assert "Unknown option: --force" in str(result["message"])


def test_download_preflight_rejects_missing_repo_without_starting_download() -> None:
    service = _service()
    preflight = service.preflight_download("")
    assert preflight["ok"] is False
    assert "Usage: /download <repo_id>" in str(preflight["message"])


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


def test_download_accepts_quoted_filename() -> None:
    downloader = _FakeDownloader(success=True, message="ok", path=Path("/tmp/quoted.gguf"))
    service = _service(downloader=downloader)
    result = service.execute(session_id="s1", command='/download user/repo "model file.gguf"')
    assert result["ok"] is True
    assert downloader.calls == [("user/repo", "model file.gguf")]


def test_download_existing_target_without_load_is_not_ok_and_skips_network_download() -> None:
    class _ExistingTargetDownloader:
        def __init__(self) -> None:
            self.download_calls = 0
            self.inspect_calls: list[tuple[str, str | None]] = []

        def inspect_download_target(self, repo_id: str, filename: str | None = None):
            self.inspect_calls.append((repo_id, filename))
            return {
                "path": Path("/tmp/existing-repo"),
                "exists": True,
                "resumable": False,
                "kind": "repo",
            }

        def download_model(self, repo_id: str, filename: str | None):
            self.download_calls += 1
            return True, "unexpected", Path("/tmp/should-not-run")

    model_service = _FakeModelService()
    downloader = _ExistingTargetDownloader()
    service = _service(model_service=model_service, downloader=downloader)

    result = service.execute(session_id="s1", command="/download user/repo")
    assert result["ok"] is False
    assert "Model already exists: /tmp/existing-repo" in str(result["message"])
    assert result["download"]["preexisting"] is True
    assert model_service.load_calls == []
    assert downloader.download_calls == 0
    assert downloader.inspect_calls == [("user/repo", None)]


def test_download_existing_target_with_load_attempts_load_without_network_download() -> None:
    class _ExistingTargetDownloader:
        def __init__(self) -> None:
            self.download_calls = 0
            self.inspect_calls: list[tuple[str, str | None]] = []

        def inspect_download_target(self, repo_id: str, filename: str | None = None):
            self.inspect_calls.append((repo_id, filename))
            return {
                "path": Path("/tmp/existing-repo"),
                "exists": True,
                "resumable": False,
                "kind": "repo",
            }

        def download_model(self, repo_id: str, filename: str | None):
            self.download_calls += 1
            return True, "unexpected", Path("/tmp/should-not-run")

    model_service = _FakeModelService()
    downloader = _ExistingTargetDownloader()
    service = _service(model_service=model_service, downloader=downloader)

    result = service.execute(session_id="s1", command="/download user/repo --load")
    assert result["ok"] is True
    assert "Model already exists: /tmp/existing-repo (loaded)" in str(result["message"])
    assert result["download"]["preexisting"] is True
    assert model_service.load_calls == ["/tmp/existing-repo"]
    assert downloader.download_calls == 0
    assert downloader.inspect_calls == [("user/repo", None)]


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


def test_model_rejects_unknown_cloud_provider_gracefully() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/model unknown:gpt")
    assert result["ok"] is False
    assert "Unsupported cloud provider" in str(result["message"])


def test_model_rejects_empty_cloud_selector() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/model openai:")
    assert result["ok"] is False
    assert "Usage: /model <provider:model>" in str(result["message"])


def test_login_rejects_unknown_provider_gracefully() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/login unknown")
    assert result["ok"] is False
    assert "Unsupported provider" in str(result["message"])


def test_login_huggingface_with_token_is_rejected_for_security() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/login huggingface hf_secret_token")
    assert result["ok"] is False
    assert "Do not paste HuggingFace tokens into chat" in str(result["message"])


def test_setup_loads_first_available_local_model() -> None:
    model_service = _FakeModelService(current_model="")
    model_service.list_models = lambda: {
        "active_target": {"label": "No model loaded"},
        "local": [{"name": "first-local-model"}, {"name": "second-local-model"}],
        "cloud": [],
    }
    service = _service(model_service=model_service)
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is True
    assert model_service.load_calls == ["first-local-model"]


def test_setup_reports_complete_when_model_already_active() -> None:
    model_service = _FakeModelService(current_model="demo-model")
    model_service.list_models = lambda: {
        "active_target": {"label": "demo-model"},
        "local": [{"name": "demo-model", "active": True, "loaded": True}],
        "cloud": [],
    }
    service = _service(model_service=model_service)
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is True
    assert "Setup complete. Active model: demo-model" in str(result["message"])


def test_setup_requires_any_installed_local_model() -> None:
    model_service = _FakeModelService(current_model="")
    model_service.list_models = lambda: {
        "active_target": {"label": "No model loaded"},
        "local": [],
        "cloud": [],
    }
    service = _service(model_service=model_service)
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is False
    assert "No local model installed." in str(result["message"])


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
        "/setup",
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


def test_execute_rejects_empty_command() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="   ")
    assert result["ok"] is False
    assert result["message"] == "Command cannot be empty."


def test_execute_rejects_non_slash_command() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="download user/repo")
    assert result["ok"] is False
    assert result["message"] == "Not a slash command: download user/repo"


@pytest.mark.parametrize("command", ["/quit", "/exit"])
def test_execute_quit_commands_return_exit(command: str) -> None:
    service = _service()
    result = service.execute(session_id="s1", command=command)
    assert result["ok"] is True
    assert result["exit"] is True


def test_execute_save_sets_default_message_from_path() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/save")
    assert result["ok"] is True
    assert result["message"] == "Saved conversation: /tmp/x.json"


def test_execute_unknown_command_returns_error() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/doesnotexist")
    assert result["ok"] is False
    assert result["message"] == "Unknown command: /doesnotexist"


def test_download_rejects_invalid_shell_syntax() -> None:
    service = _service()
    result = service.execute(session_id="s1", command='/download "unterminated')
    assert result["ok"] is False
    assert "Invalid download arguments:" in str(result["message"])


def test_download_rejects_flag_only_without_repo() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/download --load")
    assert result["ok"] is False
    assert "Usage: /download <repo_id>" in str(result["message"])


def test_download_with_load_and_missing_path_does_not_attempt_load() -> None:
    class _NoPathDownloader:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def download_model(self, repo_id: str, filename: str | None):
            self.calls.append((repo_id, filename))
            return True, "ok", None

    model_service = _FakeModelService()
    downloader = _NoPathDownloader()
    service = _service(model_service=model_service, downloader=downloader)
    result = service.execute(session_id="s1", command="/download user/repo --load")
    assert result["ok"] is True
    assert model_service.load_calls == []
    assert "load" not in result


def test_download_with_load_surfaces_model_load_failure() -> None:
    model_service = _FakeModelService()
    model_service.select_local_model = lambda _name: {"ok": False, "message": "cannot load model"}  # type: ignore[method-assign]
    downloader = _FakeDownloader(success=True, message="ok", path=Path("/tmp/failed-load.gguf"))
    service = _service(model_service=model_service, downloader=downloader)
    result = service.execute(session_id="s1", command="/download user/repo --load")
    assert result["ok"] is False
    assert "failed to load: cannot load model" in str(result["message"])


def test_download_with_load_failure_message_is_summarized_for_transcript() -> None:
    model_service = _FakeModelService()
    model_service.select_local_model = lambda _name: {  # type: ignore[method-assign]
        "ok": False,
        "message": "Failed to load model\n" + ("details " * 200),
    }
    downloader = _FakeDownloader(success=True, message="ok", path=Path("/tmp/failed-load.gguf"))
    service = _service(model_service=model_service, downloader=downloader)

    result = service.execute(session_id="s1", command="/download user/repo --load")
    message = str(result["message"])
    assert result["ok"] is False
    assert "failed to load: Failed to load model" in message
    assert len(message) < 320


def test_download_with_load_failure_message_deduplicates_repeated_prefix() -> None:
    model_service = _FakeModelService()
    model_service.select_local_model = lambda _name: {  # type: ignore[method-assign]
        "ok": False,
        "message": "Failed to load MLX model: Failed to load MLX model: Received 64 parameters not in model:",
    }
    downloader = _FakeDownloader(success=True, message="ok", path=Path("/tmp/failed-load.gguf"))
    service = _service(model_service=model_service, downloader=downloader)

    result = service.execute(session_id="s1", command="/download user/repo --load")
    message = str(result["message"])
    assert result["ok"] is False
    assert (
        "failed to load: Failed to load MLX model: Received 64 parameters not in model:" in message
    )
    assert "Failed to load MLX model: Failed to load MLX model" not in message


def test_template_list_returns_registry_templates() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/template list")
    assert result["ok"] is True
    assert isinstance(result.get("templates"), list)
    assert result["templates"][0]["name"] == "ChatML"


def test_template_configure_alias_runs_setup() -> None:
    registry = _FakeTemplateRegistry()
    service = _service(template_registry=registry)
    result = service.execute(session_id="s1", command="/template configure")
    assert result["ok"] is True
    assert registry.setup_calls == [("demo-model", False, True)]


def test_template_rejects_unknown_subcommand() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/template nonsense")
    assert result["ok"] is False
    assert result["message"] == "Usage: /template [status|reset|list|auto]"


def test_benchmark_rejects_invalid_shell_syntax() -> None:
    service = _service()
    result = service.execute(session_id="s1", command='/benchmark --prompt "unterminated')
    assert result["ok"] is False
    assert "Invalid benchmark arguments:" in str(result["message"])


def test_benchmark_rejects_missing_prompt_value() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/benchmark --prompt")
    assert result["ok"] is False
    assert result["message"] == "Usage: /benchmark [tokens] [--prompt <text>]"


def test_benchmark_rejects_out_of_range_tokens() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/benchmark --tokens 9000")
    assert result["ok"] is False
    assert "between 1 and 8192" in str(result["message"])


def test_benchmark_accepts_positional_token_count() -> None:
    engine = _FakeInferenceEngine()
    service = _service(inference_engine=engine)
    result = service.execute(session_id="s1", command="/benchmark 32")
    assert result["ok"] is True
    assert engine.calls == [("Once upon a time", 32)]


def test_benchmark_rejects_empty_prompt() -> None:
    service = _service()
    result = service.execute(session_id="s1", command='/benchmark --prompt ""')
    assert result["ok"] is False
    assert result["message"] == "Benchmark prompt cannot be empty."


def test_benchmark_rejects_when_engine_not_available() -> None:
    service = _service(inference_engine=object())
    result = service.execute(session_id="s1", command="/benchmark")
    assert result["ok"] is False
    assert result["message"] == "Benchmark engine is not available in worker mode."


def test_benchmark_rejects_when_engine_returns_none() -> None:
    class _NoMetricsEngine:
        def benchmark(self, prompt: str = "Once upon a time", num_tokens: int = 100):
            return None

    service = _service(inference_engine=_NoMetricsEngine())
    result = service.execute(session_id="s1", command="/benchmark")
    assert result["ok"] is False
    assert result["message"] == "Benchmark failed to produce metrics."


def test_finetune_rejects_invalid_shell_syntax() -> None:
    service = _service()
    result = service.execute(session_id="s1", command='/finetune "unterminated')
    assert result["ok"] is False
    assert "Invalid finetune arguments:" in str(result["message"])


@pytest.mark.parametrize("command", ["/finetune --help", "/finetune -h", "/finetune help"])
def test_finetune_help_variants_are_supported(command: str) -> None:
    service = _service()
    result = service.execute(session_id="s1", command=command)
    assert result["ok"] is True
    assert "Usage: /finetune status" in str(result["message"])


def test_finetune_unknown_subcommand_returns_actionable_error() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/finetune train")
    assert result["ok"] is False
    assert "Run `/finetune status`" in str(result["message"])


def test_model_list_and_ls_aliases_route_to_model_list() -> None:
    service = _service()
    list_result = service.execute(session_id="s1", command="/model list")
    ls_result = service.execute(session_id="s1", command="/model ls")
    assert list_result["ok"] is True
    assert ls_result["ok"] is True
    assert "Active model:" in str(list_result["message"])
    assert "Active model:" in str(ls_result["message"])


def test_model_select_local_routes_to_model_service() -> None:
    model_service = _FakeModelService()
    service = _service(model_service=model_service)
    result = service.execute(session_id="s1", command="/model /tmp/foo.gguf")
    assert result["ok"] is True
    assert model_service.load_calls == ["/tmp/foo.gguf"]


def test_model_select_local_by_index_routes_to_model_service() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [
            {"name": "mlx-community--first-model"},
            {"name": "mlx-community--second-model"},
        ],
        "cloud": [],
    }
    service = _service(model_service=model_service)

    result = service.execute(session_id="s1", command="/model 2")
    assert result["ok"] is True
    assert model_service.load_calls == ["mlx-community--second-model"]


def test_model_list_and_index_selection_use_stable_sorted_order() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [
            {"name": "zeta-model"},
            {"name": "Alpha-model"},
            {"name": "beta-model"},
        ],
        "cloud": [
            {"selector": "openai:gpt-5.1", "authenticated": False, "active": False},
            {"selector": "anthropic:claude-sonnet-4-5", "authenticated": False, "active": False},
        ],
    }
    service = _service(model_service=model_service)

    listed = service.execute(session_id="s1", command="/model")
    assert listed["ok"] is True
    message = str(listed["message"])
    assert "- [1] Alpha-model" in message
    assert "- [2] beta-model" in message
    assert "- [3] zeta-model" in message
    assert "- [4] anthropic:claude-sonnet-4-5 (login required)" in message
    assert "- [5] openai:gpt-5.1 (login required)" in message
    assert (
        "Use /model <number> to select from this list, or /model <local-name|provider:model>."
        in message
    )
    assert "<partial-local-name>" not in message

    selected = service.execute(session_id="s1", command="/model 1")
    assert selected["ok"] is True
    assert model_service.load_calls == ["Alpha-model"]


def test_model_select_cloud_by_index_routes_to_cloud_selection() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [
            {"name": "zeta-model"},
        ],
        "cloud": [
            {
                "provider": "openai",
                "model_id": "gpt-5.1",
                "selector": "openai:gpt-5.1",
                "authenticated": True,
                "active": False,
            },
            {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-5",
                "selector": "anthropic:claude-sonnet-4-5",
                "authenticated": False,
                "active": False,
            },
        ],
    }
    service = _service(model_service=model_service)

    listed = service.execute(session_id="s1", command="/model")
    assert listed["ok"] is True
    message = str(listed["message"])
    assert "- [1] zeta-model" in message
    assert "- [2] anthropic:claude-sonnet-4-5 (login required)" in message
    assert "- [3] openai:gpt-5.1 (ready)" in message

    selected = service.execute(session_id="s1", command="/model 3")
    assert selected["ok"] is True
    assert selected["message"] == "cloud openai:gpt-5.1"


def test_model_select_cloud_by_hash_index_routes_to_cloud_selection() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [],
        "cloud": [
            {
                "provider": "openai",
                "model_id": "gpt-5-nano",
                "selector": "openai:gpt-5-nano",
                "authenticated": True,
                "active": False,
            },
        ],
    }
    service = _service(model_service=model_service)

    selected = service.execute(session_id="s1", command="/model #1")
    assert selected["ok"] is True
    assert selected["message"] == "cloud openai:gpt-5-nano"


def test_model_select_local_by_unique_partial_name_routes_to_model_service() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [
            {"name": "mlx-community--llama-3.2-3b-instruct-4bit"},
            {"name": "mlx-community--mistral-7b-instruct-v0.3"},
        ],
        "cloud": [],
    }
    service = _service(model_service=model_service)

    result = service.execute(session_id="s1", command="/model mistral-7b")
    assert result["ok"] is True
    assert model_service.load_calls == ["mlx-community--mistral-7b-instruct-v0.3"]


def test_model_select_local_with_ambiguous_partial_name_returns_actionable_error() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [
            {"name": "mlx-community--llama-3.2-3b-instruct-4bit"},
            {"name": "mlx-community--llama-3.2-1b-instruct-4bit"},
        ],
        "cloud": [],
    }
    service = _service(model_service=model_service)

    result = service.execute(session_id="s1", command="/model llama-3.2")
    assert result["ok"] is False
    assert "Ambiguous local model selector" in str(result["message"])
    assert "Use /model <number>" in str(result["message"])
    assert model_service.load_calls == []


def test_model_select_local_with_out_of_range_index_returns_actionable_error() -> None:
    model_service = _FakeModelService()
    model_service.list_models = lambda: {
        "local": [{"name": "mlx-community--only-model"}],
        "cloud": [],
    }
    service = _service(model_service=model_service)

    result = service.execute(session_id="s1", command="/model 9")
    assert result["ok"] is False
    assert "Model index out of range" in str(result["message"])
    assert model_service.load_calls == []


def test_model_select_cloud_routes_to_model_service() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/model openai:gpt-5.1")
    assert result["ok"] is True
    assert result["message"] == "cloud openai:gpt-5.1"


def test_model_rejects_missing_provider_or_model_parts() -> None:
    service = _service()
    result_missing_provider = service.execute(session_id="s1", command="/model :gpt-5.1")
    result_missing_model = service.execute(session_id="s1", command="/model openai:")
    assert result_missing_provider["ok"] is False
    assert "Usage: /model <provider:model>" in str(result_missing_provider["message"])
    assert result_missing_model["ok"] is False
    assert "Usage: /model <provider:model>" in str(result_missing_model["message"])


def test_login_requires_provider_argument() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/login")
    assert result["ok"] is False
    assert "Usage: /login openai|anthropic" in str(result["message"])


def test_login_provider_key_save_sets_default_success_message() -> None:
    service = _service()
    result = service.execute(session_id="s1", command="/login openai sk-test")
    assert result["ok"] is True
    assert result["message"] == "Saved openai API key."


def test_login_huggingface_alias_uses_status_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    monkeypatch.setattr(
        CommandService,
        "_huggingface_status",
        staticmethod(lambda: {"ok": True, "message": "hf auth ok"}),
    )
    result = service.execute(session_id="s1", command="/login hf")
    assert result["ok"] is True
    assert result["message"] == "hf auth ok"


def test_setup_returns_loaded_message_when_model_service_omits_message() -> None:
    model_service = _FakeModelService(current_model="")
    model_service.list_models = lambda: {
        "active_target": {"label": "No model loaded"},
        "local": [{"name": "first-local-model"}],
        "cloud": [],
    }
    model_service.select_local_model = lambda _name: {"ok": True}  # type: ignore[method-assign]
    service = _service(model_service=model_service)
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is True
    assert result["message"] == "Loaded local model: first-local-model"


def test_setup_returns_failed_message_when_model_service_omits_message() -> None:
    model_service = _FakeModelService(current_model="")
    model_service.list_models = lambda: {
        "active_target": {"label": "No model loaded"},
        "local": [{"name": "first-local-model"}],
        "cloud": [],
    }
    model_service.select_local_model = lambda _name: {"ok": False}  # type: ignore[method-assign]
    service = _service(model_service=model_service)
    result = service.execute(session_id="s1", command="/setup")
    assert result["ok"] is False
    assert result["message"] == "Failed to load local model: first-local-model"
