"""Cortex must boot and serve cloud turns when Lumen is not installed.

The worker constructs with missing lumen binaries: local listings degrade to
empty, local selection returns the install hint, and nothing crashes.
"""

from __future__ import annotations

import io

from cortex.app.worker_runtime import WorkerRuntime
from cortex.config import Config
from cortex.conversation_manager import ConversationManager
from cortex.gpu_validator import GPUValidator


def _runtime(tmp_path, monkeypatch) -> WorkerRuntime:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = Config(config_path=tmp_path / "missing.yaml")
    config.lumen.lumen_binary = "definitely-not-lumen"
    config.lumen.lumen_server_binary = "definitely-not-lumen-server"
    return WorkerRuntime(
        config=config,
        gpu_validator=GPUValidator(),
        conversation_manager=ConversationManager(config),
        rpc_stdin=io.StringIO(),
        rpc_stdout=io.StringIO(),
    )


def test_worker_boots_without_lumen_binaries(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)

    models = runtime.model_service.list_models()
    assert models["local"] == []
    assert isinstance(models["cloud"], list)

    result = runtime.model_service.select_local_model("qwen3-5-9b:q4_0")
    assert result["ok"] is False
    assert "servelumen.com" in str(result["message"])


def test_download_without_lumen_reports_install_hint(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    result = runtime.command_service.execute(
        session_id="s1", command="/download qwen3-5-9b:q4_0"
    )
    assert result["ok"] is False
    assert "servelumen.com" in str(result["message"])
