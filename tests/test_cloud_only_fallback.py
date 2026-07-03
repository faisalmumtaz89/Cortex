"""Cloud-only fallback behavior when MLX is unavailable (e.g. Linux containers).

True absence of MLX is validated in a Linux container (see AGENTS.md); these
tests cover the availability-flag branches directly.
"""

from __future__ import annotations

import pytest

import cortex.inference_engine as inference_engine
import cortex.model_manager as model_manager
from cortex.config import Config
from cortex.gpu_validator import GPUValidator


def test_generation_request_importable_without_local_backend() -> None:
    assert hasattr(inference_engine, "GenerationRequest")


def test_model_manager_constructs_and_refuses_load_without_mlx(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(model_manager, "MLX_AVAILABLE", False)

    config = Config(config_path=tmp_path / "missing.yaml")
    config.model.model_path = tmp_path / "models"
    config.model.model_cache_dir = tmp_path / "cache"

    manager = model_manager.ModelManager(config, GPUValidator())

    assert manager.mlx_converter is None
    assert manager.mlx_accelerator is None

    ok, message = manager.load_model("mlx-community/some-model")
    assert ok is False
    assert "requires MLX" in message
    assert manager.discover_available_models() == []


def test_inference_engine_raises_cleanly_without_mlx(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_engine, "MLX_AVAILABLE", False)
    monkeypatch.setattr(model_manager, "MLX_AVAILABLE", False)

    config = Config(config_path=tmp_path / "missing.yaml")
    config.model.model_path = tmp_path / "models"
    config.model.model_cache_dir = tmp_path / "cache"
    manager = model_manager.ModelManager(config, GPUValidator())

    with pytest.raises(RuntimeError, match="requires MLX"):
        inference_engine.InferenceEngine(config, manager)
