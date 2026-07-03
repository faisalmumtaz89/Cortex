"""Env-var config overrides (CORTEX_<KEY>)."""

from __future__ import annotations

from pathlib import Path

from cortex.config import Config


def test_env_override_applies_to_known_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_TOOLS_MAX_ITERATIONS", "80")
    monkeypatch.setenv("CORTEX_CLOUD_TIMEOUT_SECONDS", "600")

    config = Config(config_path=tmp_path / "missing.yaml")

    assert config.tools.tools_max_iterations == 80
    assert config.cloud.cloud_timeout_seconds == 600


def test_env_override_beats_config_file(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("tools_max_iterations: 5\n", encoding="utf-8")
    monkeypatch.setenv("CORTEX_TOOLS_MAX_ITERATIONS", "42")

    config = Config(config_path=config_file)

    assert config.tools.tools_max_iterations == 42


def test_unknown_cortex_env_vars_are_ignored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_WORKER_MODE", "1")
    monkeypatch.setenv("CORTEX_SCRIPTED_MODEL", "/tmp/x.json")

    config = Config(config_path=tmp_path / "missing.yaml")

    assert config.tools.tools_max_iterations == 25  # defaults intact


def test_env_override_parses_booleans(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_TOOLS_ENABLED", "false")

    config = Config(config_path=tmp_path / "missing.yaml")

    assert config.tools.tools_enabled is False
