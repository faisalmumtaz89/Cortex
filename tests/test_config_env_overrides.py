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

    assert config.tools.tools_max_iterations == 100  # defaults intact


def test_env_override_parses_booleans(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_TOOLS_ENABLED", "false")

    config = Config(config_path=tmp_path / "missing.yaml")

    assert config.tools.tools_enabled is False


def test_a_bad_setting_falls_back_alone_and_is_reported(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("tools_max_iterations: 1001\ntemperature: 0.3\ntools_profile: read_only\n")

    config = Config(config_path=path)

    assert config.tools.tools_max_iterations == 100
    assert config.inference.temperature == 0.3
    assert config.tools.tools_profile == "read_only"
    assert any("tools_max_iterations" in warning for warning in config.warnings)
    assert capsys.readouterr().out == ""


def test_a_bad_environment_value_falls_back_to_the_config_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("tools_max_iterations: 80\n")
    monkeypatch.setenv("CORTEX_TOOLS_MAX_ITERATIONS", "abc")

    config = Config(config_path=path)

    assert config.tools.tools_max_iterations == 80
    (warning,) = config.warnings
    assert warning.startswith("Ignored CORTEX_TOOLS_MAX_ITERATIONS: 'abc'")
    assert warning.endswith("using the value in the config file")


def test_retired_and_meaningless_settings_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("tools_idle_timeout_seconds: 300\ntools_enabled: maybe\n")

    config = Config(config_path=path)

    assert config.tools.tools_enabled is True
    assert any("set cloud_timeout_seconds" in warning for warning in config.warnings)
    assert any("Ignored tools_enabled: 'maybe'" in warning for warning in config.warnings)
