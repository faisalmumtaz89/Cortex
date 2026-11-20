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
    assert any("cloud_timeout_seconds sets the request" in warning for warning in config.warnings)
    assert any("Ignored tools_enabled: 'maybe'" in warning for warning in config.warnings)


def test_a_setting_that_does_nothing_is_named_so_it_can_be_deleted(tmp_path: Path) -> None:
    """Settings earlier versions accepted without acting on them are gone;
    a config file that still sets one keeps working and says which."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "ui_theme: dark\nsave_format: json\nmax_conversation_history: 5\nmax_tokens: 1000\n")

    config = Config(config_path=path)

    assert config.inference.max_tokens == 1000
    assert sorted(config.warnings) == [
        f"Ignored max_conversation_history in {path}: it has no effect in Cortex, and can be deleted",
        f"Ignored save_format in {path}: it has no effect in Cortex, and can be deleted",
        f"Ignored ui_theme in {path}: it has no effect in Cortex, and can be deleted",
    ]
    assert not hasattr(config, "ui") and not hasattr(config, "paths")


def test_reasoning_effort_is_a_level_or_unset(tmp_path: Path, monkeypatch) -> None:
    """Unset by default, so nothing is asked of the model; a level from the
    file or its variable, whichever word the provider has for it; what is
    not one lower-case word is named and not used."""
    monkeypatch.delenv("CORTEX_REASONING_EFFORT", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("max_tokens: 1000\n")
    assert Config(config_path=path).inference.reasoning_effort is None

    path.write_text("reasoning_effort: low\n")
    assert Config(config_path=path).inference.reasoning_effort == "low"

    monkeypatch.setenv("CORTEX_REASONING_EFFORT", "high")
    config = Config(config_path=path)
    assert config.inference.reasoning_effort == "high"
    assert ("reasoning_effort", "high", "CORTEX_REASONING_EFFORT") in config.settings_in_effect()

    # A provider's own word for a level, which YAML reads as a switch: the
    # variable takes it as written, the file needs it in quotes.
    monkeypatch.setenv("CORTEX_REASONING_EFFORT", "off")
    assert Config(config_path=path).inference.reasoning_effort == "off"
    monkeypatch.delenv("CORTEX_REASONING_EFFORT")
    path.write_text('reasoning_effort: "off"\n')
    assert Config(config_path=path).inference.reasoning_effort == "off"
    path.write_text("reasoning_effort: off\n")
    config = Config(config_path=path)
    assert config.inference.reasoning_effort is None
    assert [line for line in config.warnings if "reasoning_effort" in line] == [
        "Ignored reasoning_effort: False (Value error, YAML reads this word as a switch: "
        'write it in quotes, as in "off"); using the default']

    path.write_text("reasoning_effort: low\n")
    for written in ("Very High", "5", "x-high", "[high]", "2024-01-01"):
        monkeypatch.setenv("CORTEX_REASONING_EFFORT", written)
        config = Config(config_path=path)
        assert config.inference.reasoning_effort == "low"  # the file's value stands
        assert [line for line in config.warnings if "REASONING_EFFORT" in line] == [
            f"Ignored CORTEX_REASONING_EFFORT: {written!r} (Value error, a level is one lower-case "
            "word, such as high); using the value in the config file"]


def test_a_text_setting_takes_its_variable_as_written(tmp_path: Path, monkeypatch) -> None:
    """A model's name that YAML would read as a number is the name."""
    path = tmp_path / "config.yaml"
    path.write_text("cloud_default_openai_model: from-file\n")
    monkeypatch.setenv("CORTEX_CLOUD_DEFAULT_OPENAI_MODEL", "4.10")
    config = Config(config_path=path)
    assert config.cloud.cloud_default_openai_model == "4.10"
    assert not [line for line in config.warnings if "OPENAI_MODEL" in line]


def test_a_variable_set_to_nothing_unsets_the_files_level_and_status_says_so(
    tmp_path: Path, monkeypatch
) -> None:
    from cortex.app.command_service import _setting_text

    path = tmp_path / "config.yaml"
    path.write_text("reasoning_effort: low\n")
    monkeypatch.setenv("CORTEX_REASONING_EFFORT", "")
    config = Config(config_path=path)
    assert config.inference.reasoning_effort is None
    (setting,) = [found for found in config.settings_in_effect() if found[0] == "reasoning_effort"]
    assert (_setting_text(setting[0], setting[1]), setting[2]) == ("unset", "CORTEX_REASONING_EFFORT")
