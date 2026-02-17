from pathlib import Path

from cortex.config import Config


def test_cloud_config_keys_load_from_yaml(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
cloud_enabled: true
cloud_timeout_seconds: 45
cloud_max_retries: 3
cloud_default_openai_model: gpt-5-mini
cloud_default_anthropic_model: claude-haiku-4-5
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(Config, "STATE_FILE", tmp_path / "state.yaml")
    config = Config(config_path=config_file)

    assert config.cloud.cloud_enabled is True
    assert config.cloud.cloud_timeout_seconds == 45
    assert config.cloud.cloud_max_retries == 3
    assert config.cloud.cloud_default_openai_model == "gpt-5-mini"
    assert config.cloud.cloud_default_anthropic_model == "claude-haiku-4-5"


def test_tools_profile_boolean_false_normalizes_to_off(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
cloud_enabled: true
tools_enabled: true
tools_profile: false
tools_local_mode: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(Config, "STATE_FILE", tmp_path / "state.yaml")
    config = Config(config_path=config_file)

    assert config.tools.tools_enabled is True
    assert config.tools.tools_profile == "off"
    assert config.tools.tools_local_mode == "disabled"
