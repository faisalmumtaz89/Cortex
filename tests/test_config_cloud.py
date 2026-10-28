from pathlib import Path

from cortex.config import Config


def test_cloud_config_keys_load_from_yaml(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
cloud_timeout_seconds: 45
cloud_max_retries: 3
cloud_default_openai_model: gpt-5-mini
cloud_default_anthropic_model: claude-haiku-4-5
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(Config, "STATE_FILE", tmp_path / "state.yaml")
    config = Config(config_path=config_file)

    assert config.cloud.cloud_timeout_seconds == 45
    assert config.cloud.cloud_max_retries == 3
    assert config.cloud.cloud_default_openai_model == "gpt-5-mini"
    assert config.cloud.cloud_default_anthropic_model == "claude-haiku-4-5"


def test_tools_profile_boolean_false_normalizes_to_off(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
tools_enabled: true
tools_profile: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(Config, "STATE_FILE", tmp_path / "state.yaml")
    config = Config(config_path=config_file)

    assert config.tools.tools_enabled is True
    assert config.tools.tools_profile == "off"


def test_state_is_readable_only_by_its_owner_and_replaced_whole(tmp_path: Path, monkeypatch):
    """state.yaml can hold an endpoint URL with credentials in it."""
    state = tmp_path / "state.yaml"
    state.write_text("last_used_cloud_model: old\n", encoding="utf-8")
    state.chmod(0o644)
    monkeypatch.setattr(Config, "STATE_FILE", state)
    config = Config(config_path=tmp_path / "config.yaml")
    config.set_state_value("openai_compatible_base_url", "http://user:secret@host/v1")
    assert state.stat().st_mode & 0o777 == 0o600
    assert "http://user:secret@host/v1" in state.read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.yaml"]
