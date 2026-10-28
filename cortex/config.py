"""Configuration management for Cortex."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class InferenceConfig(BaseModel):
    """Inference settings."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=32768, ge=1)


class CloudConfig(BaseModel):
    """Cloud inference configuration."""

    cloud_timeout_seconds: int = Field(default=60, ge=1, le=600)
    cloud_max_retries: int = Field(default=2, ge=0, le=10)
    cloud_default_openai_model: str = Field(default="gpt-5.5")
    cloud_default_anthropic_model: str = Field(default="claude-fable-5")
    cloud_azure_endpoint: str = Field(default="")
    cloud_openai_compatible_base_url: str = Field(default="")


class ToolsConfig(BaseModel):
    """Tooling runtime configuration."""

    tools_enabled: bool = True
    tools_profile: Literal["off", "read_only", "edit", "full"] = "full"
    tools_max_iterations: int = Field(default=100, ge=1, le=1000)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "cortex.log")

class ConversationConfig(BaseModel):
    """Conversation settings."""
    auto_save: bool = True
    save_directory: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "conversations")
    max_conversation_history: int = Field(default=100, ge=1)

class SystemConfig(BaseModel):
    """System settings."""
    # Daily release check for Cortex (opt-out). Consumed by the
    # worker runtime; failures are silent and startup never waits on it.
    auto_update_check: bool = True

class DeveloperConfig(BaseModel):
    """Developer settings."""
    debug_mode: bool = False

_SECTIONS: Dict[str, Type[BaseModel]] = {
    "inference": InferenceConfig,
    "cloud": CloudConfig,
    "tools": ToolsConfig,
    "logging": LoggingConfig,
    "conversation": ConversationConfig,
    "system": SystemConfig,
    "developer": DeveloperConfig,
}


# Settings earlier versions accepted without acting on them: one still in a
# config file is named once at startup, so it can be deleted.
_NO_EFFECT = frozenset({
    "auto_scroll", "copy_on_select", "crash_recovery", "enable_branching", "log_rotation",
    "markdown_rendering", "max_log_size", "mouse_support", "performance_logging",
    "plugins_dir", "save_format", "show_performance_metrics", "shutdown_timeout",
    "syntax_highlighting", "templates_dir", "tools_continue_on_reject", "ui_theme",
})
# Read by the OpenAI and Anthropic SDKs, not by Cortex: those providers
# always use their official hosts (cortex/cloud/types.py), so a variable left in
# an environment cannot send their key elsewhere. One that is set is named.
_SDK_BASE_URL_VARIABLES = {"OPENAI_BASE_URL": "openai", "ANTHROPIC_BASE_URL": "anthropic"}
_RETIRED = {
    "tools_idle_timeout_seconds": "it has no effect; cloud_timeout_seconds sets the request "
    "timeout, and a permission prompt waits for your answer",
    **{key: "it has no effect in Cortex, and can be deleted" for key in _NO_EFFECT},
}


class Config:
    """Main configuration class for Cortex."""

    # State file for runtime state (not committed to git)
    STATE_FILE = Path.home() / ".cortex" / "state.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration."""
        self.config_path = config_path or Path.home() / ".cortex" / "config.yaml"
        self._raw_config: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self._file_values: Dict[str, Any] = {}
        self._env_values: Dict[str, Any] = {}
        # Settings given a value that was not valid, so their default applies.
        self._invalid: set = set()

        self.inference: InferenceConfig
        self.cloud: CloudConfig
        self.tools: ToolsConfig
        self.logging: LoggingConfig
        self.conversation: ConversationConfig
        self.system: SystemConfig
        self.developer: DeveloperConfig

        self.load()
        self._load_state()

    def load(self) -> None:
        """Load configuration from YAML file plus CORTEX_* env overrides. A
        setting that is unknown or not valid is reported in `warnings`, and
        the others still apply; cortex/__main__.py logs the warnings once
        logging is set up."""
        self.warnings = []
        file_values: Dict[str, Any] = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                if not isinstance(loaded, dict):
                    raise ValueError("it does not hold key: value settings")
                file_values = loaded
            except Exception as e:
                first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                self.warnings.append(f"Ignored {self.config_path}: {first_line}")
        for key in sorted(str(key) for key in file_values if key not in self._known_keys()):
            hint = _RETIRED.get(key, "")
            self.warnings.append(
                f"Ignored {key} in {self.config_path}: "
                + (hint or "not a Cortex setting")
            )

        for variable, provider in _SDK_BASE_URL_VARIABLES.items():
            if os.environ.get(variable, "").strip():
                self.warnings.append(
                    f"Ignored {variable}: the {provider} provider always uses its official "
                    "host. For another endpoint, use openai-compatible "
                    "(OPENAI_COMPATIBLE_BASE_URL with OPENAI_COMPATIBLE_API_KEY)"
                )

        self._file_values = file_values
        self._env_values = self._env_overrides()
        self._invalid = set()
        self._raw_config = {**file_values, **self._env_values}
        self._parse_config()

    @staticmethod
    def _known_keys() -> set:
        """All flat config keys across section models."""
        keys: set = set()
        for section in _SECTIONS.values():
            keys.update(section.model_fields.keys())
        return keys

    def _env_overrides(self) -> Dict[str, Any]:
        """Read CORTEX_<KEY> environment overrides for known config keys.

        Example: CORTEX_TOOLS_MAX_ITERATIONS=80 overrides tools_max_iterations.
        Unknown CORTEX_* variables (e.g. CORTEX_WORKER_MODE) are ignored.
        """
        known = self._known_keys()
        overrides: Dict[str, Any] = {}
        for name, raw in os.environ.items():
            if not name.startswith("CORTEX_"):
                continue
            key = name[len("CORTEX_"):].lower()
            if key not in known:
                continue
            try:
                overrides[key] = yaml.safe_load(raw)
            except Exception:
                overrides[key] = raw
        return overrides

    def _parse_config(self) -> None:
        """Build each section from its settings. A setting that is not valid
        falls back to the config file's value when it came from a CORTEX_*
        variable, otherwise to its default; each fallback is reported."""
        for name, section in _SECTIONS.items():
            values = self._get_section({
                k: v for k, v in self._raw_config.items() if k in section.model_fields
            })
            if name == "tools":
                values = self._normalize_tools_section(values)
            while True:
                try:
                    setattr(self, name, section(**values))
                    break
                except ValidationError as exc:
                    bad = {
                        str(error["loc"][0]): error["msg"]
                        for error in exc.errors()
                        if error["loc"] and error["loc"][0] in values
                    }
                    if not bad:
                        self.warnings.append(f"Ignored the {name} settings ({exc}); using defaults")
                        setattr(self, name, section())
                        break
                    for key, problem in bad.items():
                        self._report_invalid(key, values.pop(key), problem)
                        from_env = key in self._env_values
                        if from_env and self._file_values.get(key) is not None:
                            self._env_values.pop(key)
                            values[key] = self._file_values[key]

    def _report_invalid(self, key: str, value: Any, problem: str) -> None:
        shown = repr(value)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        source = f"CORTEX_{key.upper()}" if key in self._env_values else key
        to_file = key in self._env_values and self._file_values.get(key) is not None
        fallback = "using the value in the config file" if to_file else "using the default"
        if not to_file:
            self._invalid.add(key)
        self.warnings.append(f"Ignored {source}: {shown} ({problem}); {fallback}")

    def _get_section(self, section_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get a configuration section."""
        return {k: v for k, v in section_dict.items() if v is not None}

    def _normalize_tools_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tooling keys so old/bad values don't break startup."""
        data = dict(section)

        profile = data.get("tools_profile")
        if isinstance(profile, bool):
            data["tools_profile"] = "read_only" if profile else "off"
        elif profile is not None:
            normalized = str(profile).strip().lower()
            aliases = {
                "false": "off",
                "none": "off",
                "disabled": "off",
                "readonly": "read_only",
                "read-only": "read_only",
                "true": "read_only",
                "patch": "edit",
            }
            data["tools_profile"] = aliases.get(normalized, normalized)

        return data

    def _load_state(self) -> None:
        """Load runtime state from state file."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, 'r') as f:
                    self._state = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to load state from %s: %s", self.STATE_FILE, e)
                self._state = {}
        # Selection keys for local models, which Cortex does not run.
        obsolete = [key for key in ("last_used_backend", "last_used_model") if key in self._state]
        if obsolete:
            for key in obsolete:
                del self._state[key]
            self._save_state()

    def _save_state(self) -> None:
        """Save runtime state to the state file, readable only by its owner
        (an endpoint URL in it can carry credentials), replaced in one step so
        a reader never sees half of it."""
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        partial = self.STATE_FILE.with_name(f".{self.STATE_FILE.name}.{os.getpid()}")
        try:
            descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w") as f:
                yaml.dump(self._state, f, default_flow_style=False)
            os.replace(partial, self.STATE_FILE)
        except Exception as e:
            partial.unlink(missing_ok=True)
            logger.warning("Failed to save state to %s: %s", self.STATE_FILE, e)

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get a runtime state value."""
        return self._state.get(key, default)

    def set_state_value(self, key: str, value: Any) -> None:
        """Set a runtime state value and persist it."""
        self._state[key] = value
        self._save_state()

    def settings_in_effect(self) -> List[Tuple[str, Any, str]]:
        """The settings config.yaml or a CORTEX_* variable changed, by name:
        each with the value in effect and where it came from."""
        settings = []
        for name, section in _SECTIONS.items():
            for key in section.model_fields:
                if key in self._invalid:
                    continue
                if key in self._env_values:
                    source = f"CORTEX_{key.upper()}"
                elif self._file_values.get(key) is not None:
                    source = self.config_path.name
                else:
                    continue
                settings.append((key, getattr(getattr(self, name), key), source))
        return sorted(settings)

    def is_setting_explicit(self, key: str) -> bool:
        """Return True if a config key was explicitly set in config.yaml."""
        return key in self._raw_config

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(path={self.config_path})"
