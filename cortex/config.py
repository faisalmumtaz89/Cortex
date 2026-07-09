"""Configuration management for Cortex."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class GPUConfig(BaseModel):
    """GPU-specific configuration for Apple Silicon."""
    compute_backend: Literal["metal"] = "metal"
    force_gpu: Literal[True] = True
    metal_performance_shaders: bool = True
    mlx_backend: bool = True
    gpu_memory_fraction: float = Field(default=0.85, ge=0.1, le=1.0)
    gpu_cores: int = Field(default=16, ge=1, le=128)
    metal_api_version: int = Field(default=3, ge=3)
    shader_cache: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "metal_shaders")
    compile_shaders_on_start: bool = True  # Fixed and enabled!
    gpu_optimization_level: str = Field(default="maximum")

    @field_validator("compute_backend")
    def validate_backend(cls, v):
        if v != "metal":
            raise ValueError("Only 'metal' backend is supported for Apple Silicon GPU")
        return v

    @field_validator("gpu_cores")
    def validate_gpu_cores(cls, v):
        if v < 1 or v > 128:
            raise ValueError("GPU cores must be between 1 and 128")
        return v

class MemoryConfig(BaseModel):
    """Memory management configuration."""
    unified_memory: Literal[True] = True
    max_gpu_memory: str = Field(default="20GB")
    cpu_offload: Literal[False] = False
    memory_pool_size: str = Field(default="20GB")
    kv_cache_size: str = Field(default="2GB")
    activation_memory: str = Field(default="2GB")

    @field_validator("cpu_offload")
    def validate_no_cpu_offload(cls, v):
        if v:
            raise ValueError("CPU offloading is not allowed - GPU only execution")
        return v

    def parse_memory_size(self, size_str: str) -> int:
        """Convert memory size string to bytes."""
        size_str = size_str.upper().strip()
        if size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        else:
            return int(size_str)

class PerformanceConfig(BaseModel):
    """Performance settings."""
    batch_size: int = Field(default=8, ge=1, le=32)
    max_batch_size: int = Field(default=16, ge=1, le=64)
    use_flash_attention: bool = True
    use_fused_ops: bool = True
    num_threads: int = Field(default=1, ge=1, le=4)
    context_length: int = Field(default=32768, ge=512)
    sliding_window_size: int = Field(default=4096, ge=512)

class InferenceConfig(BaseModel):
    """Inference settings."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    stream_output: bool = True
    seed: int = Field(default=-1)

class ModelConfig(BaseModel):
    """Model selection state (local models are served by Lumen)."""

    default_model: str = Field(default="")
    last_used_model: str = Field(default="")  # Track the last used model


class CloudConfig(BaseModel):
    """Cloud inference configuration."""

    cloud_enabled: bool = True
    cloud_timeout_seconds: int = Field(default=60, ge=1, le=600)
    cloud_max_retries: int = Field(default=2, ge=0, le=10)
    cloud_default_openai_model: str = Field(default="gpt-5.5")
    cloud_default_anthropic_model: str = Field(default="claude-fable-5")
    cloud_azure_endpoint: str = Field(default="")


class LumenConfig(BaseModel):
    """Managed local Lumen inference server configuration."""

    lumen_binary: str = Field(default="lumen")
    lumen_server_binary: str = Field(default="lumen-server")
    lumen_port: int = Field(default=0, ge=0, le=65535)  # 0 = pick a free port
    lumen_context_len: int = Field(default=0, ge=0)  # 0 = lumen's default
    lumen_startup_timeout_seconds: int = Field(default=180, ge=10, le=900)
    lumen_log_level: str = Field(default="warn")


class ToolsConfig(BaseModel):
    """Tooling runtime configuration."""

    tools_enabled: bool = True
    tools_profile: Literal["off", "read_only", "edit", "full"] = "full"
    tools_max_iterations: int = Field(default=25, ge=1, le=100)
    tools_idle_timeout_seconds: int = Field(default=45, ge=1, le=600)
    tools_continue_on_reject: bool = False


class UIConfig(BaseModel):
    """UI configuration."""
    ui_theme: str = Field(default="default")
    syntax_highlighting: bool = True
    markdown_rendering: bool = True
    show_performance_metrics: bool = True
    show_gpu_utilization: bool = True
    auto_scroll: bool = True
    copy_on_select: bool = True
    mouse_support: bool = True

class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "cortex.log")
    log_rotation: str = Field(default="daily")
    max_log_size: str = Field(default="100MB")
    performance_logging: bool = True
    gpu_metrics_interval: int = Field(default=1000, ge=100)

class ConversationConfig(BaseModel):
    """Conversation settings."""
    auto_save: bool = True
    save_format: str = Field(default="json")
    save_directory: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "conversations")
    max_conversation_history: int = Field(default=100, ge=1)
    enable_branching: bool = True

class SystemConfig(BaseModel):
    """System settings."""
    startup_checks: List[str] = Field(
        default_factory=lambda: [
            "verify_metal_support",
            "check_gpu_memory",
            "validate_models",
            "compile_shaders"
        ]
    )
    shutdown_timeout: int = Field(default=5, ge=1)
    crash_recovery: bool = True
    # Daily release check for Cortex and Lumen (opt-out). Consumed by the
    # worker runtime; failures are silent and startup never waits on it.
    auto_update_check: bool = True

class DeveloperConfig(BaseModel):
    """Developer settings."""
    debug_mode: bool = False
    profile_inference: bool = False
    metal_capture: bool = False
    verbose_gpu_logs: bool = False

class PathsConfig(BaseModel):
    """Path configuration."""
    templates_dir: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "templates")
    plugins_dir: Path = Field(default_factory=lambda: Path.home() / ".cortex" / "plugins")

class Config:
    """Main configuration class for Cortex."""

    # State file for runtime state (not committed to git)
    STATE_FILE = Path.home() / ".cortex" / "state.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration."""
        self.config_path = config_path or Path("config.yaml")
        self._raw_config: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}

        self.gpu: GPUConfig
        self.memory: MemoryConfig
        self.performance: PerformanceConfig
        self.inference: InferenceConfig
        self.model: ModelConfig
        self.cloud: CloudConfig
        self.lumen: LumenConfig
        self.tools: ToolsConfig
        self.ui: UIConfig
        self.logging: LoggingConfig
        self.conversation: ConversationConfig
        self.system: SystemConfig
        self.developer: DeveloperConfig
        self.paths: PathsConfig

        self.load()
        self._load_state()

    def load(self) -> None:
        """Load configuration from YAML file plus CORTEX_* env overrides."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self._raw_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                self._raw_config = {}
        else:
            self._raw_config = {}

        self._raw_config.update(self._env_overrides())
        if not self._raw_config:
            self._use_defaults()
            return

        self._parse_config()

    @staticmethod
    def _known_keys() -> set:
        """All flat config keys across section models."""
        sections: List[type[BaseModel]] = [
            GPUConfig, MemoryConfig, PerformanceConfig, InferenceConfig,
            ModelConfig, CloudConfig, LumenConfig, ToolsConfig, UIConfig, LoggingConfig,
            ConversationConfig, SystemConfig, DeveloperConfig, PathsConfig,
        ]
        keys: set = set()
        for section in sections:
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

    def _use_defaults(self) -> None:
        """Use default configuration values."""
        self.gpu = GPUConfig()
        self.memory = MemoryConfig()
        self.performance = PerformanceConfig()
        self.inference = InferenceConfig()
        self.model = ModelConfig()
        self.cloud = CloudConfig()
        self.lumen = LumenConfig()
        self.tools = ToolsConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()
        self.conversation = ConversationConfig()
        self.system = SystemConfig()
        self.developer = DeveloperConfig()
        self.paths = PathsConfig()

    def _parse_config(self) -> None:
        """Parse configuration from raw dictionary."""
        try:
            self.gpu = GPUConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["compute_backend", "force_gpu", "metal_performance_shaders",
                        "mlx_backend", "gpu_memory_fraction", "gpu_cores",
                        "metal_api_version", "shader_cache", "compile_shaders_on_start",
                        "gpu_optimization_level"]
            }))

            self.memory = MemoryConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["unified_memory", "max_gpu_memory", "cpu_offload",
                        "memory_pool_size", "kv_cache_size", "activation_memory"]
            }))

            self.performance = PerformanceConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["batch_size", "max_batch_size", "use_flash_attention",
                        "use_fused_ops", "num_threads", "context_length",
                        "sliding_window_size"]
            }))

            self.inference = InferenceConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["temperature", "top_p", "top_k", "repetition_penalty",
                        "max_tokens", "stream_output", "seed"]
            }))

            self.model = ModelConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["default_model", "last_used_model"]
            }))

            self.cloud = CloudConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in [
                    "cloud_enabled",
                    "cloud_timeout_seconds",
                    "cloud_max_retries",
                    "cloud_default_openai_model",
                    "cloud_default_anthropic_model",
                    "cloud_azure_endpoint",
                ]
            }))

            self.lumen = LumenConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in [
                    "lumen_binary",
                    "lumen_server_binary",
                    "lumen_port",
                    "lumen_context_len",
                    "lumen_startup_timeout_seconds",
                    "lumen_log_level",
                ]
            }))

            raw_tools = self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in [
                    "tools_enabled",
                    "tools_profile",
                    "tools_max_iterations",
                    "tools_idle_timeout_seconds",
                    "tools_continue_on_reject",
                ]
            })
            self.tools = ToolsConfig(**self._normalize_tools_section(raw_tools))

            self.ui = UIConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["ui_theme", "syntax_highlighting", "markdown_rendering",
                        "show_performance_metrics", "show_gpu_utilization",
                        "auto_scroll", "copy_on_select", "mouse_support"]
            }))

            self.logging = LoggingConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["log_level", "log_file", "log_rotation", "max_log_size",
                        "performance_logging", "gpu_metrics_interval"]
            }))

            self.conversation = ConversationConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["auto_save", "save_format", "save_directory",
                        "max_conversation_history", "enable_branching"]
            }))

            self.system = SystemConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["startup_checks", "shutdown_timeout", "crash_recovery",
                        "auto_update_check"]
            }))

            self.developer = DeveloperConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["debug_mode", "profile_inference", "metal_capture",
                        "verbose_gpu_logs"]
            }))

            self.paths = PathsConfig(**self._get_section({
                k: v for k, v in self._raw_config.items()
                if k in ["templates_dir", "plugins_dir"]
            }))

        except Exception as e:
            print(f"Error parsing configuration: {e}")
            self._use_defaults()

    def _get_section(self, section_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get a configuration section."""
        return {k: v for k, v in section_dict.items() if v is not None}

    def _normalize_tools_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tooling keys so old/bad values don't break startup."""
        data = dict(section)

        enabled = data.get("tools_enabled")
        if isinstance(enabled, str):
            data["tools_enabled"] = enabled.strip().lower() in {"1", "true", "yes", "on"}

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

    def validate_gpu_requirements(self) -> bool:
        """Validate that GPU requirements are met."""
        if self.gpu.compute_backend != "metal":
            print("❌ Only Metal backend is supported")
            return False

        if not self.gpu.force_gpu:
            print("❌ GPU execution is mandatory")
            return False

        if self.memory.cpu_offload:
            print("❌ CPU offloading is not allowed")
            return False

        return True

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        save_path = path or self.config_path

        # Keys that belong in state file, not config file
        state_keys = {"last_used_model"}

        # Convert Path objects to strings for YAML serialization
        config_dict = {}
        for section in [self.gpu, self.memory, self.performance, self.inference,
                       self.model, self.cloud, self.tools, self.ui, self.logging, self.conversation,
                       self.system, self.developer, self.paths]:
            section_dict = section.model_dump()
            # Convert Path objects to strings and exclude state keys
            for key, value in section_dict.items():
                if key in state_keys:
                    continue  # Skip state keys - they go in state file
                if isinstance(value, Path):
                    section_dict[key] = str(value)
            # Remove state keys from section_dict
            for key in state_keys:
                section_dict.pop(key, None)
            config_dict.update(section_dict)

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def _load_state(self) -> None:
        """Load runtime state from state file."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, 'r') as f:
                    self._state = yaml.safe_load(f) or {}
                # Apply state to model config
                if "last_used_model" in self._state:
                    self.model.last_used_model = self._state["last_used_model"]
            except Exception as e:
                print(f"Warning: Failed to load state from {self.STATE_FILE}: {e}")
                self._state = {}

    def _save_state(self) -> None:
        """Save runtime state to state file."""
        # Ensure directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.STATE_FILE, 'w') as f:
                yaml.dump(self._state, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Failed to save state to {self.STATE_FILE}: {e}")

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get a runtime state value."""
        return self._state.get(key, default)

    def set_state_value(self, key: str, value: Any) -> None:
        """Set a runtime state value and persist it."""
        self._state[key] = value
        self._save_state()

    def is_setting_explicit(self, key: str) -> bool:
        """Return True if a config key was explicitly set in config.yaml."""
        return key in self._raw_config

    def update_last_used_model(self, model_name: str) -> None:
        """Update the last used model and save to state file."""
        self.model.last_used_model = model_name
        self._state["last_used_model"] = model_name
        self._save_state()

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(gpu={self.gpu.compute_backend}, memory={self.memory.max_gpu_memory})"
