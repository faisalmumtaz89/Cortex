"""Configuration manager for template registry."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# Configuration schema for validation
CONFIG_SCHEMA = {
    "version": str,
    "models": dict,
    "global_settings": dict
}

MODEL_CONFIG_SCHEMA = {
    "detected_type": str,
    "user_preference": str,
    "custom_filters": list,
    "show_reasoning": bool,
    "last_updated": str,
    "confidence": float
}


@dataclass
class ModelTemplateConfig:
    """Configuration for a specific model's template."""
    detected_type: str
    user_preference: str
    custom_filters: List[str]
    show_reasoning: bool = False
    last_updated: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class GlobalSettings:
    """Global template settings."""
    auto_detect: bool = True
    prompt_on_unknown: bool = True
    verbose_mode: bool = False
    cache_templates: bool = True
    default_fallback: str = "simple"


class TemplateConfigManager:
    """Manage template configurations and user preferences."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the configuration manager."""
        self.config_path = config_path or (Path.home() / ".cortex" / "template_config.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure matches expected schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check top-level keys
        for key, expected_type in CONFIG_SCHEMA.items():
            if key not in config:
                logger.warning(f"Missing required config key: {key}")
                return False
            if not isinstance(config[key], expected_type):
                logger.warning(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")
                return False
        
        # Validate model configs
        if "models" in config:
            for model_name, model_config in config["models"].items():
                if not isinstance(model_config, dict):
                    logger.warning(f"Invalid model config for {model_name}")
                    return False
                # Validate model config structure
                for key, expected_type in MODEL_CONFIG_SCHEMA.items():
                    if key in model_config and not isinstance(model_config[key], expected_type):
                        logger.warning(f"Invalid type for {model_name}.{key}")
                        return False
        
        return True
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Validate loaded configuration
                if self._validate_config_structure(config):
                    return config
                else:
                    logger.warning("Invalid config structure, using defaults")
                    return self._get_default_config()
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in template config: {e}")
            except IOError as e:
                logger.error(f"Error reading template config file: {e}")
            except PermissionError as e:
                logger.error(f"Permission denied reading template config: {e}")
        
        # Return default config
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "version": "1.0",
            "models": {},
            "global_settings": asdict(GlobalSettings())
        }
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first for atomicity
            temp_path = self.config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self.config_path)
            logger.debug(f"Saved template config to {self.config_path}")
            
        except IOError as e:
            logger.error(f"IO error saving template config: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied saving template config: {e}")
        except OSError as e:
            logger.error(f"OS error saving template config: {e}")
    
    def get_model_config(self, model_name: str) -> Optional[ModelTemplateConfig]:
        """Get configuration for a specific model."""
        if model_name in self.config.get("models", {}):
            data = self.config["models"][model_name]
            return ModelTemplateConfig(
                detected_type=data.get("detected_type", "unknown"),
                user_preference=data.get("user_preference", "auto"),
                custom_filters=data.get("custom_filters", []),
                show_reasoning=data.get("show_reasoning", False),
                last_updated=data.get("last_updated", ""),
                confidence=data.get("confidence", 0.0)
            )
        return None
    
    def save_model_config(self, model_name: str, config: ModelTemplateConfig) -> None:
        """Save configuration for a specific model."""
        if "models" not in self.config:
            self.config["models"] = {}
        
        self.config["models"][model_name] = {
            "detected_type": config.detected_type,
            "user_preference": config.user_preference,
            "custom_filters": config.custom_filters,
            "show_reasoning": config.show_reasoning,
            "last_updated": datetime.now().isoformat(),
            "confidence": config.confidence
        }
        
        self._save_config()
        logger.info(f"Saved template configuration for {model_name}")
    
    def get_global_settings(self) -> GlobalSettings:
        """Get global settings."""
        data = self.config.get("global_settings", {})
        return GlobalSettings(
            auto_detect=data.get("auto_detect", True),
            prompt_on_unknown=data.get("prompt_on_unknown", True),
            verbose_mode=data.get("verbose_mode", False),
            cache_templates=data.get("cache_templates", True),
            default_fallback=data.get("default_fallback", "simple")
        )
    
    def update_global_settings(self, **kwargs) -> None:
        """Update global settings."""
        settings = self.get_global_settings()
        
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        
        self.config["global_settings"] = asdict(settings)
        self._save_config()
    
    def list_configured_models(self) -> List[str]:
        """List all models with saved configurations."""
        return list(self.config.get("models", {}).keys())
    
    def remove_model_config(self, model_name: str) -> bool:
        """Remove configuration for a model."""
        if model_name in self.config.get("models", {}):
            del self.config["models"][model_name]
            self._save_config()
            logger.info(f"Removed template configuration for {model_name}")
            return True
        return False
    
    def export_config(self) -> Dict[str, Any]:
        """Export the entire configuration."""
        return self.config.copy()
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """Import a configuration.
        
        Args:
            config: Configuration dictionary to import
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self._validate_config_structure(config):
            raise ValueError("Invalid configuration structure")
        
        self.config = config
        self._save_config()
        logger.info("Imported template configuration")