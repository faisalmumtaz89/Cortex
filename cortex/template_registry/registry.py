"""Main template registry for managing model templates."""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from rich.console import Console

from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateType
from cortex.template_registry.template_profiles.standard import (
    ChatMLProfile, LlamaProfile, AlpacaProfile, SimpleProfile, GemmaProfile
)
from cortex.template_registry.template_profiles.complex import ReasoningProfile
from cortex.template_registry.auto_detector import TemplateDetector
from cortex.template_registry.config_manager import TemplateConfigManager, ModelTemplateConfig
from cortex.template_registry.interactive import InteractiveTemplateSetup

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """Central registry for managing model templates."""
    
    def __init__(self, config_path: Optional[Path] = None, console: Optional[Console] = None):
        """Initialize the template registry.
        
        Args:
            config_path: Optional path to configuration file
            console: Optional Rich console for interactive output
        """
        self.config_manager = TemplateConfigManager(config_path)
        self.detector = TemplateDetector()
        self.interactive = InteractiveTemplateSetup(console)
        self.console = console or Console()
        
        # Cache of loaded profiles
        self._profile_cache: Dict[str, BaseTemplateProfile] = {}
        
        # Profile type mapping
        self._profile_types = {
            TemplateType.CHATML: ChatMLProfile,
            TemplateType.LLAMA: LlamaProfile,
            TemplateType.ALPACA: AlpacaProfile,
            TemplateType.REASONING: ReasoningProfile,
            TemplateType.SIMPLE: SimpleProfile,
            TemplateType.GEMMA: GemmaProfile,
            TemplateType.CUSTOM: GemmaProfile  # Use Gemma as default custom template
        }
    
    def setup_model(
        self, 
        model_name: str,
        tokenizer: Any = None,
        interactive: bool = True,
        force_setup: bool = False
    ) -> BaseTemplateProfile:
        """Setup or retrieve template for a model.
        
        Args:
            model_name: Name of the model
            tokenizer: Optional tokenizer object
            interactive: Whether to use interactive setup
            force_setup: Force re-setup even if config exists
            
        Returns:
            Configured template profile
        """
        # Check cache first
        if model_name in self._profile_cache and not force_setup:
            logger.debug(f"Using cached profile for {model_name}")
            return self._profile_cache[model_name]
        
        # Check saved configuration
        config = self.config_manager.get_model_config(model_name)
        global_settings = self.config_manager.get_global_settings()
        
        if config and not force_setup:
            # Load saved configuration
            profile = self._load_profile_from_config(config)
            if profile:
                self._profile_cache[model_name] = profile
                logger.info(f"Loaded saved template configuration for {model_name}")
                return profile
        
        # Auto-detect if enabled
        if global_settings.auto_detect or force_setup:
            detected_profile, confidence = self.detector.detect_template(model_name, tokenizer=tokenizer)
            
            # Check if we should prompt user
            should_prompt = (
                interactive and 
                global_settings.prompt_on_unknown and 
                (confidence < 0.5 or force_setup)
            )
            
            if should_prompt:
                # Interactive setup
                config = self.interactive.setup_model_template(model_name, tokenizer, config)
                self.config_manager.save_model_config(model_name, config)
                profile = self._load_profile_from_config(config)
            else:
                # Use detected profile
                config = ModelTemplateConfig(
                    detected_type=detected_profile.config.template_type.value,
                    user_preference="auto",
                    custom_filters=detected_profile.config.custom_filters,
                    show_reasoning=False,
                    confidence=confidence
                )
                
                if global_settings.cache_templates:
                    self.config_manager.save_model_config(model_name, config)
                
                profile = detected_profile
            
            self._profile_cache[model_name] = profile
            return profile
        
        # Fallback to simple profile
        logger.warning(f"Using fallback template for {model_name}")
        profile = SimpleProfile()
        self._profile_cache[model_name] = profile
        return profile
    
    def get_template(self, model_name: str) -> Optional[BaseTemplateProfile]:
        """Get template for a model without setup.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Template profile if configured, None otherwise
        """
        # Check cache
        if model_name in self._profile_cache:
            return self._profile_cache[model_name]
        
        # Check saved configuration
        config = self.config_manager.get_model_config(model_name)
        if config:
            profile = self._load_profile_from_config(config)
            if profile:
                self._profile_cache[model_name] = profile
                return profile
        
        return None
    
    def configure_template(
        self, 
        model_name: str,
        interactive: bool = True,
        **kwargs
    ) -> BaseTemplateProfile:
        """Configure or reconfigure template for a model.
        
        Args:
            model_name: Name of the model
            interactive: Whether to use interactive configuration
            **kwargs: Configuration overrides
            
        Returns:
            Updated template profile
        """
        # Get current configuration
        config = self.config_manager.get_model_config(model_name)
        
        if interactive:
            if config:
                # Quick adjust existing
                config = self.interactive.quick_adjust_template(model_name, config)
            else:
                # Full setup
                config = self.interactive.setup_model_template(model_name)
            
            self.config_manager.save_model_config(model_name, config)
        else:
            # Apply kwargs overrides
            if not config:
                # Create default config
                profile, confidence = self.detector.detect_template(model_name)
                config = ModelTemplateConfig(
                    detected_type=profile.config.template_type.value,
                    user_preference="custom",
                    custom_filters=[],
                    confidence=confidence
                )
            
            # Apply overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            self.config_manager.save_model_config(model_name, config)
        
        # Load and cache updated profile
        profile = self._load_profile_from_config(config)
        self._profile_cache[model_name] = profile
        return profile
    
    def _load_profile_from_config(self, config: ModelTemplateConfig) -> Optional[BaseTemplateProfile]:
        """Load a profile based on configuration.
        
        Args:
            config: Model template configuration
            
        Returns:
            Configured profile or None
        """
        try:
            # Determine template type
            template_type = TemplateType(config.detected_type)
            
            # Get profile class
            profile_class = self._profile_types.get(template_type)
            if not profile_class:
                logger.warning(f"Unknown template type: {config.detected_type}")
                return SimpleProfile()
            
            # Create and configure profile
            profile = profile_class()
            
            # Apply configuration
            if config.custom_filters:
                profile.config.custom_filters = config.custom_filters
            
            if hasattr(profile.config, 'show_reasoning'):
                profile.config.show_reasoning = config.show_reasoning
            
            # Handle user preferences
            if config.user_preference == "simple" and template_type == TemplateType.REASONING:
                profile.config.show_reasoning = False
            elif config.user_preference == "full" and template_type == TemplateType.REASONING:
                profile.config.show_reasoning = True
            
            return profile
            
        except ValueError as e:
            # Invalid template type enum value
            logger.error(f"Invalid template type '{config.detected_type}': {e}")
            return SimpleProfile()
        except AttributeError as e:
            # Missing expected attributes
            logger.error(f"Profile configuration error: {e}")
            return SimpleProfile()
        except TypeError as e:
            # Type-related errors in profile instantiation
            logger.error(f"Profile instantiation error: {e}")
            return SimpleProfile()
    
    def format_messages(
        self, 
        model_name: str,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """Format messages for a specific model.
        
        Args:
            model_name: Name of the model
            messages: List of message dictionaries
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Formatted prompt string
        """
        profile = self.get_template(model_name)
        if not profile:
            profile = self.setup_model(model_name, interactive=False)
        
        return profile.format_messages(messages, add_generation_prompt)
    
    def process_response(self, model_name: str, raw_output: str) -> str:
        """Process model response using appropriate template.
        
        Args:
            model_name: Name of the model
            raw_output: Raw model output
            
        Returns:
            Processed output string
        """
        profile = self.get_template(model_name)
        if not profile:
            profile = SimpleProfile()
        
        return profile.process_response(raw_output)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available template types.
        
        Returns:
            List of template information
        """
        templates = []
        for template_type, profile_class in self._profile_types.items():
            profile = profile_class()
            templates.append({
                'type': template_type.value,
                'name': profile.config.name,
                'description': profile.config.description,
                'supports_system': profile.config.supports_system_prompt,
                'supports_multi_turn': profile.config.supports_multi_turn
            })
        return templates
    
    def list_configured_models(self) -> List[str]:
        """List all models with saved configurations.
        
        Returns:
            List of model names
        """
        return self.config_manager.list_configured_models()
    
    def reset_model_config(self, model_name: str) -> bool:
        """Reset model configuration to defaults.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if reset successful
        """
        # Remove from cache
        if model_name in self._profile_cache:
            del self._profile_cache[model_name]
        
        # Remove saved config
        return self.config_manager.remove_model_config(model_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status information.
        
        Returns:
            Status dictionary
        """
        global_settings = self.config_manager.get_global_settings()
        
        return {
            'configured_models': len(self.list_configured_models()),
            'cached_profiles': len(self._profile_cache),
            'available_templates': len(self._profile_types),
            'global_settings': {
                'auto_detect': global_settings.auto_detect,
                'prompt_on_unknown': global_settings.prompt_on_unknown,
                'cache_templates': global_settings.cache_templates,
                'default_fallback': global_settings.default_fallback
            }
        }