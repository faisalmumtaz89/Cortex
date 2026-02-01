"""Base template profile for all template implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TemplateType(Enum):
    """Types of template formats."""
    CHATML = "chatml"
    LLAMA = "llama"
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    OPENAI = "openai"
    REASONING = "reasoning"
    GEMMA = "gemma"
    CUSTOM = "custom"
    SIMPLE = "simple"
    UNKNOWN = "unknown"


@dataclass
class TemplateConfig:
    """Configuration for a template profile."""
    name: str
    description: str
    template_type: TemplateType
    supports_system_prompt: bool = True
    supports_multi_turn: bool = True
    strip_special_tokens: bool = False
    show_reasoning: bool = False
    custom_filters: List[str] = None
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.custom_filters is None:
            self.custom_filters = []
        if self.stop_sequences is None:
            self.stop_sequences = []


class BaseTemplateProfile(ABC):
    """Base class for all template profiles."""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize the template profile."""
        self.config = config or self.get_default_config()
        self._tokenizer = None
    
    @abstractmethod
    def get_default_config(self) -> TemplateConfig:
        """Return the default configuration for this template."""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format a list of messages into the model's expected format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            add_generation_prompt: Whether to add the assistant prompt at the end
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def process_response(self, raw_output: str) -> str:
        """Process the model's raw output to clean it up.
        
        Args:
            raw_output: Raw text from the model
            
        Returns:
            Cleaned output text
        """
        pass
    
    def supports_streaming(self) -> bool:
        """Check if this profile supports streaming response processing.
        
        Returns:
            True if the profile has streaming capabilities, False otherwise
        """
        return hasattr(self, 'process_streaming_response')
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Check if this profile can handle the given model.
        
        Args:
            model_name: Name of the model
            tokenizer: Optional tokenizer object
            
        Returns:
            Tuple of (can_handle, confidence_score)
            confidence_score is between 0.0 and 1.0
        """
        return False, 0.0
    
    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set the tokenizer for this profile."""
        self._tokenizer = tokenizer
    
    def get_stop_sequences(self) -> List[str]:
        """Get the stop sequences for this template."""
        return self.config.stop_sequences.copy()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary representation."""
        return {
            'name': self.config.name,
            'type': self.config.template_type.value,
            'description': self.config.description,
            'supports_system': self.config.supports_system_prompt,
            'supports_multi_turn': self.config.supports_multi_turn,
            'strip_special_tokens': self.config.strip_special_tokens,
            'custom_filters': self.config.custom_filters,
            'stop_sequences': self.config.stop_sequences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseTemplateProfile':
        """Create profile from dictionary representation."""
        config = TemplateConfig(
            name=data.get('name', 'Unknown'),
            description=data.get('description', ''),
            template_type=TemplateType(data.get('type', 'unknown')),
            supports_system_prompt=data.get('supports_system', True),
            supports_multi_turn=data.get('supports_multi_turn', True),
            strip_special_tokens=data.get('strip_special_tokens', False),
            custom_filters=data.get('custom_filters', []),
            stop_sequences=data.get('stop_sequences', [])
        )
        return cls(config)