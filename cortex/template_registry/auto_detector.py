"""Automatic template detection for models."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateType
from cortex.template_registry.template_profiles.standard import (
    ChatMLProfile, LlamaProfile, AlpacaProfile, SimpleProfile, GemmaProfile
)
from cortex.template_registry.template_profiles.complex import ReasoningProfile

logger = logging.getLogger(__name__)


class TemplateDetector:
    """Automatically detect the best template for a model."""
    
    def __init__(self):
        """Initialize the detector with all available profiles."""
        self.profiles = [
            ReasoningProfile(),
            GemmaProfile(),  # Add Gemma profile with high priority
            ChatMLProfile(),
            LlamaProfile(),
            AlpacaProfile(),
            SimpleProfile()  # Fallback
        ]
    
    def detect_template(
        self, 
        model_name: str, 
        model_path: Optional[Path] = None,
        tokenizer: Any = None
    ) -> Tuple[BaseTemplateProfile, float]:
        """Detect the best template for a model.
        
        Args:
            model_name: Name of the model
            model_path: Optional path to model files
            tokenizer: Optional tokenizer object
            
        Returns:
            Tuple of (best_profile, confidence_score)
        """
        candidates = []
        
        # Check each profile
        for profile in self.profiles:
            can_handle, confidence = profile.can_handle(model_name, tokenizer)
            if can_handle:
                candidates.append((profile, confidence))
                logger.debug(f"Profile {profile.config.name} can handle {model_name} with confidence {confidence}")
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_profile, confidence = candidates[0]
            logger.info(f"Selected {best_profile.config.name} template for {model_name} (confidence: {confidence:.2f})")
            return best_profile, confidence
        
        # Fallback to simple profile
        logger.warning(f"No specific template found for {model_name}, using simple fallback")
        return SimpleProfile(), 0.1
    
    def detect_from_tokenizer_config(self, tokenizer: Any) -> Optional[TemplateType]:
        """Try to detect template type from tokenizer configuration."""
        if not tokenizer:
            return None
        
        try:
            # Check for chat_template attribute
            if hasattr(tokenizer, 'chat_template'):
                template_str = str(tokenizer.chat_template)
                
                # Check for known patterns
                if '<|im_start|>' in template_str:
                    return TemplateType.CHATML
                elif '[INST]' in template_str:
                    return TemplateType.LLAMA
                elif '<|channel|>' in template_str:
                    return TemplateType.REASONING
                elif '### Instruction' in template_str:
                    return TemplateType.ALPACA
        except (AttributeError, TypeError) as e:
            logger.debug(f"Error accessing tokenizer chat_template: {e}")
        
        try:
            # Check tokenizer config
            if hasattr(tokenizer, 'tokenizer_config'):
                config = tokenizer.tokenizer_config
                if isinstance(config, dict):
                    # Check for model type hints
                    model_type = config.get('model_type', '').lower()
                    if 'llama' in model_type:
                        return TemplateType.LLAMA
                    elif 'gpt' in model_type:
                        return TemplateType.OPENAI
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Error accessing tokenizer config: {e}")
        
        return None
    
    def test_template(
        self, 
        profile: BaseTemplateProfile,
        test_prompt: str = "Hello, how are you?"
    ) -> Dict[str, Any]:
        """Test a template profile with a sample prompt.
        
        Args:
            profile: Template profile to test
            test_prompt: Test prompt to use
            
        Returns:
            Dictionary with test results
        """
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
        formatted = profile.format_messages(messages)
        
        # Simulate a response
        sample_response = "I'm doing well, thank you for asking! How can I help you today?"
        
        # Add reasoning tokens for reasoning profile
        if profile.config.template_type == TemplateType.REASONING:
            sample_response = (
                "<|channel|>analysis<|message|>The user is greeting me and asking about my state. "
                "I should respond politely.<|end|>"
                f"<|channel|>final<|message|>{sample_response}<|end|>"
            )
        
        processed = profile.process_response(sample_response)
        
        return {
            'profile_name': profile.config.name,
            'formatted_prompt': formatted,
            'raw_response': sample_response,
            'processed_response': processed,
            'template_type': profile.config.template_type.value
        }