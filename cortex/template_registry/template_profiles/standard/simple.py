"""Simple template profile for basic conversation."""

from typing import List, Dict, Any, Tuple
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateConfig, TemplateType


class SimpleProfile(BaseTemplateProfile):
    """Simple conversation format without special tokens."""
    
    def get_default_config(self) -> TemplateConfig:
        """Return the default simple configuration."""
        return TemplateConfig(
            name="Simple",
            description="Simple conversation format without special tokens",
            template_type=TemplateType.SIMPLE,
            supports_system_prompt=True,
            supports_multi_turn=True,
            strip_special_tokens=True,
            stop_sequences=["Human:", "User:", "Assistant:"]
        )
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages in simple style."""
        formatted = ""
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted += f"System: {content}\n\n"
            elif role == 'user':
                formatted += f"Human: {content}\n\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n\n"
        
        if add_generation_prompt:
            formatted += "Assistant: "
        
        return formatted
    
    def process_response(self, raw_output: str) -> str:
        """Process simple output - minimal processing."""
        output = raw_output
        
        # Remove any role markers
        for marker in ["Human:", "User:", "Assistant:", "System:"]:
            if output.startswith(marker):
                output = output[len(marker):].strip()
        
        # Remove any common special tokens
        special_tokens = [
            "<|endoftext|>", "<|end|>", "</s>", "<s>",
            "[INST]", "[/INST]", "###"
        ]
        
        for token in special_tokens:
            output = output.replace(token, "")
        
        return output.strip()
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Simple profile can handle any model as a fallback."""
        # This is a universal fallback, so always return low confidence
        return True, 0.1