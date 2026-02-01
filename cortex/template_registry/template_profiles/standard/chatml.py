"""ChatML template profile implementation."""

from typing import List, Dict, Any, Tuple
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateConfig, TemplateType


class ChatMLProfile(BaseTemplateProfile):
    """ChatML format used by models like Qwen, OpenHermes, etc."""
    
    def get_default_config(self) -> TemplateConfig:
        """Return the default ChatML configuration."""
        return TemplateConfig(
            name="ChatML",
            description="ChatML format with <|im_start|> and <|im_end|> tokens",
            template_type=TemplateType.CHATML,
            supports_system_prompt=True,
            supports_multi_turn=True,
            strip_special_tokens=False,
            stop_sequences=["<|im_end|>", "<|endoftext|>"]
        )
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages in ChatML style."""
        formatted = ""
        
        # Add default system message if none provided
        if not messages or messages[0].get('role') != 'system':
            formatted += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def process_response(self, raw_output: str) -> str:
        """Process ChatML output."""
        # Remove special tokens if they appear in output
        output = raw_output
        
        # Remove end tokens
        for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            output = output.replace(token, "")
        
        # Clean up any role markers that might appear
        if output.startswith("assistant\n"):
            output = output[10:]  # Remove "assistant\n"
        
        return output.strip()
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Check if this profile can handle the model."""
        model_lower = model_name.lower()
        
        # High confidence for known ChatML models
        if any(name in model_lower for name in ['qwen', 'openhermes', 'neural-chat']):
            return True, 0.9
        
        # Check tokenizer for ChatML tokens
        if tokenizer:
            try:
                vocab = getattr(tokenizer, 'get_vocab', lambda: {})()
                if '<|im_start|>' in vocab or '<|im_end|>' in vocab:
                    return True, 0.8
                
                # Check special tokens
                special_tokens = getattr(tokenizer, 'special_tokens_map', {})
                if any('<|im_start|>' in str(v) for v in special_tokens.values()):
                    return True, 0.8
            except:
                pass
        
        # Check for ChatML in model name
        if 'chatml' in model_lower or 'chat-ml' in model_lower:
            return True, 0.95
        
        return False, 0.0