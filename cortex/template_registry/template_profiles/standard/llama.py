"""Llama template profile implementation."""

from typing import List, Dict, Any, Tuple
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateConfig, TemplateType


class LlamaProfile(BaseTemplateProfile):
    """Llama/Llama2 format with [INST] tokens."""
    
    def get_default_config(self) -> TemplateConfig:
        """Return the default Llama configuration."""
        return TemplateConfig(
            name="Llama",
            description="Llama format with [INST] tokens",
            template_type=TemplateType.LLAMA,
            supports_system_prompt=True,
            supports_multi_turn=True,
            strip_special_tokens=False,
            stop_sequences=["</s>", "[/INST]"]
        )
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages in Llama style."""
        formatted = ""
        
        # Handle system message
        system_msg = "You are a helpful assistant."
        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg.get('content', system_msg)
                break
        
        # Format conversation
        conversation = []
        for msg in messages:
            if msg.get('role') == 'user':
                conversation.append(('user', msg.get('content', '')))
            elif msg.get('role') == 'assistant':
                conversation.append(('assistant', msg.get('content', '')))
        
        # Build the prompt
        if conversation:
            formatted = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
            
            for i, (role, content) in enumerate(conversation):
                if role == 'user':
                    if i > 0:
                        formatted += f"<s>[INST] {content} [/INST] "
                    else:
                        formatted += f"{content} [/INST] "
                elif role == 'assistant':
                    formatted += f"{content} </s>"
        
        return formatted
    
    def process_response(self, raw_output: str) -> str:
        """Process Llama output."""
        output = raw_output
        
        # Remove special tokens
        for token in ["</s>", "<s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]:
            output = output.replace(token, "")
        
        return output.strip()
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Check if this profile can handle the model."""
        model_lower = model_name.lower()
        
        # High confidence for Llama models
        if 'llama' in model_lower or 'codellama' in model_lower:
            return True, 0.9
        
        # Check for specific Llama-based models
        if any(name in model_lower for name in ['vicuna', 'alpaca', 'guanaco']):
            return True, 0.7
        
        # Check tokenizer
        if tokenizer:
            try:
                vocab = getattr(tokenizer, 'get_vocab', lambda: {})()
                if '[INST]' in vocab or '[/INST]' in vocab:
                    return True, 0.8
            except:
                pass
        
        return False, 0.0