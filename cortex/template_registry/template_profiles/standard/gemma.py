"""Gemma template profile for Google Gemma models."""

from typing import List, Dict, Any, Tuple
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateConfig, TemplateType


class GemmaProfile(BaseTemplateProfile):
    """Template profile for Google Gemma models."""
    
    def get_default_config(self) -> TemplateConfig:
        """Return the default Gemma configuration."""
        return TemplateConfig(
            name="Gemma",
            description="Google Gemma chat template format",
            template_type=TemplateType.GEMMA,
            supports_system_prompt=True,
            supports_multi_turn=True,
            strip_special_tokens=True,
            stop_sequences=["<end_of_turn>", "<eos>"]
        )
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages using Gemma chat template format."""
        formatted = ""
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                # Gemma system messages are treated as user messages with special formatting
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'user':
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        
        if add_generation_prompt:
            formatted += "<start_of_turn>model\n"
        
        return formatted
    
    def process_response(self, raw_output: str) -> str:
        """Process Gemma model output to clean it up."""
        output = raw_output
        
        # Stop at the first occurrence of any stop token
        stop_tokens = ["<end_of_turn>", "<eos>", "</s>"]
        for token in stop_tokens:
            if token in output:
                output = output.split(token)[0]
        
        # Remove Gemma-specific tokens that might have leaked through
        gemma_tokens = [
            "<start_of_turn>", "model\n", "user\n", "assistant\n"
        ]
        
        for token in gemma_tokens:
            output = output.replace(token, "")
        
        # Remove any role markers that might have been added by incorrect templates
        role_markers = ["Assistant:", "User:", "Human:", "System:", "Model:"]
        for marker in role_markers:
            if output.startswith(marker):
                output = output[len(marker):].strip()
            output = output.replace(f"\n{marker}", "\n")
        
        # Clean up extra whitespace
        lines = output.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Check if this profile can handle Gemma models."""
        model_name_lower = model_name.lower()
        
        # High confidence for Gemma models
        if 'gemma' in model_name_lower:
            return True, 0.9
        
        # Check tokenizer for Gemma-specific tokens
        if tokenizer and hasattr(tokenizer, 'vocab'):
            vocab = getattr(tokenizer, 'vocab', {})
            if isinstance(vocab, dict):
                vocab_str = str(vocab.keys()).lower()
                if '<start_of_turn>' in vocab_str or '<end_of_turn>' in vocab_str:
                    return True, 0.8
        
        # Check for chat template patterns
        if tokenizer and hasattr(tokenizer, 'chat_template'):
            try:
                template_str = str(tokenizer.chat_template).lower()
                if '<start_of_turn>' in template_str or 'gemma' in template_str:
                    return True, 0.8
            except:
                pass
        
        return False, 0.0