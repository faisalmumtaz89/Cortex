"""Alpaca template profile implementation."""

from typing import List, Dict, Any, Tuple
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateConfig, TemplateType


class AlpacaProfile(BaseTemplateProfile):
    """Alpaca instruction format."""
    
    def get_default_config(self) -> TemplateConfig:
        """Return the default Alpaca configuration."""
        return TemplateConfig(
            name="Alpaca",
            description="Alpaca instruction-following format",
            template_type=TemplateType.ALPACA,
            supports_system_prompt=False,
            supports_multi_turn=False,
            strip_special_tokens=False,
            stop_sequences=["### Human:", "### Assistant:", "\n\n###"]
        )
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages in Alpaca style."""
        formatted = ""
        
        # Alpaca is primarily single-turn, but we'll handle multi-turn
        instruction = ""
        input_text = ""
        
        for msg in messages:
            if msg.get('role') == 'system':
                instruction = msg.get('content', '')
            elif msg.get('role') == 'user':
                if instruction:
                    input_text = msg.get('content', '')
                else:
                    instruction = msg.get('content', '')
        
        # Format as Alpaca
        if instruction and input_text:
            formatted = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n"
        elif instruction:
            formatted = f"### Instruction:\n{instruction}\n\n"
        
        if add_generation_prompt:
            formatted += "### Response:\n"
        
        return formatted
    
    def process_response(self, raw_output: str) -> str:
        """Process Alpaca output."""
        output = raw_output
        
        # Remove any instruction markers that might appear
        for marker in ["### Instruction:", "### Input:", "### Response:", "### Human:", "### Assistant:"]:
            if marker in output:
                output = output.split(marker)[0]
        
        return output.strip()
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Check if this profile can handle the model."""
        model_lower = model_name.lower()
        
        # High confidence for Alpaca models
        if 'alpaca' in model_lower:
            return True, 0.95
        
        # Medium confidence for instruction-tuned models
        if any(name in model_lower for name in ['instruct', 'instruction']):
            return True, 0.5
        
        return False, 0.0