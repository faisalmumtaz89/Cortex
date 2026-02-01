"""Reasoning-aware template profile for models with internal reasoning."""

import re
from typing import List, Dict, Any, Tuple, Optional
from cortex.template_registry.template_profiles.base import BaseTemplateProfile, TemplateConfig, TemplateType


class ReasoningProfile(BaseTemplateProfile):
    """Profile for models with internal reasoning/chain-of-thought outputs."""
    
    def __init__(self):
        """Initialize the reasoning profile with streaming state."""
        super().__init__()
        self._streaming_state = {
            'in_final': False,
            'buffer': '',
            'final_marker_seen': False
        }
    
    def get_default_config(self) -> TemplateConfig:
        """Return the default reasoning configuration."""
        return TemplateConfig(
            name="Reasoning",
            description="Models with internal reasoning/analysis channels",
            template_type=TemplateType.REASONING,
            supports_system_prompt=True,
            supports_multi_turn=True,
            strip_special_tokens=True,
            show_reasoning=False,  # By default, hide internal reasoning
            custom_filters=[
                "<|channel|>", "<|message|>", "<|end|>", 
                "<|start|>", "<|return|>", "<|endofprompt|>"
            ],
            stop_sequences=["<|return|>", "<|endoftext|>", "<|endofprompt|>"]
        )
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages for reasoning-aware models."""
        formatted = ""
        
        # Handle system message
        system_msg = None
        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg.get('content', '')
                break
        
        if system_msg:
            formatted += f"<|start|>system<|message|>{system_msg}<|end|>"
        
        # Format conversation
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                formatted += f"<|start|>user<|message|>{content}<|end|>"
            elif role == 'assistant' and content:
                # For assistant messages, use the final channel
                formatted += f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
        
        if add_generation_prompt:
            formatted += "<|start|>assistant"
        
        return formatted
    
    def process_response(self, raw_output: str) -> str:
        """Process reasoning model output to extract clean response."""
        output = raw_output
        
        # If we're showing reasoning, keep everything but clean up formatting
        if self.config.show_reasoning:
            return self._clean_reasoning_output(output)
        
        # Otherwise, extract only the final response
        return self._extract_final_response(output)
    
    def _extract_final_response(self, output: str) -> str:
        """Extract only the final response, hiding internal reasoning."""
        # Pattern to find final channel content
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)'
        final_matches = re.findall(final_pattern, output, re.DOTALL)
        
        if final_matches:
            # Return the last final response
            return final_matches[-1].strip()
        
        # Fallback: look for content after channel markers
        channel_pattern = r'<\|channel\|>\w+<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)'
        matches = re.findall(channel_pattern, output, re.DOTALL)
        
        if matches:
            # Return the last message
            return matches[-1].strip()
        
        # Check for common reasoning patterns in gpt-oss models
        # These models sometimes output reasoning without proper channel markers
        if "User says:" in output or "We can comply" in output or "There's no disallowed content" in output:
            # This looks like leaked internal reasoning
            # Try to extract a proper response if there is one
            
            # Look for a response after the reasoning
            lines = output.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines that look like internal reasoning
                if any(pattern in line for pattern in [
                    "User says:", "We need to", "We can comply", 
                    "There's no disallowed", "There's no policy",
                    "So we comply", "It's fine", "The user wants"
                ]):
                    continue
                # Keep lines that look like actual responses
                if line.strip():
                    filtered_lines.append(line)
            
            if filtered_lines:
                return '\n'.join(filtered_lines).strip()
            
            # If everything looks like reasoning, return a generic error
            return "I apologize, but I'm having trouble generating a proper response. Please try rephrasing your request."
        
        # Last resort: remove all special tokens
        cleaned = output
        for token in self.config.custom_filters:
            cleaned = cleaned.replace(token, " ")
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _clean_reasoning_output(self, output: str) -> str:
        """Clean up reasoning output for display."""
        # Replace channel markers with readable labels
        output = re.sub(r'<\|channel\|>analysis<\|message\|>', '\n[Analysis] ', output)
        output = re.sub(r'<\|channel\|>commentary<\|message\|>', '\n[Commentary] ', output)
        output = re.sub(r'<\|channel\|>final<\|message\|>', '\n[Response] ', output)
        
        # Remove other special tokens
        for token in ["<|end|>", "<|start|>", "<|return|>", "<|message|>"]:
            output = output.replace(token, "")
        
        # Clean up role markers
        output = re.sub(r'<\|start\|>assistant', '', output)
        
        return output.strip()
    
    def can_handle(self, model_name: str, tokenizer: Any = None) -> Tuple[bool, float]:
        """Check if this profile can handle the model."""
        model_lower = model_name.lower()
        
        # High confidence for known reasoning models
        if any(name in model_lower for name in ['gpt-oss', 'reasoning', 'cot', 'chain-of-thought']):
            return True, 0.9
        
        # Check tokenizer for reasoning tokens
        if tokenizer:
            try:
                vocab = getattr(tokenizer, 'get_vocab', lambda: {})()
                reasoning_tokens = ['<|channel|>', '<|message|>', '<|start|>', '<|end|>']
                if any(token in vocab for token in reasoning_tokens):
                    return True, 0.85
                
                # Check special tokens map
                special_tokens = getattr(tokenizer, 'special_tokens_map', {})
                special_tokens_str = str(special_tokens)
                if any(token in special_tokens_str for token in reasoning_tokens):
                    return True, 0.85
            except:
                pass
        
        return False, 0.0
    
    def set_show_reasoning(self, show: bool) -> None:
        """Toggle whether to show internal reasoning."""
        self.config.show_reasoning = show
    
    def reset_streaming_state(self):
        """Reset streaming state for new response."""
        self._streaming_state = {
            'in_final': False,
            'buffer': '',
            'final_marker_seen': False
        }
    
    def process_streaming_response(self, token: str, accumulated: str) -> Tuple[str, bool]:
        """Process tokens in streaming mode for reasoning models.
        
        Returns:
            Tuple of (output_token, should_display)
            - output_token: The token to display (may be empty)
            - should_display: Whether this token should be shown to user
        """
        # If showing reasoning, pass through everything with formatting
        if self.config.show_reasoning:
            # Simple pass-through with basic formatting
            return token, True
        
        # Add token to buffer
        self._streaming_state['buffer'] += token
        buffer = self._streaming_state['buffer']
        
        # State machine for filtering
        if not self._streaming_state['in_final']:
            # Look for final channel marker
            if '<|channel|>final<|message|>' in buffer:
                # Found it! Transition to final state
                self._streaming_state['in_final'] = True
                # Clear buffer of everything up to and including the marker
                idx = buffer.index('<|channel|>final<|message|>')
                self._streaming_state['buffer'] = buffer[idx + len('<|channel|>final<|message|>'):]
                # Don't output anything yet
                return '', False
            else:
                # Still accumulating, check if we might be building a marker
                # Keep last 30 chars in buffer to handle partial markers
                if len(buffer) > 30:
                    self._streaming_state['buffer'] = buffer[-30:]
                return '', False
        else:
            # We're in final channel, output everything except end markers
            output = ''
            remaining = ''
            
            # Check for end markers
            if '<|end|>' in buffer:
                # Output everything before the end marker
                idx = buffer.index('<|end|>')
                output = buffer[:idx]
                self._streaming_state['buffer'] = ''
                # Reset for potential next response
                self.reset_streaming_state()
            elif '<|return|>' in buffer:
                # Output everything before the return marker
                idx = buffer.index('<|return|>')
                output = buffer[:idx]
                self._streaming_state['buffer'] = ''
                # Reset for potential next response
                self.reset_streaming_state()
            elif '<|start|>' in buffer:
                # Another message starting, output what we have
                idx = buffer.index('<|start|>')
                output = buffer[:idx]
                self._streaming_state['buffer'] = buffer[idx:]
                self._streaming_state['in_final'] = False
            else:
                # Check if we might be building an end marker
                potential_markers = ['<', '<|', '<|e', '<|en', '<|end', '<|end|',
                                    '<|r', '<|re', '<|ret', '<|retu', '<|retur', '<|return',
                                    '<|s', '<|st', '<|sta', '<|star', '<|start']
                for marker in potential_markers:
                    if buffer.endswith(marker):
                        # Might be building a marker, output everything except potential marker
                        output = buffer[:-len(marker)]
                        self._streaming_state['buffer'] = marker
                        return output, bool(output)
                
                # No potential markers, output everything
                output = buffer
                self._streaming_state['buffer'] = ''
            
            return output, bool(output)