# Template Registry System

The Template Registry is an intelligent system in Cortex that automatically detects and manages chat templates for different model architectures, ensuring optimal formatting and response handling.

## Overview

Different language models use different chat template formats for structuring conversations. The Template Registry:
- **Automatically detects** the appropriate template based on model name and tokenizer configuration
- **Manages prompt formatting** to ensure models receive properly structured input
- **Handles response processing** to filter internal reasoning tokens when needed
- **Persists configurations** for consistent behavior across sessions

## Supported Template Types

### 1. ChatML Template
**Used by**: Qwen models (Qwen2, Qwen3, etc.)
**Format**: Uses `<|im_start|>` and `<|im_end|>` tokens
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

### 2. Llama Template
**Used by**: Llama models, Vicuna, TinyLlama
**Format**: Uses `[INST]` and `[/INST]` tokens
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST]
```

### 3. Gemma Template
**Used by**: Gemma models
**Format**: Uses Gemma chat template tokens

### 4. Alpaca Template
**Used by**: Alpaca models, instruction-tuned variants
**Format**: Uses structured instruction format
```
### System:
You are a helpful assistant.

### Instruction:
Hello!

### Response:
```

### 5. Reasoning Template
**Used by**: Models with internal reasoning (gpt-oss series)
**Format**: Handles channel-based reasoning outputs
```
<|channel|>analysis<|message|>Internal reasoning...<|end|>
<|channel|>final<|message|>User-facing response<|end|>
```
**Special Features**: 
- Automatically filters internal reasoning channels in real-time
- Streaming token processing with stateful filtering
- Shows only the final response to users by default
- Optional reasoning display toggle for debugging

#### Streaming State Management

The Reasoning template includes advanced streaming capabilities:
- **Real-time Filtering**: Filters out reasoning tokens as they're generated
- **Stateful Processing**: Maintains streaming state across token generation
- **Low Latency**: Streams final channel content to users as it arrives
- **Automatic Reset**: Streaming state resets for each new response

**Key Methods**:
- `reset_streaming_state()`: Resets internal state for new generation
- `process_streaming_response(token, accumulated)`: Processes tokens in real-time
- `supports_streaming()`: Returns True for reasoning templates

### 6. Simple Template
**Used by**: Default fallback for unknown models
**Format**: Basic conversation format without special tokens

## Automatic Detection

The Template Registry uses multiple strategies to detect the appropriate template:

1. **Model Name Analysis**: Recognizes patterns like "qwen", "llama", "alpaca" in model names
2. **Tokenizer Configuration**: Checks for special tokens and chat templates
3. **Confidence Scoring**: Assigns confidence levels to ensure the best match
4. **Fallback Strategy**: Uses Simple template when detection confidence is low

## Interactive Configuration

### Using the `/template` Command

Configure or adjust templates interactively:

```bash
> /template
```

This command allows you to:
- View current template configuration
- Toggle reasoning display (for reasoning models)
- Manually select a different template
- Reset to auto-detected defaults

### Template Status

View template information with `/status`:
```bash
> /status

Model: Qwen2-1.5B-Instruct
Template: ChatML
```

## Configuration Persistence

Template configurations are automatically saved in:
```
~/.cortex/template_config.json
```

### Configuration Structure

```json
{
  "model_configs": {
    "Qwen2-1.5B-Instruct": {
      "detected_type": "chatml",
      "user_preference": "auto",
      "custom_filters": [],
      "show_reasoning": false,
      "confidence": 0.95,
      "last_updated": "2024-01-15T10:30:00"
    }
  },
  "global_settings": {
    "auto_detect": true,
    "prompt_on_unknown": true,
    "cache_templates": true,
    "default_fallback": "simple"
  }
}
```

## Programmatic Usage

### Python API

```python
from cortex.template_registry import TemplateRegistry

# Initialize registry
registry = TemplateRegistry()

# Setup model (auto-detect template)
profile = registry.setup_model("Qwen2-1.5B-Instruct", interactive=False)

# Format messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
formatted_prompt = profile.format_messages(messages, add_generation_prompt=True)

# Process model response
raw_output = model.generate(formatted_prompt)
cleaned_response = profile.process_response(raw_output)
```

### Manual Template Selection

```python
# Force specific template
profile = registry.configure_template(
    "my-model",
    template_type="llama",
    show_reasoning=False
)
```

## Special Features

### Reasoning Model Handling

For models that output internal reasoning (like gpt-oss series):

1. **Automatic Streaming Filter**: Internal reasoning is filtered in real-time during generation
2. **Stateful Processing**: Maintains streaming state to track channel transitions
3. **Low Latency Output**: Only `<|channel|>final<|message|>` content streams to users
4. **Toggle Display**: Use `/template` to show/hide reasoning for debugging
5. **Channel Detection**: Recognizes `<|channel|>analysis`, `<|channel|>thinking`, etc.
6. **Clean Output**: Seamless user experience without internal analysis tokens

#### Streaming Integration

```python
# For reasoning models with streaming support
if template_profile.supports_streaming():
    template_profile.reset_streaming_state()
    
    for token in generate_tokens():
        display_token, should_display = template_profile.process_streaming_response(
            token, accumulated_response
        )
        if should_display:
            # Stream filtered token to user
            print(display_token, end='', flush=True)
```

### Custom Filters

Add custom token filters for specific models:

```python
registry.configure_template(
    "my-model",
    custom_filters=["<special_token>", "[UNUSED]"]
)
```

## Troubleshooting

### Template Not Detected

If auto-detection fails:
1. Use `/template` to manually select the appropriate template
2. Check if the model name follows standard conventions
3. Verify tokenizer configuration is accessible

### Wrong Template Applied

1. Reset to defaults: `/template reset`
2. Manually configure: `/template`
3. Check confidence score in logs

### Reasoning Tokens Visible

For reasoning models showing internal thoughts:
1. Verify template type is "Reasoning"
2. Check `show_reasoning` is set to `false`
3. Use `/template` to toggle setting

## Best Practices

1. **Let Auto-Detection Work**: The system accurately detects templates for most models
2. **Interactive Setup for Unknown Models**: Use `/template` for first-time model setup
3. **Persistent Configurations**: Settings are saved automatically
4. **Check Template on Load**: Model load messages show the detected template

## Technical Details

### Template Detection Flow

```
Model Load
    ↓
Check Saved Config → Found → Load Template
    ↓ Not Found
Auto-Detect
    ↓
Analyze Model Name
    ↓
Check Tokenizer Config
    ↓
Score Confidence
    ↓
Select Best Match → Save Config → Apply Template
```

### Adding New Templates

To add support for new template formats, create a new profile in:
```
cortex/template_registry/template_profiles/
```

Implement the `BaseTemplateProfile` interface:
- `format_messages()`: Format conversation for model input
- `process_response()`: Clean model output
- `can_handle()`: Detect if template applies to model

## Related Documentation

- [CLI Commands](cli.md) - Template-related commands
- [Configuration](configuration.md) - Template system settings
- [Inference Engine](inference-engine.md) - How templates integrate with generation
