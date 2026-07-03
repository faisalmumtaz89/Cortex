# Template Registry

The Template Registry (`cortex/template_registry/`) detects and manages chat templates for local models, so prompts are formatted the way each model family expects and responses are cleaned before display. Cloud models handle their own formatting and do not use the registry.

## Supported Template Types

| Template | Used by | Format |
|---|---|---|
| **ChatML** | Qwen family and others | `<\|im_start\|>role ... <\|im_end\|>` |
| **Llama** | Llama, Vicuna, TinyLlama | `[INST] ... [/INST]` with `<<SYS>>` blocks |
| **Gemma** | Gemma models | Gemma turn tokens |
| **Alpaca** | Alpaca-style instruction models | `### Instruction:` / `### Response:` sections |
| **Reasoning** | Channel-based reasoning models (gpt-oss series) | `<\|channel\|>analysis/final<\|message\|>` |
| **Simple** | Fallback for unknown models | Plain role-prefixed text |

The Reasoning profile filters internal reasoning channels in real time during streaming, showing only the final channel by default; reasoning display can be toggled per model.

## Automatic Detection

On model load the registry:

1. Checks the saved per-model configuration
2. Otherwise analyzes the model name and tokenizer configuration
3. Scores confidence per profile (`can_handle`) and picks the best match
4. Falls back to Simple when confidence is low
5. Persists the result

## The `/template` Command

Applies to the currently loaded local model:

```bash
/template            # auto-detect and (re)configure the template
/template status     # show detected type, preference, confidence
/template list       # list available templates
/template reset      # clear saved config, re-detect on next setup
```

## Persistence

Per-model configurations are stored in `~/.cortex/template_config.json`, including detected type, user preference, custom filters, `show_reasoning`, and detection confidence.

## Programmatic Usage

```python
from cortex.template_registry import TemplateRegistry

registry = TemplateRegistry()
profile = registry.setup_model("Qwen2-1.5B-Instruct", interactive=False)

prompt = profile.format_messages(
    [{"role": "user", "content": "Hello!"}],
    add_generation_prompt=True,
)
cleaned = profile.process_response(raw_output)
```

Force a specific template or add custom token filters:

```python
registry.configure_template("my-model", template_type="llama", show_reasoning=False)
registry.configure_template("my-model", custom_filters=["<special_token>", "[UNUSED]"])
```

## Adding New Templates

Create a profile in `cortex/template_registry/template_profiles/` implementing `BaseTemplateProfile`:

- `format_messages()` — format the conversation for model input
- `process_response()` — clean model output
- `can_handle()` — return `(matches, confidence)` for a model name/tokenizer

## Troubleshooting

- **Wrong or missing template**: `/template reset`, then `/template` to re-detect.
- **Reasoning tokens visible**: confirm the template type is Reasoning and `show_reasoning` is false in `/template status`.

## Related Documentation

- [CLI Commands](cli.md)
- [Inference Engine](inference-engine.md)
