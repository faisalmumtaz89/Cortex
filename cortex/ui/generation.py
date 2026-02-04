"""Response generation flow for the CLI."""

from __future__ import annotations

import sys
import time
import logging

from rich.live import Live
from rich.style import Style

from cortex.conversation_manager import MessageRole
from cortex.inference_engine import GenerationRequest
from cortex.tools import protocol as tool_protocol
from cortex.ui import tools as ui_tools
from cortex.ui.markdown_render import ThinkMarkdown, PrefixedRenderable, render_plain_with_think


logger = logging.getLogger(__name__)


def generate_response(*, cli, user_input: str) -> None:
    """Generate and stream response from the model."""
    if not cli.model_manager.current_model:
        print(
            "\n\033[31m✗\033[0m No model loaded. Use \033[93m/model\033[0m to load a model or "
            "\033[93m/download\033[0m to download one."
        )
        return

    model_name = cli.model_manager.current_model
    tokenizer = cli.model_manager.tokenizers.get(model_name)

    template_profile = None
    uses_reasoning_template = False
    try:
        template_profile = cli.template_registry.setup_model(
            model_name,
            tokenizer=tokenizer,
            interactive=False,
        )
        if template_profile and hasattr(template_profile.config, "template_type"):
            from cortex.template_registry.template_profiles.base import TemplateType

            uses_reasoning_template = template_profile.config.template_type == TemplateType.REASONING
    except Exception as e:
        logger.debug(f"Failed to get template profile: {e}")

    cli._ensure_tool_instructions()
    cli.conversation_manager.add_message(MessageRole.USER, user_input)

    print()

    stop_sequences = []
    if template_profile and hasattr(template_profile, "get_stop_sequences"):
        try:
            stop_sequences = template_profile.get_stop_sequences()
            logger.debug(f"Using stop sequences from template: {stop_sequences}")
        except Exception as e:
            logger.debug(f"Could not get stop sequences: {e}")

    cli.generating = True

    try:
        tool_iterations = 0
        while tool_iterations < cli.max_tool_iterations:
            tool_iterations += 1

            formatted_prompt = cli._format_prompt_with_chat_template(user_input, include_user=False)

            request = GenerationRequest(
                prompt=formatted_prompt,
                max_tokens=cli.config.inference.max_tokens,
                temperature=cli.config.inference.temperature,
                top_p=cli.config.inference.top_p,
                top_k=cli.config.inference.top_k,
                repetition_penalty=cli.config.inference.repetition_penalty,
                stream=cli.config.inference.stream_output,
                seed=cli.config.inference.seed if cli.config.inference.seed >= 0 else None,
                stop_sequences=stop_sequences,
            )

            generated_text = ""
            start_time = time.time()
            token_count = 0
            first_token_time = None
            tool_calls_started = False

            if uses_reasoning_template and template_profile and template_profile.supports_streaming():
                if hasattr(template_profile, "reset_streaming_state"):
                    template_profile.reset_streaming_state()

            display_text = ""
            accumulated_response = ""
            last_render_time = 0.0
            render_interval = 0.05
            prefix_style = Style(color="cyan")

            def build_renderable(text: str):
                if getattr(cli.config.ui, "markdown_rendering", True):
                    markdown = ThinkMarkdown(
                        text,
                        code_theme="monokai",
                        use_line_numbers=False,
                        syntax_highlighting=getattr(cli.config.ui, "syntax_highlighting", True),
                    )
                    renderable = markdown
                else:
                    renderable = render_plain_with_think(text)

                return PrefixedRenderable(renderable, prefix="⏺", prefix_style=prefix_style, indent="  ", auto_space=True)

            original_console_width = cli.console._width
            target_width = max(40, int(cli.get_terminal_width() * 0.75))
            cli.console.width = target_width
            try:
                with Live(
                    build_renderable(""),
                    console=cli.console,
                    auto_refresh=False,
                    refresh_per_second=20,
                    transient=False,
                    vertical_overflow="visible",
                ) as live:
                    for token in cli.inference_engine.generate(request):
                        if first_token_time is None:
                            first_token_time = time.time()

                        generated_text += token
                        token_count += 1

                        if not tool_calls_started and tool_protocol.find_tool_calls_block(generated_text)[0] is not None:
                            tool_calls_started = True
                            display_text = "<think>tools running...</think>"
                            live.update(build_renderable(display_text), refresh=True)

                        display_token = token
                        if uses_reasoning_template and template_profile and template_profile.supports_streaming():
                            display_token, should_display = template_profile.process_streaming_response(
                                token, accumulated_response
                            )
                            accumulated_response += token
                            if not should_display:
                                display_token = ""

                        if not tool_calls_started and display_token:
                            display_text += display_token

                        now = time.time()
                        if (
                            not tool_calls_started
                            and display_token
                            and ("\n" in display_token or now - last_render_time >= render_interval)
                        ):
                            live.update(build_renderable(display_text), refresh=True)
                            last_render_time = now

                    if not tool_calls_started and uses_reasoning_template and template_profile:
                        final_text = template_profile.process_response(generated_text)
                        generated_text = final_text
                        if not template_profile.config.show_reasoning:
                            display_text = final_text
                        live.update(build_renderable(display_text), refresh=True)
            finally:
                cli.console._width = original_console_width

            action, final_text = ui_tools.handle_tool_calls(
                cli=cli,
                generated_text=generated_text,
                tool_calls_started=tool_calls_started,
                tool_iteration=tool_iterations,
                render_final=lambda text: cli.console.print(build_renderable(text)),
            )

            if action is ui_tools.ToolLoopAction.CONTINUE:
                continue
            if action is ui_tools.ToolLoopAction.STOP:
                break

            elapsed = time.time() - start_time
            if token_count > 0 and elapsed > 0:
                tokens_per_sec = token_count / elapsed
                first_token_latency = first_token_time - start_time if first_token_time else 0

                metrics_parts = []
                if first_token_latency > 0.1:
                    metrics_parts.append(f"first {first_token_latency:.2f}s")
                metrics_parts.append(f"total {elapsed:.1f}s")
                metrics_parts.append(f"tokens {token_count}")
                metrics_parts.append(f"speed {tokens_per_sec:.1f} tok/s")
                metrics_line = " · ".join(metrics_parts)
                print(f"  \033[2m{metrics_line}\033[0m")

            if token_count >= request.max_tokens:
                print(f"  \033[2m(output truncated at max_tokens={request.max_tokens}; increase in config.yaml)\033[0m")

            cli.conversation_manager.add_message(MessageRole.ASSISTANT, final_text)
            break

    except Exception as e:
        print(f"\n\033[31m✗ Error:\033[0m {str(e)}", file=sys.stderr)

    finally:
        cli.generating = False
