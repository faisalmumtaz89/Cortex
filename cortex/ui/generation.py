"""Response generation flow for the CLI."""

from __future__ import annotations

import logging
import sys
import time
from typing import Dict, List

from cortex.conversation_manager import MessageRole
from cortex.runtime_io import bound_redirected_stdio_files
from cortex.tooling.types import ErrorEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent

logger = logging.getLogger(__name__)


def _looks_like_repo_inspection_request(text: str) -> bool:
    content = (text or "").strip().lower()
    if not content:
        return False

    keywords = (
        "codebase",
        "repository",
        "repo",
        "script",
        "file",
        "function",
        "class",
        "where is",
        "find",
        "inspect",
        "search",
        "grep",
        "input field",
    )
    return any(keyword in content for keyword in keywords)


def _coerce_stream_delta(raw_chunk: str, assembled_text: str) -> tuple[str, str]:
    """Backward-compatible wrapper for stream-delta normalization."""
    from cortex.tooling.stream_normalizer import merge_stream_text

    return merge_stream_text(raw_chunk, assembled_text)


def _build_cloud_messages(*, cli) -> List[Dict[str, str]]:
    """Build cloud-provider-ready messages from conversation history."""
    conversation = cli.conversation_manager.get_current_conversation()
    if conversation is None:
        return []

    all_messages = conversation.messages
    if not all_messages:
        return []

    window = 30
    messages: List[Dict[str, str]] = []
    for message in all_messages[-window:]:
        role = message.role.value
        content = message.content.strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _console_print(cli, text: str = "", *, end: str = "\n") -> None:
    """Print via attached console when available, else stdout."""
    console = getattr(cli, "console", None)
    if console is not None and hasattr(console, "print"):
        try:
            if text:
                console.print(text, end=end)
            else:
                console.print(end=end)
            return
        except Exception:
            pass

    if text:
        print(text, end=end)
    else:
        print(end=end)


def generate_response(*, cli, user_input: str) -> None:
    """Generate and stream response from the model."""
    bound_redirected_stdio_files()

    active_target = cli.active_model_target
    use_cloud_backend = active_target.backend == "cloud"
    cloud_model_ref = active_target.cloud_model
    logger.info(
        "Generation start backend=%s model=%s",
        active_target.backend,
        cloud_model_ref.selector if cloud_model_ref else cli.model_manager.current_model,
    )

    if use_cloud_backend:
        if not getattr(cli.config.cloud, "cloud_enabled", True):
            _console_print(cli, "\n✗ Cloud features are disabled in config.yaml (cloud_enabled=false).")
            return
        if cloud_model_ref is None:
            _console_print(cli, "\n✗ No cloud model selected. Use /model.")
            return
        is_auth, _ = cli.cloud_router.get_auth_status(cloud_model_ref.provider)
        if not is_auth:
            _console_print(
                cli,
                f"\n✗ Missing API key for {cloud_model_ref.provider.value}. "
                f"Run /login {cloud_model_ref.provider.value}.",
            )
            return
    else:
        if not cli.model_manager.current_model:
            _console_print(
                cli,
                "\n✗ No model loaded. Use /model to load a model or /download to download one.",
            )
            return

    model_name = cli.model_manager.current_model if not use_cloud_backend else None
    tokenizer = cli.model_manager.tokenizers.get(model_name) if model_name else None

    template_profile = None
    uses_reasoning_template = False
    if not use_cloud_backend and model_name:
        try:
            template_profile = cli.template_registry.setup_model(
                model_name,
                tokenizer=tokenizer,
                interactive=False,
            )
            if template_profile and hasattr(template_profile.config, "template_type"):
                from cortex.template_registry.template_profiles.base import TemplateType

                uses_reasoning_template = template_profile.config.template_type == TemplateType.REASONING
        except Exception as exc:
            logger.debug("Failed to get template profile: %s", exc)

    stop_sequences = []
    if template_profile and hasattr(template_profile, "get_stop_sequences"):
        try:
            stop_sequences = template_profile.get_stop_sequences()
        except Exception as exc:
            logger.debug("Could not get stop sequences: %s", exc)

    cli.conversation_manager.add_message(MessageRole.USER, user_input)
    _console_print(cli)

    cli.generating = True

    displayed_text = ""
    accumulated_response = ""
    has_visible_output = False
    first_token_time: float | None = None
    last_was_newline = False
    thinking_visible = True
    suppress_stream_text = bool(
        uses_reasoning_template
        and template_profile is not None
        and not getattr(template_profile.config, "show_reasoning", True)
    )
    started_at = time.time()
    final_text = ""
    tools_cfg = getattr(cli.config, "tools", None)
    restore_tools_state: tuple[bool, str] | None = None
    previous_modal_start = getattr(cli, "on_modal_prompt_start", None)
    previous_modal_end = getattr(cli, "on_modal_prompt_end", None)

    pending_status_base = "Cortex is thinking"
    pending_frames = ("..", "...", "....")
    pending_frame_idx = 0
    pending_last_line = ""
    pending_line_visible = False

    def _clear_pending_line() -> None:
        nonlocal pending_line_visible, pending_last_line
        if not pending_line_visible:
            return
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        pending_line_visible = False
        pending_last_line = ""

    def _render_pending_status(
        *,
        advance: bool = False,
        force: bool = False,
        base: str | None = None,
    ) -> None:
        nonlocal pending_frame_idx, pending_last_line, pending_status_base, pending_line_visible
        if not thinking_visible:
            return
        if base is not None:
            pending_status_base = base
        if getattr(cli, "ui_modal_prompt_active", False):
            _clear_pending_line()
            return
        if advance:
            pending_frame_idx = (pending_frame_idx + 1) % len(pending_frames)
        line = f"{pending_status_base}{pending_frames[pending_frame_idx]}"
        if not force and line == pending_last_line:
            return
        pending_last_line = line
        sys.stdout.write(f"\r\033[2K\033[2m{line}\033[0m")
        sys.stdout.flush()
        pending_line_visible = True

    def _start_visible_output() -> None:
        nonlocal has_visible_output, last_was_newline, thinking_visible
        if has_visible_output:
            return
        if thinking_visible:
            _clear_pending_line()
            thinking_visible = False
            last_was_newline = False
        sys.stdout.write("\033[96m⏺\033[0m ")
        sys.stdout.flush()
        has_visible_output = True

    def _emit_text(text: str) -> None:
        nonlocal last_was_newline
        if not text:
            return
        _start_visible_output()
        sys.stdout.write(text)
        sys.stdout.flush()
        last_was_newline = text.endswith("\n")

    def _pause_for_modal() -> None:
        _clear_pending_line()

    def _resume_after_modal() -> None:
        if not has_visible_output and thinking_visible:
            _render_pending_status(force=True)

    try:
        if use_cloud_backend and tools_cfg is not None and _looks_like_repo_inspection_request(user_input):
            current_enabled = bool(getattr(tools_cfg, "tools_enabled", False))
            current_profile = str(getattr(tools_cfg, "tools_profile", "off") or "off")
            if (not current_enabled) or current_profile == "off":
                restore_tools_state = (current_enabled, current_profile)
                tools_cfg.tools_enabled = True
                tools_cfg.tools_profile = "read_only"
                logger.info("Auto-enabled read-only tools for one turn (cloud repo inspection request).")

        _render_pending_status(force=True)
        last_was_newline = False

        def on_event(event):
            nonlocal displayed_text, accumulated_response, first_token_time

            if isinstance(event, TextDeltaEvent):
                delta = event.delta or ""
                if not delta:
                    return
                if first_token_time is None:
                    first_token_time = time.time()

                display_delta = delta
                if uses_reasoning_template and template_profile and template_profile.supports_streaming():
                    display_delta, should_display = template_profile.process_streaming_response(
                        delta, accumulated_response
                    )
                    accumulated_response += delta
                    if not should_display:
                        display_delta = ""

                if suppress_stream_text:
                    return

                if display_delta:
                    displayed_text += display_delta
                    _emit_text(display_delta)
                return

            if isinstance(event, ToolCallEvent):
                if not has_visible_output:
                    tool_name = event.call.name.strip() or "tool"
                    _render_pending_status(force=True, base=f"Running tool: {tool_name}")
                return

            if isinstance(event, ToolResultEvent):
                if not has_visible_output:
                    tool_name = event.result.name.strip() or "tool"
                    base = f"Tool completed: {tool_name}" if event.result.ok else f"Tool failed: {tool_name}"
                    _render_pending_status(force=True, base=base)
                return

            if isinstance(event, ErrorEvent):
                if not has_visible_output:
                    _render_pending_status(force=True, base="Processing response")

        def on_wait(waited_seconds: int, attempt_num: int, total_attempts: int) -> None:
            if has_visible_output:
                return
            if getattr(cli, "ui_modal_prompt_active", False):
                return
            if waited_seconds <= 0:
                return
            _render_pending_status(advance=True)

        def on_retry(attempt_num: int, total_attempts: int, reason: str) -> None:
            if has_visible_output:
                return
            if getattr(cli, "ui_modal_prompt_active", False):
                return
            _render_pending_status(advance=True, force=True)

        cli.on_modal_prompt_start = _pause_for_modal
        cli.on_modal_prompt_end = _resume_after_modal

        conversation = cli.conversation_manager.get_current_conversation()
        turn_result = cli.tooling_orchestrator.run_turn(
            user_input=user_input,
            active_target=active_target,
            conversation=conversation,
            stop_sequences=stop_sequences,
            on_event=on_event,
            on_wait=on_wait,
            on_retry=on_retry,
        )

        final_text = turn_result.text or displayed_text
        if uses_reasoning_template and template_profile:
            final_text = template_profile.process_response(final_text)

        if suppress_stream_text and final_text:
            displayed_text += final_text
            _emit_text(final_text)

        if thinking_visible:
            _clear_pending_line()
            thinking_visible = False

        if has_visible_output:
            if not last_was_newline:
                sys.stdout.write("\n")
            # Keep one blank line between streamed answer and metrics.
            sys.stdout.write("\n")
            sys.stdout.flush()

        elapsed = time.time() - started_at
        token_count = max(0, int(turn_result.token_count))
        if final_text:
            estimated_tokens = max(1, len(final_text) // 4)
            if use_cloud_backend:
                token_count = max(token_count, estimated_tokens)
            elif token_count <= 1:
                token_count = max(token_count, estimated_tokens)

        if token_count > 0 and elapsed > 0:
            tokens_per_sec = token_count / elapsed
            first_latency = (first_token_time - started_at) if first_token_time else 0

            metrics_parts = []
            if first_latency > 0.1:
                metrics_parts.append(f"first {first_latency:.2f}s")
            metrics_parts.append(f"total {elapsed:.1f}s")
            if use_cloud_backend:
                metrics_parts.append(f"tokens~ {token_count}")
                metrics_parts.append(f"speed~ {tokens_per_sec:.1f} tok/s")
            else:
                metrics_parts.append(f"tokens {token_count}")
                metrics_parts.append(f"speed {tokens_per_sec:.1f} tok/s")
            _console_print(cli, f"  {' · '.join(metrics_parts)}")
            logger.info(
                "Generation complete backend=%s tokens=%s elapsed=%.2fs first_token=%.2fs",
                active_target.backend,
                token_count,
                elapsed,
                first_latency,
            )

        if token_count >= cli.config.inference.max_tokens:
            _console_print(
                cli,
                f"  (output truncated at max_tokens={cli.config.inference.max_tokens}; "
                "increase in config.yaml)",
            )

        cli.conversation_manager.add_message(
            MessageRole.ASSISTANT,
            final_text,
            parts=turn_result.parts,
        )

    except Exception as exc:
        logger.exception("Generation error backend=%s: %s", active_target.backend, exc)
        _console_print(cli, f"\n✗ Error: {str(exc)}")

    finally:
        bound_redirected_stdio_files()
        cli.on_modal_prompt_start = previous_modal_start
        cli.on_modal_prompt_end = previous_modal_end
        if restore_tools_state is not None and tools_cfg is not None:
            tools_cfg.tools_enabled, tools_cfg.tools_profile = restore_tools_state
        cli.generating = False
