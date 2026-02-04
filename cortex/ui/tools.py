"""Tool orchestration helpers for the CLI."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Tuple
import sys

from cortex.conversation_manager import MessageRole
from cortex.tools import protocol as tool_protocol
from cortex.ui.tool_activity import print_tool_activity


class ToolLoopAction(Enum):
    CONTINUE = "continue"
    FINALIZE = "finalize"
    STOP = "stop"


def handle_tool_calls(
    *,
    cli,
    generated_text: str,
    tool_calls_started: bool,
    tool_iteration: int,
    render_final: Callable[[str], None],
) -> Tuple[ToolLoopAction, str]:
    """Process tool calls and decide whether to continue the tool loop."""
    tool_calls, parse_error = tool_protocol.parse_tool_calls(generated_text)
    if parse_error:
        print(f"\n\033[31m✗ Tool call parse error:\033[0m {parse_error}", file=sys.stderr)

    if tool_calls:
        tool_results = cli.tool_runner.run_calls(tool_calls)
        print_tool_activity(cli.console, tool_calls, tool_results, cli.get_terminal_width())
        cli.conversation_manager.add_message(
            MessageRole.SYSTEM,
            tool_protocol.format_tool_results(tool_results),
        )
        if tool_iteration >= cli.max_tool_iterations:
            print("\n\033[31m✗\033[0m Tool loop limit reached.", file=sys.stderr)
            return ToolLoopAction.STOP, ""
        return ToolLoopAction.CONTINUE, ""

    final_text = generated_text
    if parse_error:
        final_text = tool_protocol.strip_tool_blocks(generated_text)
        if tool_calls_started and final_text.strip():
            render_final(final_text)

    return ToolLoopAction.FINALIZE, final_text
