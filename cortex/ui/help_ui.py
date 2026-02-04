"""Help and shortcuts rendering for CLI."""

from __future__ import annotations

from typing import Any


def show_shortcuts(*, terminal_width: int, box: Any) -> None:
    """Show keyboard shortcuts."""
    width = min(terminal_width - 2, 70)

    print()
    box.print_box_header("Keyboard Shortcuts", width)
    box.print_empty_line(width)

    shortcuts = [
        ("Ctrl+C", "Cancel current generation"),
        ("Ctrl+D", "Exit Cortex"),
        ("Tab", "Auto-complete commands"),
        ("/help", "Show all commands"),
        ("?", "Show this help"),
    ]

    for key, desc in shortcuts:
        colored_key = f"\033[93m{key}\033[0m"
        key_width = len(key)
        padding = " " * (12 - key_width)
        line = f"  {colored_key}{padding}{desc}"
        box.print_box_line(line, width)

    box.print_empty_line(width)
    box.print_box_footer(width)


def show_help(*, terminal_width: int, box: Any) -> None:
    """Show available commands."""
    width = min(terminal_width - 2, 70)

    print()
    box.print_box_header("Available Commands", width)
    box.print_empty_line(width)

    commands = [
        ("/help", "Show this help message"),
        ("/status", "Show current setup and GPU info"),
        ("/download", "Download a model from HuggingFace"),
        ("/model", "Manage models (load/delete/info)"),
        ("/finetune", "Fine-tune a model interactively"),
        ("/clear", "Clear conversation history"),
        ("/save", "Save current conversation"),
        ("/template", "Manage chat templates"),
        ("/gpu", "Show GPU status"),
        ("/benchmark", "Run performance benchmark"),
        ("/login", "Login to HuggingFace for gated models"),
        ("/quit", "Exit Cortex"),
    ]

    for cmd, desc in commands:
        colored_cmd = f"\033[93m{cmd}\033[0m"
        cmd_width = len(cmd)
        padding = " " * (12 - cmd_width)
        line = f"  {colored_cmd}{padding}{desc}"
        box.print_box_line(line, width)

    box.print_empty_line(width)
    box.print_box_footer(width)
