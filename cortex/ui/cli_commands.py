"""Slash command parsing and dispatch for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandHandlers:
    show_help: Callable[[], None]
    manage_models: Callable[[str], None]
    download_model: Callable[[str], None]
    clear_conversation: Callable[[], None]
    save_conversation: Callable[[], None]
    show_status: Callable[[], None]
    show_gpu_status: Callable[[], None]
    run_benchmark: Callable[[], None]
    manage_template: Callable[[str], None]
    run_finetune: Callable[[], None]
    hf_login: Callable[[], None]
    show_shortcuts: Callable[[], None]


def handle_command(command: str, handlers: CommandHandlers) -> bool:
    """Handle slash commands. Returns False to exit."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        handlers.show_help()
    elif cmd == "/model":
        handlers.manage_models(args)
    elif cmd == "/download":
        handlers.download_model(args)
    elif cmd == "/clear":
        handlers.clear_conversation()
    elif cmd == "/save":
        handlers.save_conversation()
    elif cmd == "/status":
        handlers.show_status()
    elif cmd == "/gpu":
        handlers.show_gpu_status()
    elif cmd == "/benchmark":
        handlers.run_benchmark()
    elif cmd == "/template":
        handlers.manage_template(args)
    elif cmd == "/finetune":
        handlers.run_finetune()
    elif cmd == "/login":
        handlers.hf_login()
    elif cmd in ["/quit", "/exit"]:
        return False
    elif cmd == "?":
        handlers.show_shortcuts()
    else:
        print(f"\033[31mUnknown command: {cmd}\033[0m")
        print("\033[2mType /help for available commands\033[0m")

    return True
