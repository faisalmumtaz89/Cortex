"""Input box rendering and protected input handling for CLI."""

from __future__ import annotations

import os
import sys
import termios
from typing import Optional

INPUT_BG = "\033[48;5;236m"  # Dark gray background (256-color)
INPUT_FG = "\033[30m"        # Black text
RESET = "\033[0m"
PROMPT_PREFIX = "  > "
BOX_HEIGHT = 3
INPUT_LINE_OFFSET = BOX_HEIGHT // 2


def prompt_input_box(
    *,
    terminal_width: int,
    current_model_path: Optional[str],
    bottom_gutter_lines: int = 3,
) -> str:
    """Render the solid input box, read user input, and clean up the UI."""
    width = terminal_width

    # ANSI codes
    yellow = "\033[93m"
    dim = "\033[2m"
    clear_line = "\033[2K"
    cursor_up = "\033[A"
    cursor_down = "\033[B"

    # Model name line
    current_model = ""
    if current_model_path:
        model_name = os.path.basename(current_model_path)
        current_model = f"{dim}Model:{RESET} {yellow}{model_name}{RESET}"

    _reserve_bottom_gutter(bottom_gutter_lines)

    # Draw the input box with a solid background (no borders)
    print()
    fill_line = f"{INPUT_BG}{' ' * width}{RESET}"
    for _ in range(BOX_HEIGHT):
        print(fill_line)

    # Bottom hint: show current model aligned with box
    if current_model:
        print(f"{current_model}")
    else:
        print()

    # Move cursor to input position inside the box (center line)
    move_up = BOX_HEIGHT - INPUT_LINE_OFFSET + 1
    sys.stdout.write(f"\033[{move_up}A")
    sys.stdout.write(f"\r{INPUT_BG}{INPUT_FG}{PROMPT_PREFIX}")
    sys.stdout.flush()

    try:
        user_input = _get_protected_input(width)

        # Clear the input box region using relative moves.
        sys.stdout.write(f"{cursor_down}\r{clear_line}")
        for _ in range(INPUT_LINE_OFFSET + 3):
            sys.stdout.write(f"{cursor_up}\r{clear_line}")

        # Print the clean prompt that represents the submitted user message.
        sys.stdout.write("\r> " + user_input.strip() + "\n")
        sys.stdout.flush()

        return user_input.strip()

    except EOFError:
        try:
            sys.stdout.write(f"\r{clear_line}")
            for _ in range(BOX_HEIGHT - INPUT_LINE_OFFSET):
                sys.stdout.write(f"{cursor_down}\r{clear_line}")
            for _ in range(BOX_HEIGHT):
                sys.stdout.write(f"{cursor_up}\r{clear_line}")
            sys.stdout.flush()
        finally:
            pass
        raise


def _reserve_bottom_gutter(lines: int) -> None:
    """Reserve a blank gutter at the bottom of the terminal."""
    gutter = max(0, int(lines))
    if gutter <= 0:
        return

    sys.stdout.write("\n" * gutter)
    sys.stdout.write(f"\033[{gutter}A")
    sys.stdout.flush()


def _get_protected_input(box_width: int) -> str:
    """Read input in raw mode and prevent deleting the prompt."""
    # Calculate usable width for text (leave one trailing space to avoid wrap)
    max_display_width = box_width - len(PROMPT_PREFIX) - 1
    clear_line = "\033[2K"
    cursor_down = "\033[1B"
    cursor_up = "\033[1A"

    # Store terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        # Set terminal to raw mode for character-by-character input
        # Disable ISIG so we can handle Ctrl+C/Ctrl+Z/Ctrl+\ manually for clean exit
        new_settings = termios.tcgetattr(sys.stdin)
        new_settings[3] = new_settings[3] & ~termios.ICANON
        new_settings[3] = new_settings[3] & ~termios.ECHO
        new_settings[3] = new_settings[3] & ~termios.ISIG
        new_settings[6][termios.VMIN] = 1
        new_settings[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

        input_buffer: list[str] = []
        cursor_pos = 0
        view_offset = 0

        def redraw_line() -> None:
            nonlocal view_offset

            if len(input_buffer) <= max_display_width:
                display_text = "".join(input_buffer)
                display_cursor_pos = cursor_pos
            else:
                if cursor_pos < view_offset:
                    view_offset = cursor_pos
                elif cursor_pos >= view_offset + max_display_width:
                    view_offset = cursor_pos - max_display_width + 1

                display_text = "".join(input_buffer[view_offset:view_offset + max_display_width])
                display_cursor_pos = cursor_pos - view_offset

            pad_len = box_width - len(PROMPT_PREFIX) - len(display_text)
            if pad_len < 0:
                pad_len = 0

            sys.stdout.write(
                f"\r{INPUT_BG}{INPUT_FG}{PROMPT_PREFIX}{display_text}"
                f"{' ' * pad_len}{RESET}"
            )

            cursor_column = len(PROMPT_PREFIX) + 1 + display_cursor_pos
            sys.stdout.write(f"\033[{cursor_column}G")
            sys.stdout.flush()

        redraw_line()

        def clear_box_from_input():
            sys.stdout.write(f"\r{clear_line}")
            for _ in range(BOX_HEIGHT - INPUT_LINE_OFFSET):
                sys.stdout.write(f"{cursor_down}\r{clear_line}")
            for _ in range(BOX_HEIGHT):
                sys.stdout.write(f"{cursor_up}\r{clear_line}")
            sys.stdout.write("\r")
            sys.stdout.flush()

        try:
            while True:
                char = sys.stdin.read(1)

                if char == "\r" or char == "\n":
                    sys.stdout.write("\r\n")
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    break

                if char == "\x7f" or char == "\x08":
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        input_buffer.pop(cursor_pos)
                        redraw_line()
                    continue

                if char in ("\x03", "\x1a", "\x1c"):
                    raise KeyboardInterrupt

                if char == "\x04":
                    raise EOFError

                if char == "\x1b":
                    next1 = sys.stdin.read(1)
                    if next1 == "[":
                        next2 = sys.stdin.read(1)
                        if next2 == "D":
                            if cursor_pos > 0:
                                cursor_pos -= 1
                                redraw_line()
                        elif next2 == "C":
                            if cursor_pos < len(input_buffer):
                                cursor_pos += 1
                                redraw_line()
                        elif next2 == "H":
                            cursor_pos = 0
                            view_offset = 0
                            redraw_line()
                        elif next2 == "F":
                            cursor_pos = len(input_buffer)
                            redraw_line()
                    continue

                if ord(char) >= 32:
                    input_buffer.insert(cursor_pos, char)
                    cursor_pos += 1
                    redraw_line()
        except KeyboardInterrupt:
            clear_box_from_input()
            raise

        return "".join(input_buffer)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
