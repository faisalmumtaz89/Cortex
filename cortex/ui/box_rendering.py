"""Box rendering utilities for CLI."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


def get_visible_length(text: str) -> int:
    """Get visible length of text, ignoring ANSI escape codes and accounting for wide characters."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    visible_text = ansi_escape.sub("", text)

    display_width = 0
    for char in visible_text:
        width = unicodedata.east_asian_width(char)
        if width in ("W", "F"):
            display_width += 2
        elif width == "A" and char in "●○":
            display_width += 1
        else:
            display_width += 1

    return display_width


def print_box_line(content: str, width: int, align: str = "left") -> None:
    """Print a single line in a box with proper padding."""
    visible_len = get_visible_length(content)
    padding = width - visible_len - 2

    if align == "center":
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"│{' ' * left_pad}{content}{' ' * right_pad}│")
    else:
        print(f"│{content}{' ' * padding}│")


def print_box_header(title: str, width: int) -> None:
    """Print a box header with title."""
    if title:
        title_with_color = f" \033[96m{title}\033[0m "
        visible_len = get_visible_length(title_with_color)
        padding = width - visible_len - 3
        print(f"╭─{title_with_color}" + "─" * padding + "╮")
    else:
        print("╭" + "─" * (width - 2) + "╮")


def print_box_footer(width: int) -> None:
    """Print a box footer."""
    print("╰" + "─" * (width - 2) + "╯")


def print_box_separator(width: int) -> None:
    """Print a separator line inside a box."""
    print("├" + "─" * (width - 2) + "┤")


def print_empty_line(width: int) -> None:
    """Print an empty line inside a box."""
    print("│" + " " * (width - 2) + "│")


def create_box(
    lines: List[str],
    *,
    width: Optional[int],
    terminal_width: int,
) -> str:
    """Create a box with Unicode borders."""
    if width is None:
        width = min(terminal_width - 2, 80)

    top_left = "╭"
    top_right = "╮"
    bottom_left = "╰"
    bottom_right = "╯"
    horizontal = "─"
    vertical = "│"

    inner_width = width - 4

    result = []
    result.append(top_left + horizontal * (width - 2) + top_right)

    for line in lines:
        visible_len = get_visible_length(line)
        padding_needed = inner_width - visible_len
        padded = f" {line}{' ' * padding_needed} "
        result.append(vertical + padded + vertical)

    result.append(bottom_left + horizontal * (width - 2) + bottom_right)

    return "\n".join(result)
