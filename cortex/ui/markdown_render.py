"""Markdown rendering helpers with <think> dimming support."""

from typing import List

from rich.console import Console
from rich.cells import cell_len
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

THINK_START_MARKER = "[[[THINK_START]]]"
THINK_END_MARKER = "[[[THINK_END]]]"


def _mark_think_sections(text: str) -> str:
    """Replace <think> tags with sentinel markers outside fenced code blocks."""
    lines = text.splitlines(keepends=True)
    in_code_block = False
    output: List[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            output.append(line)
            continue

        if in_code_block:
            output.append(line)
            continue

        output.append(
            line.replace("<think>", THINK_START_MARKER).replace("</think>", THINK_END_MARKER)
        )

    return "".join(output)


class CodeBlockWithLineNumbers(Markdown.elements["fence"]):
    """Markdown code block with line numbers."""

    def __rich_console__(self, console: Console, options):
        code = str(self.text).rstrip()
        syntax = Syntax(code, self.lexer_name, theme=self.theme, line_numbers=True)
        yield syntax


class CodeBlockPlain(Markdown.elements["fence"]):
    """Markdown code block rendered as plain text (no syntax highlighting)."""

    def __rich_console__(self, console: Console, options):
        code = str(self.text).rstrip()
        yield Text(code)


class MarkdownWithLineNumbers(Markdown):
    """Markdown renderer that keeps line numbers for fenced code blocks."""

    elements = Markdown.elements.copy()
    elements.update({
        "fence": CodeBlockWithLineNumbers,
        "code_block": CodeBlockWithLineNumbers,
    })


class MarkdownPlainCode(Markdown):
    """Markdown renderer that disables syntax highlighting for code blocks."""

    elements = Markdown.elements.copy()
    elements.update({
        "fence": CodeBlockPlain,
        "code_block": CodeBlockPlain,
    })


class MarkdownPlainCodeWithLineNumbers(Markdown):
    """Markdown renderer with plain code blocks and line numbers."""

    elements = MarkdownWithLineNumbers.elements.copy()
    elements.update({
        "fence": CodeBlockPlain,
        "code_block": CodeBlockPlain,
    })


class ThinkMarkdown:
    """Markdown renderer that dims content inside <think> tags."""

    def __init__(
        self,
        markup: str,
        code_theme: str = "monokai",
        use_line_numbers: bool = False,
        syntax_highlighting: bool = True,
    ) -> None:
        marked = _mark_think_sections(markup)
        if syntax_highlighting:
            markdown_cls = MarkdownWithLineNumbers if use_line_numbers else Markdown
            self._markdown = markdown_cls(marked, code_theme=code_theme)
        else:
            markdown_cls = MarkdownPlainCodeWithLineNumbers if use_line_numbers else MarkdownPlainCode
            self._markdown = markdown_cls(marked)

    def __rich_console__(self, console: Console, options):
        segments = console.render(self._markdown, options)
        start_marker = THINK_START_MARKER
        end_marker = THINK_END_MARKER
        markers = (start_marker, end_marker)
        in_think = False
        carry = ""
        carry_style = None

        def pending_suffix(text: str) -> tuple[str, str]:
            max_prefix = 0
            for marker in markers:
                for i in range(1, len(marker)):
                    if text.endswith(marker[:i]) and i > max_prefix:
                        max_prefix = i
            if max_prefix:
                return text[:-max_prefix], text[-max_prefix:]
            return text, ""

        def emit(text: str, style: Style | None):
            if not text:
                return
            if in_think:
                style = (style or Style()) + Style(dim=True)
            yield Segment(text, style)

        for segment in segments:
            if segment.control:
                if carry:
                    yield from emit(carry, carry_style)
                    carry = ""
                    carry_style = None
                yield segment
                continue

            text = segment.text
            style = segment.style

            if carry:
                if carry_style != style:
                    yield from emit(carry, carry_style)
                    carry = ""
                    carry_style = None
                else:
                    text = carry + text
                    carry = ""
                    carry_style = None

            output = ""
            index = 0
            while index < len(text):
                if text.startswith(start_marker, index):
                    if output:
                        yield from emit(output, style)
                        output = ""
                    in_think = True
                    index += len(start_marker)
                    continue
                if text.startswith(end_marker, index):
                    if output:
                        yield from emit(output, style)
                        output = ""
                    in_think = False
                    index += len(end_marker)
                    continue
                output += text[index]
                index += 1

            output, carry = pending_suffix(output)
            if output:
                yield from emit(output, style)
            if carry:
                carry_style = style

        if carry:
            yield from emit(carry, carry_style)


class PrefixedRenderable:
    """Render a prefix before the first line and an indent after newlines."""

    def __init__(
        self,
        renderable,
        prefix: str,
        prefix_style: Style | None = None,
        indent: str | None = None,
    ) -> None:
        self.renderable = renderable
        self.prefix = prefix
        self.prefix_style = prefix_style
        self.indent = indent if indent is not None else " " * len(prefix)

    def __rich_console__(self, console: Console, options):
        prefix_width = cell_len(self.prefix)
        indent_width = cell_len(self.indent) if self.indent is not None else prefix_width
        offset = max(prefix_width, indent_width)
        inner_width = max(1, options.max_width - offset)
        inner_options = options.update_width(inner_width)

        yield Segment(self.prefix, self.prefix_style)

        for segment in console.render(self.renderable, inner_options):
            if segment.control:
                yield segment
                continue

            text = segment.text
            style = segment.style

            if "\n" not in text:
                yield segment
                continue

            parts = text.split("\n")
            for index, part in enumerate(parts):
                if part:
                    yield Segment(part, style)
                if index < len(parts) - 1:
                    yield Segment("\n", style)
                    yield Segment(self.indent, None)


def render_plain_with_think(text: str) -> Text:
    """Render plain text while dimming content inside <think> tags."""
    output = Text()
    dim_style = Style(dim=True)
    idx = 0
    in_think = False

    while idx < len(text):
        if text.startswith("<think>", idx):
            in_think = True
            idx += len("<think>")
            continue
        if text.startswith("</think>", idx):
            in_think = False
            idx += len("</think>")
            continue

        char = text[idx]
        output.append(char, dim_style if in_think else None)
        idx += 1

    return output
