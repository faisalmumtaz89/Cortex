"""Markdown rendering helpers with <think> dimming support."""

from typing import Any, List, cast

from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

THINK_START_MARKER = "[[[THINK_START]]]"
THINK_END_MARKER = "[[[THINK_END]]]"
FenceElementBase = cast(Any, Markdown.elements["fence"])


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


class CodeBlockWithLineNumbers(FenceElementBase):  # type: ignore[misc, valid-type]
    """Markdown code block with line numbers."""

    def __rich_console__(self, console: Console, options):
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            line_numbers=True,
            word_wrap=True,
        )
        yield syntax


class CodeBlockSyntax(FenceElementBase):  # type: ignore[misc, valid-type]
    """Markdown code block with syntax highlighting and wrapped lines."""

    def __rich_console__(self, console: Console, options):
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            line_numbers=False,
            word_wrap=True,
        )
        yield syntax


class CodeBlockPlain(FenceElementBase):  # type: ignore[misc, valid-type]
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


class MarkdownSyntaxCode(Markdown):
    """Markdown renderer that keeps syntax highlighting for fenced code blocks."""

    elements = Markdown.elements.copy()
    elements.update({
        "fence": CodeBlockSyntax,
        "code_block": CodeBlockSyntax,
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
        self._markdown: Markdown
        marked = _mark_think_sections(markup)
        if syntax_highlighting:
            self._markdown = (MarkdownWithLineNumbers if use_line_numbers else MarkdownSyntaxCode)(
                marked,
                code_theme=code_theme,
            )
        else:
            self._markdown = (MarkdownPlainCodeWithLineNumbers if use_line_numbers else MarkdownPlainCode)(
                marked
            )

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

            output_chars: List[str] = []
            index = 0
            while index < len(text):
                if text.startswith(start_marker, index):
                    if output_chars:
                        output = "".join(output_chars)
                        yield from emit(output, style)
                        output_chars.clear()
                    in_think = True
                    index += len(start_marker)
                    continue
                if text.startswith(end_marker, index):
                    if output_chars:
                        output = "".join(output_chars)
                        yield from emit(output, style)
                        output_chars.clear()
                    in_think = False
                    index += len(end_marker)
                    continue
                output_chars.append(text[index])
                index += 1

            output = "".join(output_chars)
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
        auto_space: bool = False,
    ) -> None:
        self.renderable = renderable
        self.prefix = prefix
        self.prefix_style = prefix_style
        self.indent = indent if indent is not None else " " * len(prefix)
        self.auto_space = auto_space

    def __rich_console__(self, console: Console, options):
        prefix_width = cell_len(self.prefix)
        indent_width = cell_len(self.indent) if self.indent is not None else prefix_width
        offset = max(prefix_width, indent_width)
        inner_width = max(1, options.max_width - offset)
        inner_options = options.update_width(inner_width)

        yield Segment(self.prefix, self.prefix_style)

        inserted_space = False
        for segment in console.render(self.renderable, inner_options):
            if segment.control:
                yield segment
                continue

            text = segment.text
            style = segment.style

            if self.auto_space and not inserted_space:
                if text:
                    if not text[0].isspace():
                        yield Segment(" ", None)
                    inserted_space = True

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
