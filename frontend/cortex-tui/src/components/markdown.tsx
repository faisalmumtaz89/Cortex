import type { RGBA } from "@opentui/core"
import { For, Show, createMemo } from "solid-js"
import { terminalColumns } from "../lib/terminal_size"
import { highlightLine } from "./syntax_highlight"
import { UI_PALETTE } from "./ui_palette"

// ---- Inline parsing (bold / italic / code / links) ----

type InlineToken =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "italic"; text: string }
  | { kind: "code"; text: string }
  | { kind: "link"; text: string }

const ALNUM = /[A-Za-z0-9]/

export function parseInline(input: string): InlineToken[] {
  const tokens: InlineToken[] = []
  let plain = ""
  let i = 0

  const flush = () => {
    if (plain) {
      tokens.push({ kind: "text", text: plain })
      plain = ""
    }
  }

  while (i < input.length) {
    const rest = input.slice(i)

    // Inline code — highest precedence, no nested parsing.
    if (rest[0] === "`") {
      const end = rest.indexOf("`", 1)
      if (end > 0) {
        flush()
        tokens.push({ kind: "code", text: rest.slice(1, end) })
        i += end + 1
        continue
      }
    }

    // Underscore emphasis (bold __ / italic _) is only valid at word
    // boundaries so snake_case identifiers and file_paths stay literal.
    const prevChar = i > 0 ? input[i - 1] : ""
    const underscoreBoundary = !ALNUM.test(prevChar)

    // Bold **...** or __...__
    if (rest.startsWith("**") || (rest.startsWith("__") && underscoreBoundary)) {
      const marker = rest.slice(0, 2)
      const end = rest.indexOf(marker, 2)
      const afterUnderscore = marker === "__" ? input[i + end + 2] ?? "" : ""
      if (end > 1 && (marker === "**" || !ALNUM.test(afterUnderscore))) {
        flush()
        tokens.push({ kind: "bold", text: rest.slice(2, end) })
        i += end + 2
        continue
      }
    }

    // Italic *...* or _..._
    if ((rest[0] === "*" || (rest[0] === "_" && underscoreBoundary)) && rest[1] !== rest[0]) {
      const marker = rest[0]
      const end = rest.indexOf(marker, 1)
      const afterUnderscore = marker === "_" ? input[i + end + 1] ?? "" : ""
      if (end > 0 && rest[end - 1] !== " " && (marker === "*" || !ALNUM.test(afterUnderscore))) {
        flush()
        tokens.push({ kind: "italic", text: rest.slice(1, end) })
        i += end + 1
        continue
      }
    }

    // Link [text](url) — render just the text, colored.
    if (rest[0] === "[") {
      const close = rest.indexOf("]")
      if (close > 0 && rest[close + 1] === "(") {
        const urlEnd = rest.indexOf(")", close)
        if (urlEnd > close) {
          flush()
          tokens.push({ kind: "link", text: rest.slice(1, close) })
          i += urlEnd + 1
          continue
        }
      }
    }

    plain += rest[0]
    i += 1
  }

  flush()
  return tokens
}

/** Renders inline markdown (bold/italic/code/links) as one text line. An
 * optional prefix (list marker) and baseColor/bold keep list markers and
 * headings in the same text flow so wrapping never drops spacing. */
function InlineText(props: {
  text: string
  prefix?: string
  prefixColor?: RGBA
  baseColor?: RGBA
  bold?: boolean
}) {
  const tokens = createMemo(() => parseInline(props.text))
  const base = () => props.baseColor ?? UI_PALETTE.text
  return (
    <text fg={base()}>
      <Show when={props.prefix}>
        <span style={{ fg: props.prefixColor ?? UI_PALETTE.accent }}>{props.prefix}</span>
      </Show>
      <For each={tokens()}>
        {(token) => (
          <>
            <Show when={token.kind === "text"}>
              <Show when={props.bold} fallback={<span style={{ fg: base() }}>{token.text}</span>}>
                <strong>
                  <span style={{ fg: base() }}>{token.text}</span>
                </strong>
              </Show>
            </Show>
            <Show when={token.kind === "bold"}>
              <strong>
                <span style={{ fg: base() }}>{token.text}</span>
              </strong>
            </Show>
            <Show when={token.kind === "italic"}>
              <em>
                <span style={{ fg: base() }}>{token.text}</span>
              </em>
            </Show>
            <Show when={token.kind === "code"}>
              {/* No pad spaces — they doubled with the surrounding literal
                  spaces; the bg hugs the code text, separation comes from prose. */}
              <span style={{ fg: UI_PALETTE.statusBusy, bg: UI_PALETTE.codeBackground }}>{token.text}</span>
            </Show>
            <Show when={token.kind === "link"}>
              <u>
                <span style={{ fg: UI_PALETTE.heading }}>{token.text}</span>
              </u>
            </Show>
          </>
        )}
      </For>
    </text>
  )
}

// ---- Block parsing ----

type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "bullet"; text: string; indent: number }
  | { kind: "ordered"; marker: string; text: string; indent: number }
  | { kind: "quote"; lines: string[] }
  | { kind: "code"; language?: string; lines: string[] }
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "rule" }

function splitTableRow(line: string): string[] {
  let inner = line.trim()
  if (inner.startsWith("|")) inner = inner.slice(1)
  if (inner.endsWith("|")) inner = inner.slice(0, -1)
  return inner.split("|").map((cell) => cell.trim())
}

const TABLE_ROW = /^\s*\|.*\|\s*$/
const TABLE_SEPARATOR = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/

export function parseBlocks(content: string): Block[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n")
  const blocks: Block[] = []
  let paragraph: string[] = []

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ kind: "paragraph", text: paragraph.join(" ") })
      paragraph = []
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    // Fenced code block
    const fence = /^```+\s*([\w+-]*)\s*$/.exec(trimmed)
    if (fence) {
      flushParagraph()
      const language = fence[1] || undefined
      const codeLines: string[] = []
      i += 1
      while (i < lines.length && !/^```+\s*$/.test(lines[i].trim())) {
        codeLines.push(lines[i])
        i += 1
      }
      blocks.push({ kind: "code", language, lines: codeLines })
      continue
    }

    // GitHub table: a header row followed by a --- separator row.
    if (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_SEPARATOR.test(lines[i + 1])) {
      flushParagraph()
      const header = splitTableRow(line)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && TABLE_ROW.test(lines[i])) {
        rows.push(splitTableRow(lines[i]))
        i += 1
      }
      i -= 1
      blocks.push({ kind: "table", header, rows })
      continue
    }

    if (trimmed === "") {
      flushParagraph()
      continue
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      flushParagraph()
      blocks.push({ kind: "rule" })
      continue
    }

    // Heading
    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed)
    if (heading) {
      flushParagraph()
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2].trim() })
      continue
    }

    // Blockquote — coalesce consecutive "> " lines into one bordered block.
    const quote = /^>\s?(.*)$/.exec(trimmed)
    if (quote) {
      flushParagraph()
      const quoteLines = [quote[1]]
      while (i + 1 < lines.length) {
        const nextQuote = /^>\s?(.*)$/.exec(lines[i + 1].trim())
        if (!nextQuote) break
        quoteLines.push(nextQuote[1])
        i += 1
      }
      blocks.push({ kind: "quote", lines: quoteLines })
      continue
    }

    // Bullet list
    const bullet = /^(\s*)[-*+]\s+(.*)$/.exec(line)
    if (bullet) {
      flushParagraph()
      blocks.push({ kind: "bullet", text: bullet[2], indent: Math.floor(bullet[1].length / 2) })
      continue
    }

    // Ordered list
    const ordered = /^(\s*)(\d+)[.)]\s+(.*)$/.exec(line)
    if (ordered) {
      flushParagraph()
      blocks.push({
        kind: "ordered",
        marker: ordered[2],
        text: ordered[3],
        indent: Math.floor(ordered[1].length / 2),
      })
      continue
    }

    paragraph.push(trimmed)
  }

  flushParagraph()
  return blocks
}

const SPINE_BORDER = {
  topLeft: "",
  bottomLeft: "",
  vertical: "┃",
  topRight: "",
  bottomRight: "",
  horizontal: " ",
  bottomT: "",
  topT: "",
  cross: "",
  leftT: "",
  rightT: "",
}

function CodeBlock(props: { language?: string; lines: string[]; marginTop: number }) {
  // No language caption — the syntax coloring conveys it, and a muted label row
  // reads as a stray "# comment". The ┃ spine matches every other gutter.
  return (
    <box
      flexDirection="column"
      flexShrink={0}
      marginTop={props.marginTop}
      paddingLeft={2}
      paddingRight={2}
      backgroundColor={UI_PALETTE.codeBackground}
      border={["left"]}
      borderColor={UI_PALETTE.accent}
      customBorderChars={SPINE_BORDER}
    >
      <For each={props.lines}>
        {(line) => (
          <text>
            <For each={highlightLine(line, props.language)}>
              {(token) => <span style={{ fg: token.color }}>{token.text}</span>}
            </For>
          </text>
        )}
      </For>
    </box>
  )
}

function TableBlock(props: { header: string[]; rows: string[][]; marginTop: number }) {
  const columnCount = () => Math.max(props.header.length, ...props.rows.map((row) => row.length))
  const widths = () => {
    const cols = columnCount()
    const w = new Array(cols).fill(0)
    const consider = (cells: string[]) => {
      for (let c = 0; c < cols; c++) {
        w[c] = Math.max(w[c], (cells[c] ?? "").length)
      }
    }
    consider(props.header)
    props.rows.forEach(consider)
    return w
  }
  const pad = (text: string, width: number) => text + " ".repeat(Math.max(0, width - text.length))
  const rowText = (cells: string[]) =>
    widths()
      .map((width, c) => pad(cells[c] ?? "", width))
      .join("  │  ")
  const separator = () =>
    widths()
      .map((width) => "─".repeat(width))
      .join("──┼──")

  return (
    // Inter-block spacing is owned by blockMarginTop() (no bottom margin —
    // it would stack with the next block's marginTop into a doubled gap).
    <box flexDirection="column" flexShrink={0} marginTop={props.marginTop}>
      <text attributes={1 /* bold */} fg={UI_PALETTE.heading}>
        {rowText(props.header)}
      </text>
      <text fg={UI_PALETTE.textMuted}>{separator()}</text>
      <For each={props.rows}>{(row) => <text fg={UI_PALETTE.text}>{rowText(row)}</text>}</For>
    </box>
  )
}

const HEADING_COLORS = [
  UI_PALETTE.heading,
  UI_PALETTE.heading,
  UI_PALETTE.accent,
  UI_PALETTE.accent,
  UI_PALETTE.text,
  UI_PALETTE.text,
]

const LIST_KINDS = new Set(["bullet", "ordered"])

/** Horizontal-rule width: spans the content region (capped at the prose width)
 * instead of a fixed stub. Reactive to terminalColumns() so it reflows on resize. */
function RULE_WIDTH(): number {
  return Math.max(8, Math.min(terminalColumns() - 8, 96))
}

/** Adaptive vertical rhythm (opencode-style): 1 blank row between blocks, but
 * consecutive list items in the same list stay tight (0). So a list is set off
 * from surrounding prose by one row, while its items pack together. */
function blockMarginTop(block: Block, previous: Block | undefined): number {
  if (!previous) {
    return 0
  }
  // Consecutive items of the SAME list type stay tight; a bullet↔ordered
  // transition is a new list and gets a blank row like any other block.
  if (LIST_KINDS.has(block.kind) && block.kind === previous.kind) {
    return 0
  }
  return 1
}

export function Markdown(props: { content: string }) {
  // Memoized so the whole answer only re-parses when its text changes, not on
  // every unrelated re-render.
  const blocks = createMemo(() => parseBlocks(props.content))
  return (
    <box flexDirection="column" flexShrink={0}>
      <For each={blocks()}>
        {(block, index) => {
          const mt = () => blockMarginTop(block, blocks()[index() - 1])
          return (
            <>
              <Show when={block.kind === "heading"}>
                {(() => {
                  const b = block as Extract<Block, { kind: "heading" }>
                  return (
                    <box flexShrink={0} marginTop={mt()}>
                      <InlineText text={b.text} baseColor={HEADING_COLORS[b.level - 1]} bold />
                    </box>
                  )
                })()}
              </Show>

              <Show when={block.kind === "paragraph"}>
                <box flexShrink={0} marginTop={mt()}>
                  <InlineText text={(block as Extract<Block, { kind: "paragraph" }>).text} />
                </box>
              </Show>

              <Show when={block.kind === "bullet"}>
                {(() => {
                  const b = block as Extract<Block, { kind: "bullet" }>
                  return (
                    <box flexShrink={0} marginTop={mt()} paddingLeft={b.indent * 2} flexDirection="row">
                      <text fg={UI_PALETTE.accent}>{"• "}</text>
                      <box flexGrow={1} flexShrink={1}>
                        <InlineText text={b.text} />
                      </box>
                    </box>
                  )
                })()}
              </Show>

              <Show when={block.kind === "ordered"}>
                {(() => {
                  const b = block as Extract<Block, { kind: "ordered" }>
                  return (
                    <box flexShrink={0} marginTop={mt()} paddingLeft={b.indent * 2} flexDirection="row">
                      <text fg={UI_PALETTE.heading}>{`${b.marker}. `}</text>
                      <box flexGrow={1} flexShrink={1}>
                        <InlineText text={b.text} />
                      </box>
                    </box>
                  )
                })()}
              </Show>

              <Show when={block.kind === "quote"}>
                <box
                  flexShrink={0}
                  marginTop={mt()}
                  paddingLeft={2}
                  border={["left"]}
                  borderColor={UI_PALETTE.textMuted}
                  customBorderChars={SPINE_BORDER}
                >
                  <For each={(block as Extract<Block, { kind: "quote" }>).lines}>
                    {(line) => <InlineText text={line} baseColor={UI_PALETTE.textMuted} />}
                  </For>
                </box>
              </Show>

              <Show when={block.kind === "rule"}>
                <box flexShrink={0} marginTop={mt()}>
                  <text fg={UI_PALETTE.textMuted}>{"─".repeat(RULE_WIDTH())}</text>
                </box>
              </Show>

              <Show when={block.kind === "code"}>
                <CodeBlock
                  language={(block as Extract<Block, { kind: "code" }>).language}
                  lines={(block as Extract<Block, { kind: "code" }>).lines}
                  marginTop={mt()}
                />
              </Show>

              <Show when={block.kind === "table"}>
                <TableBlock
                  header={(block as Extract<Block, { kind: "table" }>).header}
                  rows={(block as Extract<Block, { kind: "table" }>).rows}
                  marginTop={mt()}
                />
              </Show>
            </>
          )
        }}
      </For>
    </box>
  )
}
