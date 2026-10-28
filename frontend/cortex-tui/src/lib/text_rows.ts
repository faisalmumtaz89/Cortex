// Text cut into screen rows here rather than by the terminal UI: each row
// takes exactly one terminal row, so a panel that counts its rows never
// draws past the screen, a wrapped row can start under its own column, and
// a row breaks only at a space (or inside a word too long for a row), with
// every character kept.
import type { RGBA } from "@opentui/core"

export type PromptTone = "text" | "muted" | "addGutter" | "delGutter" | "delText"
export type PromptSpan = { text: string; tone: PromptTone; color?: RGBA }
/** One screen row; `continued` when it goes on with the line the row above began. */
export type PromptRow = { spans: PromptSpan[]; band?: "add" | "del"; continued?: boolean }

const columnsOf = (ch: string) => (ch.codePointAt(0)! < 0x80 ? 1 : 2)
const width = (spans: { text: string }[]) =>
  spans.reduce((total, span) => total + [...span.text].reduce((sum, ch) => sum + columnsOf(ch), 0), 0)

/** `body` cut into rows of at most `columns`, each after `first` (the first
 * row) or `rest` (the rows below). A row breaks after its last space (with
 * `fill`, only one in its second half, so code rows stay full), else where it
 * is full. A character that may be wide counts as two columns. */
export function cutRows<S extends { text: string }>(
  first: S[],
  rest: S[],
  body: S[],
  columns: number,
  fill = false,
): S[][] {
  const chars: { ch: string; span: number }[] = []
  body.forEach((span, index) => {
    for (const ch of span.text) chars.push({ ch, span: index })
  })
  const rows: S[][] = []
  let start = 0
  do {
    const prefix = rows.length === 0 ? first : rest
    const room = Math.max(1, columns - width(prefix))
    let end = start
    let used = 0
    while (end < chars.length && used + columnsOf(chars[end].ch) <= room) {
      used += columnsOf(chars[end].ch)
      end += 1
    }
    if (end === start && end < chars.length) end += 1
    // A row that ends right before a space breaks there.
    if (end < chars.length && chars[end].ch !== " ") {
      const lowest = fill ? start + Math.floor((end - start) / 2) : start
      for (let index = end - 1; index > lowest; index--) {
        if (chars[index].ch === " ") {
          end = index + 1
          break
        }
      }
    }
    const spans: S[] = [...prefix]
    let lastSpan = -1
    for (const { ch, span } of chars.slice(start, end)) {
      if (span === lastSpan) {
        spans[spans.length - 1] = { ...spans[spans.length - 1], text: spans[spans.length - 1].text + ch }
      } else {
        spans.push({ ...body[span], text: ch })
        lastSpan = span
      }
    }
    rows.push(spans)
    start = end
    // The space a row broke at ends that row, not the next.
    while (start < chars.length && chars[start].ch === " " && !fill) start += 1
  } while (start < chars.length)
  return rows
}

/** `body` as a permission prompt's rows (see `cutRows`, rows kept full). */
export function wrapSpans(
  first: PromptSpan[],
  rest: PromptSpan[],
  body: PromptSpan[],
  columns: number,
  band?: "add" | "del",
): PromptRow[] {
  return cutRows(first, rest, body, columns, true).map((spans, index) => ({
    spans,
    band,
    continued: index > 0,
  }))
}

/** `line` cut into rows of at most `columns`, those after the first indented
 * by `hang` spaces (under a description column, say). */
export function wrapText(line: string, columns: number, hang = 0): string[] {
  const indent = hang > 0 ? [{ text: " ".repeat(hang) }] : []
  return cutRows([], indent, [{ text: line }], columns).map((row) =>
    row.map((span) => span.text).join(""),
  )
}
