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

// What the terminal draws as one character: a letter with its accents, an
// emoji sequence.
const graphemes = new Intl.Segmenter(undefined, { granularity: "grapheme" })
// Text whose every character is one column wide on its own: most code and prose.
const PRINTABLE_ASCII = /^[\x20-\x7e]*$/

/** The columns `text` may take on the terminal, never fewer than it does: a
 * printable ASCII character takes one, any other code point (a tab, an
 * accent, a wide or emoji character) is counted as two. A row counted as
 * fitting is then drawn in one row, at worst a little short. */
function textColumns(text: string): number {
  let columns = 0
  for (const ch of text) columns += PRINTABLE_ASCII.test(ch) ? 1 : 2
  return columns
}

const width = (spans: { text: string }[]) => spans.reduce((total, span) => total + textColumns(span.text), 0)

/** `body` cut into rows of at most `columns`, each after `first` (the first
 * row) or `rest` (the rows below). A row breaks after its last space (with
 * `fill`, only one in its second half, so code rows stay full), else where it
 * is full; never inside a character the terminal draws as one. */
export function cutRows<S extends { text: string }>(
  first: S[],
  rest: S[],
  body: S[],
  columns: number,
  fill = false,
): S[][] {
  const chars: { ch: string; span: number; columns: number }[] = []
  body.forEach((span, index) => {
    if (PRINTABLE_ASCII.test(span.text)) {
      for (const ch of span.text) chars.push({ ch, span: index, columns: 1 })
      return
    }
    for (const { segment } of graphemes.segment(span.text)) {
      chars.push({ ch: segment, span: index, columns: textColumns(segment) })
    }
  })
  const rows: S[][] = []
  const rooms = [first, rest].map((prefix) => Math.max(1, columns - width(prefix)))
  let start = 0
  do {
    const prefix = rows.length === 0 ? first : rest
    const room = rooms[rows.length === 0 ? 0 : 1]
    let end = start
    let used = 0
    while (end < chars.length && used + chars[end].columns <= room) {
      used += chars[end].columns
      end += 1
    }
    if (end === start && end < chars.length) end += 1
    // A row that ends right before a space breaks there; one kept full breaks
    // after a space in its second half even then, so the next row does not
    // start with the space.
    if (end < chars.length && (fill || chars[end].ch !== " ")) {
      const lowest = fill ? start + Math.floor((end - start) / 2) : start + 1
      for (let index = end - 1; index >= lowest; index--) {
        if (chars[index].ch === " ") {
          end = index + 1
          break
        }
      }
    }
    const spans: S[] = [...prefix]
    for (let index = start; index < end; ) {
      const span = chars[index].span
      let next = index
      while (next < end && chars[next].span === span) next += 1
      spans.push({ ...body[span], text: chars.slice(index, next).map((char) => char.ch).join("") })
      index = next
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

/** `text`, line by line, cut into rows of at most `columns` (see `wrapText`). */
export function textRows(text: string, columns: number, hang = 0): string[] {
  return text.split("\n").flatMap((line) => (line ? wrapText(line, columns, hang) : [""]))
}

/** `line` cut into rows of at most `columns`, those after the first indented
 * by `hang` spaces (under a description column, say). */
export function wrapText(line: string, columns: number, hang = 0): string[] {
  const indent = hang > 0 ? [{ text: " ".repeat(hang) }] : []
  return cutRows([], indent, [{ text: line }], columns).map((row) =>
    row.map((span) => span.text).join(""),
  )
}
