// What a permission prompt shows, laid out to fit the screen. Each row is cut
// to the prompt's width here, so it takes exactly one terminal row, and the
// call's rows go in a window that leaves the transcript at least half the
// free rows; PgUp/PgDn move the window. The store decides the prompt's
// starting choice from this layout, so Enter never approves a call whose
// text was not on screen.
import type { RGBA } from "@opentui/core"
import { highlightLine } from "../components/syntax_highlight"
import type { DiffData } from "../context/session_store"
import { displayPath } from "./paths"
import { rowsAtLineBreaks, visibleCode, visibleText } from "./visible_text"

export type PermissionPrompt = {
  permission: string
  patterns: string[]
  metadata: Record<string, unknown>
}

function rawSummary(pending: PermissionPrompt): string {
  // Tool metadata is {tool, arguments: {...}} — the details live in arguments.
  const meta = (pending.metadata || {}) as Record<string, unknown>
  const args = (meta.arguments ?? {}) as Record<string, unknown>
  const tool = String(meta.tool ?? "")
  const argPath = displayPath(String(args.path ?? pending.patterns[0] ?? ""))

  if (pending.permission === "read") {
    return `Read file: ${argPath}`
  }
  if (pending.permission === "list") {
    return `List directory: ${argPath}`
  }
  if (pending.permission === "grep") {
    return `Search pattern: ${String(args.query ?? args.pattern ?? "")}`
  }
  if (pending.permission === "bash") {
    // The pattern is the command as approved, with the directory it runs in.
    const pattern = pending.patterns[0]
    return `Run command: ${pattern && pattern !== "*" ? pattern : String(args.command ?? "")}`
  }
  if (pending.permission === "edit") {
    const action = tool === "write_file" ? "Write file" : "Edit file"
    return `${action}: ${argPath}`
  }
  if (pending.permission === "external_directory") {
    return `Access external directory: ${String(args.parentDir ?? args.filepath ?? pending.patterns[0] ?? "")}`
  }
  if (pending.permission === "doom_loop") {
    return "Continue after repeated identical tool calls."
  }
  return `Permission: ${pending.permission}${tool ? ` (${tool})` : ""}`
}

/** The prompt's lines: the summary, and the Target line when it adds
 * something the summary does not show. */
export function permissionLines(pending: PermissionPrompt): string[] {
  const summary = rawSummary(pending)
  // Patterns/paths display repo-relative when inside the repo (display only).
  const target = pending.patterns.map(displayPath).join(", ") || "*"
  const lines = [summary]
  if (pending.permission !== "bash" && !visibleText(summary).includes(visibleText(target))) {
    lines.push(`Target: ${target}`)
  }
  return lines
}

export type PromptTone = "text" | "muted" | "addGutter" | "delGutter" | "delText"
export type PromptSpan = { text: string; tone: PromptTone; color?: RGBA }
export type PromptRow = { spans: PromptSpan[]; band?: "add" | "del" }

export type PromptLayout = {
  /** Always on screen: an edit's file and the size of its change. */
  pinned: PromptRow[]
  /** The call's text: the command, or the edit's diff. */
  rows: PromptRow[]
  /** The first of `rows` shown, and how many are. */
  offset: number
  window: number
  /** Every row is on screen at once, and nothing of the call was left out. */
  full: boolean
}

// Columns the prompt's rows lose: the screen's side padding (2 + 2), the
// prompt's rail and the padding after it (1 + 2). Diff rows indent 2 more.
const TEXT_INDENT = 7
const DIFF_INDENT = 2
// Rows the screen keeps while a prompt is open, besides the transcript and
// the call's rows: its top and bottom padding (2), the header and the row
// after it (2), the row after the transcript (1), the prompt's title, the row
// for the scroll position and the 3 choices (5), the row after the prompt
// (1), and the input panel around one row of text (4).
const FIXED_ROWS = 15
const MIN_BODY_ROWS = 3

const columnsOf = (ch: string) => (ch.codePointAt(0)! < 0x80 ? 1 : 2)
const width = (spans: PromptSpan[]) =>
  spans.reduce((total, span) => total + [...span.text].reduce((sum, ch) => sum + columnsOf(ch), 0), 0)

/** `body` cut into rows of at most `columns`, each after `first` (the first
 * row) or `rest` (the rows below), at a space when the row's second half has
 * one. A character that may be wide counts as two columns. */
function wrap(
  first: PromptSpan[],
  rest: PromptSpan[],
  body: PromptSpan[],
  columns: number,
  band?: "add" | "del",
): PromptRow[] {
  const chars: { ch: string; span: number }[] = []
  body.forEach((span, index) => {
    for (const ch of span.text) chars.push({ ch, span: index })
  })
  const rows: PromptRow[] = []
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
    if (end < chars.length) {
      for (let index = end - 1; index > start + Math.floor((end - start) / 2); index--) {
        if (chars[index].ch === " ") {
          end = index + 1
          break
        }
      }
    }
    const spans: PromptSpan[] = [...prefix]
    for (const { ch, span } of chars.slice(start, end)) {
      const last = spans[spans.length - 1]
      if (last && last.tone === body[span].tone && last.color === body[span].color && spans.length > prefix.length) {
        last.text += ch
      } else {
        spans.push({ ...body[span], text: ch })
      }
    }
    rows.push({ spans, band })
    start = end
  } while (start < chars.length)
  return rows
}

function textRows(line: string, tone: PromptTone, columns: number): PromptRow[] {
  return rowsAtLineBreaks(visibleText(line, Infinity)).flatMap((row) =>
    wrap([], [], [{ text: row, tone }], columns),
  )
}

function diffRows(diff: DiffData, columns: number): PromptRow[] {
  let numbers = 2
  for (const hunk of diff.hunks) {
    for (const row of hunk.rows) {
      numbers = Math.max(numbers, String(row.kind === "del" ? row.oldNo : row.newNo).length)
    }
  }
  const blank: PromptSpan[] = [{ text: " ".repeat(numbers + 3), tone: "muted" }]
  const rows: PromptRow[] = []
  diff.hunks.forEach((hunk, index) => {
    if (index > 0) rows.push({ spans: [{ text: `${" ".repeat(numbers)} ⋯`, tone: "muted" }] })
    for (const row of hunk.rows) {
      const code = visibleCode(row.text.replace(/\t/g, "    "))
      const number = String(row.kind === "del" ? row.oldNo : row.newNo).padStart(numbers)
      if (row.kind === "del") {
        rows.push(...wrap([{ text: `${number} - `, tone: "delGutter" }], blank,
          [{ text: code, tone: "delText" }], columns, "del"))
        continue
      }
      const body = highlightLine(code, diff.language).map((token): PromptSpan => (
        { text: token.text, tone: "text", color: token.color }))
      const gutter: PromptSpan = row.kind === "add"
        ? { text: `${number} + `, tone: "addGutter" }
        : { text: `${number}   `, tone: "muted" }
      rows.push(...wrap([gutter], blank, body, columns, row.kind === "add" ? "add" : undefined))
    }
  })
  if (diff.truncated) {
    rows.push({ spans: [{ text: `… +${diff.truncatedRows} more lines not shown`, tone: "muted" }] })
  }
  return rows
}

/** Rows the call's text may take: half the rows the rest of the screen
 * leaves (the other half stays with the transcript), and at least a few. */
function bodyRows(rows: number, queued: number, inputRows: number): number {
  const free = rows - FIXED_ROWS - inputRows - (queued > 0 ? queued + 1 : 0)
  return Math.max(MIN_BODY_ROWS, Math.floor(free / 2))
}

/** The prompt as drawn on a screen of `columns` x `rows`, with `queued`
 * messages shown above it and `inputRows` rows of typed text below it,
 * starting at row `offset` of the call (where it starts best, if omitted:
 * an edit's first change). */
export function permissionLayout(
  pending: PermissionPrompt,
  diff: DiffData | undefined,
  screen: { columns: number; rows: number; queued?: number; inputRows?: number },
  offset?: number,
): PromptLayout {
  const columns = Math.max(10, screen.columns - TEXT_INDENT)
  const [summary, target] = permissionLines(pending)
  const pinned: PromptRow[] = []
  let rows: PromptRow[]
  if (diff) {
    const counts: PromptSpan[] = [
      { text: "  ", tone: "muted" },
      { text: `+${diff.added}`, tone: "addGutter" },
      { text: " ", tone: "muted" },
      { text: `-${diff.removed}`, tone: "delGutter" },
    ]
    pinned.push(...wrap([], [], [{ text: visibleText(summary, Infinity), tone: "text" }, ...counts], columns))
    if (target) pinned.push(...textRows(target, "muted", columns))
    rows = diffRows(diff, Math.max(10, columns - DIFF_INDENT))
  } else {
    rows = textRows(summary, "text", columns)
    if (target) rows.push(...textRows(target, "muted", columns))
  }

  const room = bodyRows(screen.rows, screen.queued ?? 0, screen.inputRows ?? 1)
  const window = Math.min(rows.length, Math.max(1, room - pinned.length))
  const last = rows.length - window
  let start = offset
  if (start === undefined) {
    // From the row before an edit's first change, so the change is in view.
    const first = rows.findIndex((row) => row.band !== undefined)
    start = first > 0 ? first - 1 : 0
  }
  start = Math.max(0, Math.min(start, last))
  const clipped = diff?.hunks.some((hunk) => hunk.rows.some((row) => row.clipped)) ?? false
  // An edit without a diff (a file too large to compare) shows nothing it writes.
  const hidden = pending.permission === "edit" && !diff
  return {
    pinned,
    rows,
    offset: start,
    window,
    full: window === rows.length && !diff?.truncated && !clipped && !hidden,
  }
}
