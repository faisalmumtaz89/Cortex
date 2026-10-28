// What a permission prompt shows, laid out to fit the screen. Each row is cut
// to the prompt's width here, so it takes exactly one terminal row, and the
// call's rows go in a window of the rows the screen leaves the prompt (see
// screen_rows); PgUp/PgDn page through them, Home/End jump to either end. The store decides the prompt's
// starting choice from this layout, so Enter never approves a call whose
// text was not on screen.
import { highlightLine } from "../components/syntax_highlight"
import type { DiffData } from "../context/session_store"
import { displayPath } from "./paths"
import { type Screen, panelRows } from "./screen_rows"
import { type PromptRow, type PromptSpan, type PromptTone, wrapSpans } from "./text_rows"
import { visibleCode, visibleLines } from "./visible_text"

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
  const shown = (text: string) => visibleLines(text).join("\n")
  if (pending.permission !== "bash" && !shown(summary).includes(shown(target))) {
    lines.push(`Target: ${target}`)
  }
  return lines
}


export type PromptLayout = {
  /** Always on screen: what the call is (an edit's file and the size of its change). */
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
// The prompt's rows besides the call's: its title, the row for the scroll
// position and the 3 choices.
const PROMPT_ROWS = 5
const MIN_BODY_ROWS = 3

function textRows(text: string, tone: PromptTone, columns: number): PromptRow[] {
  return visibleLines(text).flatMap((line) => wrapSpans([], [], [{ text: line, tone }], columns))
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
        rows.push(...wrapSpans([{ text: `${number} - `, tone: "delGutter" }], blank,
          [{ text: code, tone: "delText" }], columns, "del"))
        continue
      }
      const body = highlightLine(code, diff.language).map((token): PromptSpan => (
        { text: token.text, tone: "text", color: token.color }))
      const gutter: PromptSpan = row.kind === "add"
        ? { text: `${number} + `, tone: "addGutter" }
        : { text: `${number}   `, tone: "muted" }
      rows.push(...wrapSpans([gutter], blank, body, columns, row.kind === "add" ? "add" : undefined))
    }
  })
  if (diff.truncated) {
    rows.push({ spans: [{ text: `… +${diff.truncatedRows} more lines not shown`, tone: "muted" }] })
  }
  return rows
}

/** The prompt as drawn on `screen`, starting at row `offset` of the call
 * (where it starts best, if omitted: an edit's first change). */
export function permissionLayout(
  pending: PermissionPrompt,
  diff: DiffData | undefined,
  screen: Screen,
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
    pinned.push(...wrapSpans([], [], [{ text: visibleLines(summary).join(" "), tone: "text" }, ...counts], columns))
    if (target) pinned.push(...textRows(target, "muted", columns))
    rows = diffRows(diff, Math.max(10, columns - DIFF_INDENT))
  } else if (pending.permission === "bash") {
    // The label stays in view; the command itself scrolls.
    const label = "Run command: "
    pinned.push({ spans: [{ text: label.trimEnd(), tone: "text" }] })
    rows = textRows(summary.slice(label.length), "text", columns)
  } else {
    rows = textRows(summary, "text", columns)
    if (target) rows.push(...textRows(target, "muted", columns))
  }

  // The call's rows: the panel's but the prompt's own, and at least a few.
  const room = Math.max(MIN_BODY_ROWS, panelRows(screen) - PROMPT_ROWS)
  const window = Math.min(rows.length, Math.max(1, room - pinned.length))
  const last = rows.length - window
  let start = offset
  if (start === undefined) {
    // From the line before an edit's first change, its first row, so the
    // change is in view after a whole line of context.
    const first = rows.findIndex((row) => row.band !== undefined)
    start = Math.max(0, first - 1)
    while (start > 0 && rows[start].continued) start -= 1
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
