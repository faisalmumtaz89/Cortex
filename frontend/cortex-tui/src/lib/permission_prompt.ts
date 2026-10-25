// What a permission prompt shows, and whether it shows the whole call. The
// store decides the prompt's starting choice from exactly these lines, so
// Enter never approves a call whose text the user could not see.
import { displayPath } from "./paths"
import { shownInFull, visibleText } from "./visible_text"

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

/** The prompt's lines as shown: the summary, and the Target line when it adds
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

// Rows the rest of the screen keeps (header, input panel, status line) and
// that the prompt adds around its lines (title, options, footer, spacing).
const RESERVED_ROWS = 10
const PROMPT_ROWS = 7

/** Columns `text` can take at most: a non-ASCII character may be wide (CJK,
 * emoji), so it counts as two. */
function columnsAtMost(text: string): number {
  let count = 0
  for (const ch of text) count += ch.codePointAt(0)! < 0x80 ? 1 : 2
  return count
}

/** Whether the prompt shows all of the call and fits on screen, with
 * `extraRows` taken by other lines (queued messages). */
export function permissionShownInFull(
  pending: PermissionPrompt,
  columns: number,
  rows: number,
  extraRows = 0,
): boolean {
  const lines = permissionLines(pending)
  if (!lines.every(shownInFull)) return false
  const width = Math.max(10, columns - 4)
  const textRows = lines
    // One row more per line: wrapping at word breaks can leave rows short.
    .map((line) => Math.ceil(columnsAtMost(visibleText(line)) / width) + 1)
    .reduce((sum, count) => sum + count, 0)
  return textRows + PROMPT_ROWS + RESERVED_ROWS + extraRows <= rows
}
