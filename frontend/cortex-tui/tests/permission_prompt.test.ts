import { describe, expect, test } from "bun:test"
import type { DiffData } from "../src/context/session_store"
import { permissionLayout, permissionLines, type PromptRow } from "../src/lib/permission_prompt"
import { visibleText } from "../src/lib/visible_text"

const bash = (command: string) => ({
  permission: "bash",
  patterns: [command],
  metadata: { tool: "bash", arguments: { command } },
})
const edit = (path: string) => ({
  permission: "edit",
  patterns: [path],
  metadata: { tool: "edit_file", arguments: { path } },
})
const text = (row: PromptRow) => row.spans.map((span) => span.text).join("")
const columns = (row: PromptRow) => [...text(row)].reduce((n, ch) => n + (ch.codePointAt(0)! < 0x80 ? 1 : 2), 0)

function diff(rows: DiffData["hunks"][number]["rows"], extra: Partial<DiffData> = {}): DiffData {
  return {
    path: "values.py",
    op: "edit",
    added: rows.filter((row) => row.kind === "add").length,
    removed: rows.filter((row) => row.kind === "del").length,
    truncated: false,
    truncatedRows: 0,
    hunks: [{ oldStart: 1, newStart: 1, rows }],
    ...extra,
  }
}
const context = (n: number) => ({ kind: "ctx" as const, text: `line_${n} = ${n}`, oldNo: n, newNo: n })

describe("permission prompt", () => {
  test("shows a bash call as its command, once", () => {
    expect(permissionLines(bash("make test"))).toEqual(["Run command: make test"])
  })

  test("a short call is shown in full", () => {
    const layout = permissionLayout(bash("make test"), undefined, { columns: 80, rows: 24 })
    expect(layout.rows.map(text)).toEqual(["Run command: make test"])
    expect(layout.full).toBe(true)
  })

  test("a command of several lines reads line by line, its indentation as typed", () => {
    const layout = permissionLayout(bash("for f in *.py; do\n        wc -l $f\ndone"), undefined, { columns: 80, rows: 40 })
    expect(layout.rows.map(text)).toEqual(["Run command: for f in *.py; do\\n", "        wc -l $f\\n", "done"])
  })

  test("a long command wraps into rows that fit, losing nothing", () => {
    const command = "python3 -c \"" + "x = 1; ".repeat(120) + "print(x)\""
    const layout = permissionLayout(bash(command), undefined, { columns: 80, rows: 60 })
    expect(layout.rows.every((row) => columns(row) <= 80 - 7)).toBe(true)
    expect(layout.rows.map(text).join("")).toBe(visibleText(`Run command: ${command}`, Infinity))
    expect(layout.full).toBe(true)
  })

  test("a command taller than its window scrolls, and is not shown in full", () => {
    const command = "echo " + "A".repeat(5000) + "; touch LONG.txt"
    const layout = permissionLayout(bash(command), undefined, { columns: 80, rows: 24 })
    expect(layout.full).toBe(false)
    expect(layout.window).toBeLessThan(layout.rows.length)
    expect(layout.offset).toBe(0)
    expect(text(layout.rows[layout.rows.length - 1])).toContain("touch LONG.txt")
    const end = permissionLayout(bash(command), undefined, { columns: 80, rows: 24 }, 10_000)
    expect(end.offset).toBe(end.rows.length - end.window)
  })

  test("the call takes at most half the rows the screen leaves, the transcript the rest", () => {
    const command = Array.from({ length: 40 }, (_, n) => `echo ${n}`).join("\n")
    // 24 rows: 15 for the screen's chrome and 1 for the input leave 8, 4 of them for the call.
    expect(permissionLayout(bash(command), undefined, { columns: 80, rows: 24 }).window).toBe(4)
    expect(permissionLayout(bash(command), undefined, { columns: 80, rows: 24, queued: 1, inputRows: 3 }).window).toBe(3)
    expect(permissionLayout(bash(command), undefined, { columns: 120, rows: 40 }).window).toBe(12)
  })

  test("wide characters count as two columns", () => {
    const layout = permissionLayout(bash("echo " + "漢".repeat(100)), undefined, { columns: 80, rows: 40 })
    expect(layout.rows.every((row) => columns(row) <= 80 - 7)).toBe(true)
    // 18 columns of "Run command: echo " and 27 characters, then 36 a row.
    expect(layout.rows.map((row) => [...text(row)].length)).toEqual([45, 36, 36, 1])
  })

  test("an edit keeps its file in view and starts the window just before the change", () => {
    const rows = [1, 2, 3, 4, 5].map(context)
    const change = diff([...rows, { kind: "del", text: "line_6 = 6", oldNo: 6 }, { kind: "add", text: "line_6 = 60", newNo: 6 }, context(7)])
    const layout = permissionLayout(edit("values.py"), change, { columns: 80, rows: 24 })
    expect(layout.pinned.map(text)).toEqual(["Edit file: values.py  +1 -1"])
    expect(layout.window).toBe(3)
    expect(layout.rows.slice(layout.offset, layout.offset + layout.window).map(text)).toEqual([
      " 5   line_5 = 5",
      " 6 - line_6 = 6",
      " 6 + line_6 = 60",
    ])
    expect(layout.full).toBe(false)
    expect(permissionLayout(edit("values.py"), change, { columns: 80, rows: 40 }).full).toBe(true)
  })

  test("a long line of an edit wraps under its code, not its line number", () => {
    const change = diff([{ kind: "add", text: "x = '" + "a".repeat(100) + "'", newNo: 1 }])
    const layout = permissionLayout(edit("values.py"), change, { columns: 80, rows: 40 })
    expect(layout.rows.length).toBe(2)
    expect(text(layout.rows[0]).startsWith(" 1 + x = '")).toBe(true)
    expect(text(layout.rows[1]).startsWith("     a")).toBe(true)
    expect(layout.rows.every((row) => row.band === "add")).toBe(true)
  })

  test("an edit is not shown in full when a line or rows of it were left out", () => {
    const clipped = diff([{ kind: "add", text: "x…", newNo: 1, clipped: true }])
    expect(permissionLayout(edit("a.py"), clipped, { columns: 80, rows: 40 }).full).toBe(false)
    const truncated = diff([context(1)], { truncated: true, truncatedRows: 9 })
    const layout = permissionLayout(edit("a.py"), truncated, { columns: 80, rows: 40 })
    expect(layout.full).toBe(false)
    expect(text(layout.rows[layout.rows.length - 1])).toBe("… +9 more lines not shown")
    expect(permissionLayout(edit("a.py"), undefined, { columns: 80, rows: 40 }).full).toBe(false)
  })

  test("invisible characters in the new text are shown as escapes", () => {
    const change = diff([{ kind: "add", text: "ok = True\u202e  # \\n", newNo: 1 }])
    const layout = permissionLayout(edit("a.py"), change, { columns: 80, rows: 40 })
    expect(text(layout.rows[0])).toBe(" 1 + ok = True\\u{202e}  # \\n")
  })
})
