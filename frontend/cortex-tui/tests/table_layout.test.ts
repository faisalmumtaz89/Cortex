// Markdown tables in the transcript (src/lib/table_layout.ts): a table wider
// than the screen must be narrowed to fit it, never wrapped by the terminal.
// Run via `bun test` (wired into the pytest gate by tests/test_frontend_unit.py).
import { expect, test } from "bun:test"
import { columnWidths, tableLines, wrapCell } from "../src/lib/table_layout"

test("a cell wraps between words, and a word longer than the column is cut", () => {
  expect(wrapCell("4 run, 1 FAIL", 8)).toEqual(["4 run, 1", "FAIL"])
  expect(wrapCell("abcdefghij", 4)).toEqual(["abcd", "efgh", "ij"])
  expect(wrapCell("", 4)).toEqual([""])
})

test("a table that fits keeps each column as wide as its widest cell", () => {
  expect(columnWidths([["Path", "Role"], ["src/app.py", "entry"]], 80)).toEqual([10, 5])
  const lines = tableLines(["Path", "Role"], [["src/app.py", "entry"]], 80)
  expect(lines.map((line) => line.text)).toEqual([
    "Path        │  Role",
    `${"─".repeat(12)}┼${"─".repeat(7)}`,
    "src/app.py  │  entry",
  ])
})

test("a table wider than the screen is narrowed to fit it, widest column first", () => {
  const command = "python3 -m unittest discover -s tests -v (the README's command)"
  const rows = [[command, "4 run, 1 FAIL", "4 run, OK"]]
  const lines = tableLines(["Command", "Before", "After"], rows, 60)
  for (const line of lines) {
    expect(line.text.length).toBeLessThanOrEqual(60)
  }
  // The long cell wraps inside its own column; the short ones stay whole.
  const body = lines.filter((line) => line.kind === "row").map((line) => line.text)
  expect(body.length).toBeGreaterThan(1)
  expect(body[0]).toContain("4 run, 1 FAIL")
  expect(body[0]).toContain("4 run, OK")
})
