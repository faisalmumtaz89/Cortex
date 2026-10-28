// A command result's rows and window (src/lib/result_layout.ts).
import { expect, test } from "bun:test"
import { resultLayout } from "../src/lib/result_layout"

const help = [
  "/help           List available commands",
  "/map-and-plan   Use when asked how code works, where something happens, or for a change planned but not yet made.",
].join("\n")

test("a wrapped description goes on under its own column", () => {
  const { rows } = resultLayout(help, 16, { columns: 80, rows: 40 })
  expect(rows[0]).toBe("/help           List available commands")
  expect(rows.length).toBe(3)
  expect(rows.every((row) => row.length <= 80 - 8)).toBe(true)
  expect(rows[2].startsWith(" ".repeat(16))).toBe(true)
  expect(rows[2][16]).not.toBe(" ")
})

test("the rows follow the screen's width", () => {
  expect(resultLayout(help, 16, { columns: 140, rows: 40 }).rows.length).toBe(2)
})

test("a result taller than its room shows a window, and keeps a row for its position", () => {
  const text = Array.from({ length: 30 }, (_, n) => `line ${n}`).join("\n")
  const small = resultLayout(text, 0, { columns: 80, rows: 24 })
  // 24 rows leave the panel 10: its title and the row under it, then 8 for
  // the text, one of them saying which rows are shown.
  expect(small.window).toBe(7)
  expect(resultLayout(text, 0, { columns: 80, rows: 24 }, 99).offset).toBe(30 - 7)
  expect(resultLayout("one\ntwo", 0, { columns: 80, rows: 24 }).window).toBe(2)
})
