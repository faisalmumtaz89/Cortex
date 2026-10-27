// The rows an edit prompt shows when the diff is taller than the room for it
// (src/lib/diff_window.ts). Run via `bun test` (wired into the pytest gate by
// tests/test_frontend_unit.py).
import { expect, test } from "bun:test"
import { diffWindow } from "../src/lib/diff_window"

test("a diff that fits is shown whole", () => {
  expect(diffWindow(["ctx", "add", "ctx"], 5)).toEqual({ start: 0, end: 3 })
})

test("a cut diff starts one row before its first change, after the note", () => {
  const kinds = ["ctx", "ctx", "ctx", "add", "ctx", "ctx", "ctx"]
  // Four rows: the note, one unchanged row, the change, one more.
  expect(diffWindow(kinds, 4)).toEqual({ start: 2, end: 5 })
})

test("with little room the change comes first", () => {
  // Three rows on a 24-row screen: the note, then the removed and added rows.
  const kinds = ["ctx", "ctx", "ctx", "ctx", "del", "add", "ctx", "ctx", "ctx"]
  expect(diffWindow(kinds, 3)).toEqual({ start: 4, end: 6 })
})

test("a change at the end still fills the room", () => {
  expect(diffWindow(["ctx", "ctx", "ctx", "ctx", "ctx", "del"], 4)).toEqual({ start: 3, end: 6 })
})

test("a change on the first row keeps the start", () => {
  expect(diffWindow(["add", "ctx", "ctx", "ctx"], 2)).toEqual({ start: 0, end: 2 })
})
