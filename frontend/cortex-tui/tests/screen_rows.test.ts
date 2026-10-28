// The rows a panel above the prompt is given (src/lib/screen_rows.ts).
import { expect, test } from "bun:test"
import { panelRows } from "../src/lib/screen_rows"
import { listWindow } from "../src/components/selection_list"

test("the panel takes the rows the chrome, the input and the transcript's 3 leave", () => {
  expect(panelRows({ columns: 80, rows: 24 })).toBe(10)
  expect(panelRows({ columns: 80, rows: 24, inputRows: 3 })).toBe(8)
  // Each queued message, and the row under them.
  expect(panelRows({ columns: 80, rows: 24, queued: 2 })).toBe(7)
})

test("a list shows every row that fits, else a window with room for its '+N more' rows", () => {
  // 10 rows, 4 of them the title and footer: 6 items fit at once.
  expect(listWindow(10, 4, 6)).toBe(6)
  expect(listWindow(10, 4, 7)).toBe(4)
  // Never more than 8 at once, nor fewer than 3.
  expect(listWindow(40, 4, 12)).toBe(8)
  expect(listWindow(40, 4, 8)).toBe(8)
  expect(listWindow(5, 4, 12)).toBe(3)
})
