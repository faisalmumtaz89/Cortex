// Text cut into rows by its drawn width (src/lib/text_rows.ts).
import { expect, test } from "bun:test"
import { cutRows } from "../src/lib/text_rows"

const text = (rows: { text: string }[][]) => rows.map((row) => row.map((span) => span.text).join(""))
const marker = [{ text: "↪ " }]

test("a character is counted no narrower than the terminal draws it", () => {
  // CJK takes two columns, and so does a tab, counted as any non-ASCII character is.
  expect(text(cutRows([], marker, [{ text: "漢字かな" }], 5, true))).toEqual(["漢字", "↪ か", "↪ な"])
  expect(text(cutRows([], [], [{ text: "a\tb" }], 3, true))).toEqual(["a\t", "b"])
})

test("a letter and its accents, or an emoji sequence, stay in one row", () => {
  expect(text(cutRows([], [], [{ text: "abcde\u0301f" }], 5, true))).toEqual(["abcd", "e\u0301f"])
  const family = "xxxxxx👨‍👩‍👧abc"
  expect(text(cutRows([], marker, [{ text: family }], 14, true))).toEqual(["xxxxxx", "↪ 👨‍👩‍👧a", "↪ bc"])
})

test("a row kept full ends with the space it breaks at, so none starts with one", () => {
  expect(text(cutRows([], marker, [{ text: "abc def ghi jkl" }], 7, true))).toEqual(["abc ", "↪ def ", "↪ ghi ", "↪ jkl"])
})

test("every character is kept, in order, across rows and spans", () => {
  const spans = [{ text: "def summary(name, values):" }, { text: "  return f\"{name}: {mean(values):.2f}\"" }]
  const rows = cutRows([], marker, spans, 20, true)
  expect(text(rows).map((row, index) => (index ? row.slice(2) : row)).join("")).toBe(spans.map((s) => s.text).join(""))
})
