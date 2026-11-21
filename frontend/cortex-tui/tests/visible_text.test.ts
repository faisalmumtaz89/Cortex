import { describe, expect, test } from "bun:test"
import { cappedLines, visibleCode, visibleLines } from "../src/lib/visible_text"

describe("visibleLines", () => {
  test("keeps ordinary text as written, backslashes and accents included", () => {
    expect(visibleLines("printf 'a\\nb' && echo café")).toEqual(["printf 'a\\nb' && echo café"])
  })

  test("gives each line a row, a line break shown as ↵ at its end", () => {
    expect(visibleLines("cd src\nmake test")).toEqual(["cd src ↵", "make test"])
    expect(visibleLines("a\n\nb\n")).toEqual(["a ↵", " ↵", "b ↵", ""])
  })

  test("marks control, invisible, bidi, blank-looking and non-ASCII space characters", () => {
    expect(visibleLines("git\u200b status\u202e\u00a0x\u3000\t\r\u001b[2J")).toEqual([
      "git⟨U+200B⟩ status⟨U+202E⟩⟨U+00A0⟩x⟨U+3000⟩⟨tab⟩⟨CR⟩⟨U+001B⟩[2J",
    ])
  })

  test("counts long runs of blank or invisible characters, but not indentation", () => {
    expect(visibleLines("echo hi" + " ".repeat(20000) + "; touch pwn")).toEqual([
      "echo hi⟨20000 blank or invisible characters⟩; touch pwn",
    ])
    expect(visibleLines("for x in y:\n         c = x")).toEqual(["for x in y: ↵", "         c = x"])
    expect(visibleLines("a\n\t\t\t\t\t\t\t\tb")).toEqual(["a ↵", "⟨8 blank or invisible characters⟩b"])
  })

  test("marks its own marker characters, so they cannot be faked", () => {
    expect(visibleLines("⟨3 blank or invisible characters⟩ ↵")).toEqual([
      "⟨U+27E8⟩3 blank or invisible characters⟨U+27E9⟩ ⟨U+21B5⟩",
    ])
  })
})

describe("visibleCode", () => {
  test("keeps code as written, backslashes and runs of spaces included", () => {
    expect(visibleCode('print("a\\nb")        # x')).toBe('print("a\\nb")        # x')
  })

  test("marks control and invisible characters", () => {
    expect(visibleCode("ok\u202e = 1\u200b\u00a0\u001b⟨")).toBe("ok⟨U+202E⟩ = 1⟨U+200B⟩⟨U+00A0⟩⟨U+001B⟩⟨U+27E8⟩")
  })
})

describe("the lines of a command or of output, capped", () => {
  const lines = (count: number) => Array.from({ length: count }, (_, index) => `line ${index + 1}`)

  test("up to the limit they are all shown, and a final line break is not a line", () => {
    expect(cappedLines(lines(40))).toEqual(lines(40))
    expect(cappedLines([...lines(40), ""])).toEqual(lines(40))
    expect(cappedLines([""])).toEqual([""])
  })

  test("past it, the rest is counted in the brackets no text can draw", () => {
    expect(cappedLines(lines(41)).slice(-2)).toEqual(["line 40", "⟨1 more line⟩"])
    expect(cappedLines(lines(45)).slice(-1)).toEqual(["⟨5 more lines⟩"])
    expect(cappedLines([...lines(45), ""]).slice(-1)).toEqual(["⟨5 more lines⟩"])
  })

  test("a line that reads like the count is drawn as text, not as the count", () => {
    expect(cappedLines(visibleLines("echo a\n⟨5 more lines⟩"))).toEqual([
      "echo a ↵",
      "⟨U+27E8⟩5 more lines⟨U+27E9⟩",
    ])
    expect(cappedLines(["out", visibleCode("⟨5 more lines⟩")])).toEqual([
      "out",
      "⟨U+27E8⟩5 more lines⟨U+27E9⟩",
    ])
  })
})
