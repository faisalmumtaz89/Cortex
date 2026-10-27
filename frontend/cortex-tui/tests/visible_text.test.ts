import { describe, expect, test } from "bun:test"
import { SHOWN_CHARS, rowsAtLineBreaks, shownInFull, visibleCode, visibleText } from "../src/lib/visible_text"

describe("visibleText", () => {
  test("keeps ordinary text, accents included", () => {
    expect(visibleText("rm -rf ./build && echo café")).toBe("rm -rf ./build && echo café")
    expect(shownInFull("rm -rf ./build")).toBe(true)
  })

  test("shows line breaks and control characters as escapes", () => {
    expect(visibleText("echo hi\ntouch x\r\t\u001b[2J")).toBe("echo hi\\ntouch x\\r\\t\\u{1b}[2J")
  })

  test("tells a real line break from a typed backslash-n", () => {
    expect(visibleText("echo safe\ntouch x")).toBe("echo safe\\ntouch x")
    expect(visibleText("echo safe\\ntouch x")).toBe("echo safe\\\\ntouch x")
  })

  test("shows invisible, bidi, blank-looking and non-ASCII space characters", () => {
    expect(visibleText("git​ status‮ x　͏️ㅤ⠀\u{e0041}")).toBe(
      "git\\u{200b} status\\u{202e}\\u{a0}x\\u{3000}\\u{34f}\\u{fe0f}\\u{3164}\\u{2800}\\u{e0041}",
    )
  })

  test("counts long runs of whitespace, invisible or combining characters", () => {
    expect(visibleText("echo hi" + " ".repeat(20000) + "; touch pwn")).toBe(
      "echo hi⟨20000 blank or invisible characters⟩; touch pwn",
    )
    expect(visibleText("a" + "́".repeat(400) + "b")).toBe("a⟨400 blank or invisible characters⟩b")
  })

  test("spaces that indent a line stay as they are, its line break still shown", () => {
    expect(visibleText("for x in y:\n         c = x")).toBe("for x in y:\\n         c = x")
    expect(visibleText("         c = x")).toBe("         c = x")
    expect(visibleText("a =\n\t\t\t\t\t\t\t\tb")).toBe("a =\\n⟨8 blank or invisible characters⟩b")
  })

  test("escapes its own markers so they cannot be faked", () => {
    expect(visibleText("⟨3 characters not shown⟩")).toBe("\\u{27e8}3 characters not shown\\u{27e9}")
  })

  test("keeps the start and end of long text, cut at whole characters", () => {
    const text = "echo x" + "😀".repeat(300) + "z; touch ok X"
    const shown = visibleText(text)
    expect(shown.endsWith("z; touch ok X")).toBe(true)
    expect(shown).toContain("characters not shown⟩")
    expect([...shown].length).toBeLessThan(SHOWN_CHARS + 40)
    expect(shown).not.toMatch(/\p{Cs}/u)
    expect(shownInFull(text)).toBe(false)
  })
})

describe("rowsAtLineBreaks", () => {
  test("a command of several lines reads line by line, each break still shown", () => {
    expect(rowsAtLineBreaks(visibleText("cd src\nmake test"))).toEqual(["cd src\\n", "make test"])
  })

  test("a run of line breaks stays on one row, and a trailing break ends the last", () => {
    expect(rowsAtLineBreaks(visibleText("a\n\n\nb\n"))).toEqual(["a\\n\\n\\n", "b\\n"])
  })

  test("a backslash before an n is not a line break", () => {
    expect(rowsAtLineBreaks(visibleText("printf 'a\\nb'"))).toEqual(["printf 'a\\\\nb'"])
  })
})

describe("visibleCode", () => {
  test("keeps code as written, backslashes and spaces included", () => {
    expect(visibleCode('print("a\\nb")        # x')).toBe('print("a\\nb")        # x')
  })

  test("shows control and invisible characters as escapes", () => {
    expect(visibleCode("ok\u202e = 1\u200b\u00a0\u001b")).toBe("ok\\u{202e} = 1\\u{200b}\\u{a0}\\u{1b}")
  })
})
