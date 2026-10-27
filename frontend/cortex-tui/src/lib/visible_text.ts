/** How a permission prompt shows text the user approves: one line in which
 * nothing is invisible or ambiguous, short enough to leave the prompt's
 * options on screen.
 *
 * - Backslashes, control characters, invisible and default-ignorable
 *   characters (zero-width, bidi overrides, variation selectors), blank
 *   lookalikes, spaces other than U+0020, lone surrogates and the ⟨ ⟩ of this
 *   function's own markers become escapes like `\n` or `\u{202e}`.
 * - A run of 8 or more whitespace, invisible or combining characters becomes
 *   a count.
 * - Text longer than `limit` characters (SHOWN_CHARS unless the screen has
 *   room for more) keeps its start and end, cut at whole characters, with a
 *   count of the characters in between; `shownInFull` then returns false,
 *   and the prompt starts on Reject. */
// Not \n or \r: a line break is shown on its own, never folded into a count.
const BLANK = String.raw`\t\v\f\ufeff\p{Zs}\p{Zl}\p{Zp}\p{Cf}\p{M}\p{Default_Ignorable_Code_Point}ᅟᅠ⠀ㅤﾠ`
const ESCAPE = String.raw`\\\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}\p{Zs}\p{Default_Ignorable_Code_Point}ᅟᅠ⠀ㅤﾠ⟨⟩`
const TOKEN = new RegExp(`([${BLANK}]{8,})|([${ESCAPE}])`, "gu")
export const SHOWN_CHARS = 240

function escape(ch: string): string {
  if (ch === " ") return " "
  if (ch === "\\") return "\\\\"
  if (ch === "\n") return "\\n"
  if (ch === "\r") return "\\r"
  if (ch === "\t") return "\\t"
  return `\\u{${ch.codePointAt(0)!.toString(16)}}`
}

function escaped(text: string): string[] {
  const shown = text.replace(TOKEN, (_match, run: string | undefined, ch: string | undefined) =>
    run ? `⟨${[...run].length} blank or invisible characters⟩` : escape(ch!),
  )
  return [...shown]
}

/** Whether `visibleText(text, limit)` shows all of `text`. */
export function shownInFull(text: string, limit = SHOWN_CHARS): boolean {
  return escaped(text).length <= limit
}

export function visibleText(text: string, limit = SHOWN_CHARS): string {
  const chars = escaped(text)
  if (chars.length <= limit) return chars.join("")
  const half = Math.floor(limit / 2)
  const hidden = chars.length - 2 * half
  return `${chars.slice(0, half).join("")} ⟨${hidden} characters not shown⟩ ${chars.slice(-half).join("")}`
}

/** Shown text split into rows after each line break it shows (`\n`, kept at
 * the row's end; a run of them stays on one row), so a command of several
 * lines reads line by line while every break stays visible. */
export function rowsAtLineBreaks(shown: string): string[] {
  const rows: string[] = []
  let row = ""
  let index = 0
  while (index < shown.length) {
    if (shown[index] !== "\\") {
      row += shown[index]
      index += 1
      continue
    }
    // An escape: "\\" (a backslash), "\n", "\r", "\t" or "\u{...}".
    const end = shown[index + 1] === "u" ? shown.indexOf("}", index) + 1 : index + 2
    const token = shown.slice(index, end > index ? end : index + 2)
    row += token
    index += token.length
    if (token === "\\n" && !shown.startsWith("\\n", index) && index < shown.length) {
      rows.push(row)
      row = ""
    }
  }
  rows.push(row)
  return rows
}
