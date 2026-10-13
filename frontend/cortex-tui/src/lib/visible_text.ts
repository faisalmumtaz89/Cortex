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
 * - Text longer than SHOWN_CHARS keeps its start and end, cut at whole
 *   characters, with a count of the characters in between; `shownInFull`
 *   then returns false, and the prompt starts on Reject. */
const BLANK = String.raw`\s\p{Zs}\p{Zl}\p{Zp}\p{Cf}\p{M}\p{Default_Ignorable_Code_Point}ᅟᅠ⠀ㅤﾠ`
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

/** Whether `visibleText(text)` shows all of `text`. */
export function shownInFull(text: string): boolean {
  return escaped(text).length <= SHOWN_CHARS
}

export function visibleText(text: string): string {
  const chars = escaped(text)
  if (chars.length <= SHOWN_CHARS) return chars.join("")
  const half = SHOWN_CHARS / 2
  const hidden = chars.length - 2 * half
  return `${chars.slice(0, half).join("")} ⟨${hidden} characters not shown⟩ ${chars.slice(-half).join("")}`
}
