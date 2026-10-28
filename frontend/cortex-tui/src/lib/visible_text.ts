/** How a permission prompt shows text the user approves: as it was written,
 * with nothing in it invisible or ambiguous.
 *
 * - Each line of the text is a line of its own, ended by ` ↵` when a line
 *   break follows it.
 * - Control characters, invisible and default-ignorable characters
 *   (zero-width, bidi overrides, variation selectors), blank lookalikes,
 *   spaces other than U+0020 and lone surrogates show as markers such as
 *   `⟨U+200B⟩`, a tab as `⟨tab⟩` and a carriage return as `⟨CR⟩`.
 * - A run of 8 or more whitespace, invisible or combining characters is
 *   counted instead, except spaces that indent a line: those stay as they are.
 * - The marker characters themselves (⟨ ⟩ ↵) are shown as markers, so a
 *   marker in the text cannot pass for one the prompt drew. Backslashes are
 *   text like any other. */
const BLANK = String.raw`\t\v\f﻿\p{Zs}\p{Zl}\p{Zp}\p{Cf}\p{M}\p{Default_Ignorable_Code_Point}ᅟᅠ⠀ㅤﾠ`
const MARKED = String.raw`\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}\p{Zs}\p{Default_Ignorable_Code_Point}ᅟᅠ⠀ㅤﾠ⟨⟩↵`
const TOKEN = new RegExp(`([${BLANK}]{8,})|([${MARKED}])`, "gu")

function marker(ch: string): string {
  if (ch === " ") return " "
  if (ch === "\t") return "⟨tab⟩"
  if (ch === "\r") return "⟨CR⟩"
  return `⟨U+${ch.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")}⟩`
}

function visibleLine(line: string): string {
  return line.replace(TOKEN, (_match, run: string | undefined, ch: string | undefined, offset: number) => {
    if (!run) return marker(ch!)
    const indent = offset === 0 && /^ +$/.test(run)
    return indent ? run : `⟨${[...run].length} blank or invisible characters⟩`
  })
}

/** `text` as the prompt's rows show it, one entry per line. */
export function visibleLines(text: string): string[] {
  const lines = text.split("\n")
  return lines.map((line, index) => visibleLine(line) + (index < lines.length - 1 ? " ↵" : ""))
}

// What a diff row marks: control and invisible characters, so nothing in the
// text an edit writes is hidden. Spaces are code, as written.
const CODE_MARKED = new RegExp(
  String.raw`[\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}\p{Default_Ignorable_Code_Point}ᅟᅠ⠀ㅤﾠ⟨⟩]|(?! )\p{Zs}`,
  "gu",
)

/** A line of code as a permission prompt shows it: `visibleLines` without
 * the counting of blank runs, which only a command needs. */
export function visibleCode(text: string): string {
  return text.replace(CODE_MARKED, marker)
}
