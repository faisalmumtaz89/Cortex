// How the input shows a /login line: the key typed after the provider as one
// • per character, so the screen (and what reads it: a screen reader, copy
// and paste, a recording) never holds the key. The text itself is kept by
// the caller, and each edit of the shown text is applied to it.

/** Where the key is in `text`, as [start, end), if it is a /login line. */
export function keyRange(text: string): [number, number] | undefined {
  const match = /^(\s*\/login\s+\S+\s+)(\S+)/i.exec(text)
  return match ? [match[1].length, match[1].length + match[2].length] : undefined
}

/** `text` as the input shows it. */
export function masked(text: string): string {
  const range = keyRange(text)
  if (!range) return text
  return text.slice(0, range[0]) + "•".repeat(range[1] - range[0]) + text.slice(range[1])
}

/** The text after an edit turned `shownBefore` (how `before` was shown) into
 * `shownAfter`, with the cursor at `cursor`. Every • looks the same, so the
 * edit is taken to end at the cursor, as typing, pasting and deleting do. */
export function applyEdit(before: string, shownBefore: string, shownAfter: string, cursor: number): string {
  if (shownBefore === before) return shownAfter // nothing was hidden
  const suffix = Math.max(0, Math.min(shownAfter.length - cursor, shownBefore.length))
  let prefix = 0
  const limit = Math.min(cursor, shownBefore.length - suffix)
  while (prefix < limit && shownBefore[prefix] === shownAfter[prefix]) prefix += 1
  return before.slice(0, prefix) + shownAfter.slice(prefix, shownAfter.length - suffix) + before.slice(before.length - suffix)
}
