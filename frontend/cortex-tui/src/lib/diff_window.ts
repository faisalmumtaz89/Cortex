// Which rows of an edit's diff a permission prompt shows when only some fit:
// from the row before the first change (or the change itself when the room
// is tight), so the change is never what is cut off.

export type DiffWindow = { start: number; end: number }

const CHANGE = new Set(["add", "del"])

/** The rows from `start` to `end` fit `limit`; a window that does not start
 * at the top takes one of those rows for its "… N lines above" note. */
export function diffWindow(kinds: string[], limit: number): DiffWindow {
  if (kinds.length <= limit) {
    return { start: 0, end: kinds.length }
  }
  const first = Math.max(0, kinds.findIndex((kind) => CHANGE.has(kind)))
  let change = 0
  while (first + change < kinds.length && CHANGE.has(kinds[first + change])) change += 1
  const room = (start: number) => (start > 0 ? Math.max(1, limit - 1) : limit)
  // One unchanged row before the change, when the change still fits after it.
  let start = first > 0 && change + 1 <= room(first - 1) ? first - 1 : first
  // Near the end, start earlier so the window stays full.
  start = Math.max(0, Math.min(start, kinds.length - room(start)))
  return { start, end: Math.min(kinds.length, start + room(start)) }
}
