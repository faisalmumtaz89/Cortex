// How a command's result panel (/help, /status, ...) shows its text: cut into
// rows that fit the panel on this screen, in a window of the rows the screen
// leaves the panel (see screen_rows). When the text has more rows, a row under
// it says which are shown, and PgUp/PgDn/Home/End move the window.
import { type Screen, panelRows } from "./screen_rows"
import { wrapText } from "./text_rows"

const RESULT_MAX_ROWS = 24
const RESULT_MIN_ROWS = 3
// The panel's rows besides its text: the ✓/✕ title and the row under it.
const TITLE_ROWS = 2
// Columns a row loses: the screen's side padding (2 + 2), the panel's rail and
// its padding (1 + 2 + 1).
const PANEL_INDENT = 8

export type ResultLayout = { rows: string[]; offset: number; window: number }

/** `text` laid out for `screen`, from row `offset`. A line that wraps goes on
 * `hang` columns in (under /help's descriptions). */
export function resultLayout(text: string, hang: number, screen: Screen, offset = 0): ResultLayout {
  const columns = Math.max(10, screen.columns - PANEL_INDENT)
  const rows = text.split("\n").flatMap((line) => (line ? wrapText(line, columns, hang) : [""]))
  const room = Math.max(RESULT_MIN_ROWS, Math.min(RESULT_MAX_ROWS, panelRows(screen) - TITLE_ROWS))
  // The row that says which rows are shown takes one of them.
  const window = rows.length <= room ? rows.length : room - 1
  return { rows, offset: Math.max(0, Math.min(offset, rows.length - window)), window }
}
