// How a screen's rows are shared out while a panel is open above the prompt
// (a picker, the slash palette, a command's result, a permission prompt): the
// rows every screen has, a few for the transcript so what led to the panel
// stays in view, and the rest for the panel. Each panel lays itself out in
// the rows it is given here, so on a screen of 24 rows or more they add up.

export type Screen = { columns: number; rows: number; queued?: number; inputRows?: number }

// Rows besides the transcript, the panel and the typed text: the screen's top
// and bottom padding (2), the header (1), the rows between the header, the
// transcript, the panel and the prompt (3), the input panel's rows around its
// text (2), and the hint under it with the row above (2).
const CHROME_ROWS = 10
const TRANSCRIPT_ROWS = 3

/** Rows the panel above the prompt may take on `screen`, with `queued`
 * messages shown above it and `inputRows` rows of typed text below it. */
export function panelRows(screen: Screen): number {
  const queued = screen.queued ?? 0
  return screen.rows - CHROME_ROWS - (screen.inputRows ?? 1) - (queued > 0 ? queued + 1 : 0) - TRANSCRIPT_ROWS
}
