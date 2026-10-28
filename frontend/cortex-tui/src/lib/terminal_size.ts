import { createSignal } from "solid-js"

// Reactive terminal dimensions. process.stdout.columns is only sampled at
// launch; reading it inline freezes width-dependent layout (header path,
// horizontal rules, tool-path budgets) so they never reflow on resize. This
// signal updates on SIGWINCH so anything that reads terminalColumns() inside a
// reactive scope re-computes when the terminal is resized.

const [columns, setColumns] = createSignal(process.stdout.columns ?? 80)
const [rows, setRows] = createSignal(process.stdout.rows ?? 24)

const sync = () => {
  setColumns(process.stdout.columns ?? 80)
  setRows(process.stdout.rows ?? 24)
}

// tty.WriteStream emits "resize" on SIGWINCH; belt-and-suspenders with the
// signal directly in case stdout isn't a TTY under the sidecar.
process.stdout.on("resize", sync)
process.on("SIGWINCH", sync)

export const terminalColumns = columns
export const terminalRows = rows

/** The columns inside the screen's side padding (2 + 2): the transcript's,
 * and the panels' around the prompt. */
export function screenColumns(): number {
  return Math.max(10, terminalColumns() - 4)
}

// A reply's widest line on a wide terminal (its box's maxWidth), for blocks
// such as tables that lay themselves out to fit it.
export const PROSE_MAX_WIDTH = 104
// Columns a reply's text loses to the transcript's margins and its own indent.
const PROSE_MARGINS = 8

/** The columns a reply's text has on this terminal. */
export function proseColumns(): number {
  return Math.max(20, Math.min(PROSE_MAX_WIDTH - 3, terminalColumns() - PROSE_MARGINS))
}
