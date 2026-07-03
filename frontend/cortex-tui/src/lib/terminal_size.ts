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
