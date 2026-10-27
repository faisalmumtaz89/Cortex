// How a markdown table is laid out in the transcript: columns sized to their
// text and narrowed (widest first) until the table fits the reply's width; a
// cell too wide for its column wraps onto more lines of its row.

const COLUMN_GAP = "  │  "
const SEPARATOR_GAP = "──┼──"
const MIN_COLUMN = 6

export type TableLine = { kind: "header" | "separator" | "row"; text: string }

/** `text` broken into lines of at most `width` characters, between words
 * where it can be. */
export function wrapCell(text: string, width: number): string[] {
  const lines: string[] = []
  let line = ""
  for (const word of text.split(/\s+/).filter(Boolean)) {
    let rest = word
    while (rest.length > width) {
      // A word longer than the column is cut where the line ends.
      const room = line ? width - line.length - 1 : width
      if (room <= 0) {
        lines.push(line)
        line = ""
        continue
      }
      lines.push(line ? `${line} ${rest.slice(0, room)}` : rest.slice(0, room))
      line = ""
      rest = rest.slice(room)
    }
    if (!rest) continue
    if (!line) {
      line = rest
    } else if (line.length + 1 + rest.length <= width) {
      line += ` ${rest}`
    } else {
      lines.push(line)
      line = rest
    }
  }
  if (line || lines.length === 0) lines.push(line)
  return lines
}

/** Column widths: each column's widest cell, narrowed widest first until the
 * table and the gaps between its columns fit `maxWidth`. */
export function columnWidths(cells: string[][], maxWidth: number): number[] {
  const count = Math.max(0, ...cells.map((row) => row.length))
  const widths = new Array<number>(count).fill(1)
  for (const row of cells) {
    row.forEach((cell, column) => {
      widths[column] = Math.max(widths[column], cell.length)
    })
  }
  const budget = maxWidth - COLUMN_GAP.length * Math.max(0, count - 1)
  let total = widths.reduce((sum, width) => sum + width, 0)
  while (total > budget) {
    const widest = widths.indexOf(Math.max(...widths))
    if (widths[widest] <= MIN_COLUMN) break // too many columns to fit: they wrap
    widths[widest] -= 1
    total -= 1
  }
  return widths
}

export function tableLines(header: string[], rows: string[][], maxWidth: number): TableLine[] {
  const widths = columnWidths([header, ...rows], maxWidth)
  const render = (cells: string[], kind: "header" | "row"): TableLine[] => {
    const wrapped = widths.map((width, column) => wrapCell(cells[column] ?? "", width))
    const height = Math.max(...wrapped.map((lines) => lines.length))
    const lines: TableLine[] = []
    for (let index = 0; index < height; index++) {
      const text = wrapped
        .map((cell, column) => (cell[index] ?? "").padEnd(widths[column]))
        .join(COLUMN_GAP)
        .trimEnd()
      lines.push({ kind, text })
    }
    return lines
  }
  return [
    ...render(header, "header"),
    { kind: "separator", text: widths.map((width) => "─".repeat(width)).join(SEPARATOR_GAP) },
    ...rows.flatMap((row) => render(row, "row")),
  ]
}
