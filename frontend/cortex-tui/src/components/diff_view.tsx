import { createMemo, For, Show } from "solid-js"
import type { DiffData, DiffRow } from "../context/store"
import { diffWindow } from "../lib/diff_window"
import { highlightLine } from "./syntax_highlight"
import { UI_PALETTE } from "./ui_palette"

const COLLAPSED_ROWS = 40

// A hunk separator or a real diff row.
type DisplayRow = { sep: true } | { sep: false; row: DiffRow }

function lineNumber(row: DiffRow): number {
  return row.kind === "del" ? row.oldNo : row.newNo
}

function plural(count: number): string {
  return count === 1 ? "" : "s"
}

/** Green/red file-edit diff (GitHub/Claude-Code style): a "⎿ Added N, removed
 * M" connector then per-row colored bands with a single line-number gutter. */
export function DiffView(props: { diff: DiffData; expanded: boolean; maxRows?: number }) {
  const diff = createMemo(() => props.diff)

  const gutterWidth = createMemo(() => {
    let max = 0
    for (const hunk of diff().hunks) {
      for (const row of hunk.rows) {
        const n = lineNumber(row)
        if (n > max) max = n
      }
    }
    return Math.max(2, String(max).length)
  })

  const flatRows = createMemo<DisplayRow[]>(() => {
    const rows: DisplayRow[] = []
    diff().hunks.forEach((hunk, index) => {
      if (index > 0) rows.push({ sep: true })
      for (const row of hunk.rows) rows.push({ sep: false, row })
    })
    return rows
  })

  // The rows shown: all when expanded, else as many as fit, starting just
  // before the first change so the change is never the part cut off.
  const window = createMemo(() => {
    const rows = flatRows()
    if (props.expanded) return { start: 0, end: rows.length }
    const kinds = rows.map((entry) => (entry.sep ? "sep" : entry.row.kind))
    return diffWindow(kinds, props.maxRows ?? COLLAPSED_ROWS)
  })
  const visibleRows = createMemo<DisplayRow[]>(() => flatRows().slice(window().start, window().end))
  const rowsAbove = createMemo(() => flatRows().slice(0, window().start).filter((entry) => !entry.sep).length)
  const hiddenRows = createMemo(() => Math.max(0, flatRows().length - window().end))

  const summary = createMemo(() => {
    const { added, removed } = diff()
    if (added > 0 && removed > 0) {
      return { addN: added, delM: removed, both: true }
    }
    return { addN: added, delM: removed, both: false }
  })

  const pad = (n: number) => String(n).padStart(gutterWidth())

  return (
    <box flexDirection="column" flexShrink={0} paddingLeft={2}>
      {/* Connector / summary line */}
      <text>
        <span style={{ fg: UI_PALETTE.textMuted }}>{"⎿ "}</span>
        <Show when={summary().addN > 0}>
          <span style={{ fg: UI_PALETTE.diffAddGutter }}>{`+${summary().addN}`}</span>
          <span style={{ fg: UI_PALETTE.textMuted }}>{` added line${plural(summary().addN)}`}</span>
        </Show>
        <Show when={summary().both}>
          <span style={{ fg: UI_PALETTE.textMuted }}>{", "}</span>
        </Show>
        <Show when={summary().delM > 0}>
          <span style={{ fg: UI_PALETTE.diffDelGutter }}>{`-${summary().delM}`}</span>
          <span style={{ fg: UI_PALETTE.textMuted }}>{` removed line${plural(summary().delM)}`}</span>
        </Show>
      </text>

      {/* Diff body: full-width colored bands, one line-number gutter. */}
      <box flexDirection="column" flexShrink={0}>
        <Show when={rowsAbove() > 0}>
          <text fg={UI_PALETTE.textMuted}>{`  … ${rowsAbove()} unchanged line${plural(rowsAbove())} above`}</text>
        </Show>
        <For each={visibleRows()}>
          {(entry) => {
            if (entry.sep) {
              return (
                <text fg={UI_PALETTE.textMuted}>{`${" ".repeat(gutterWidth())} ⋯`}</text>
              )
            }
            const row = entry.row
            const code = () => row.text.replace(/\t/g, "    ")
            if (row.kind === "add") {
              return (
                <box flexShrink={0} backgroundColor={UI_PALETTE.diffAddBg}>
                  <text>
                    <span style={{ fg: UI_PALETTE.diffAddGutter }}>{`${pad(row.newNo)} + `}</span>
                    <For each={highlightLine(code(), diff().language)}>
                      {(token) => <span style={{ fg: token.color }}>{token.text}</span>}
                    </For>
                  </text>
                </box>
              )
            }
            if (row.kind === "del") {
              return (
                <box flexShrink={0} backgroundColor={UI_PALETTE.diffDelBg}>
                  <text>
                    <span style={{ fg: UI_PALETTE.diffDelGutter }}>{`${pad(row.oldNo)} - `}</span>
                    <span style={{ fg: UI_PALETTE.diffDelFg }}>{code()}</span>
                  </text>
                </box>
              )
            }
            return (
              <box flexShrink={0}>
                <text>
                  <span style={{ fg: UI_PALETTE.textMuted }}>{`${pad(row.newNo)}   `}</span>
                  <For each={highlightLine(code(), diff().language)}>
                    {(token) => <span style={{ fg: token.color }}>{token.text}</span>}
                  </For>
                </text>
              </box>
            )
          }}
        </For>

        <Show when={hiddenRows() > 0}>
          {/* With maxRows (a permission prompt) Ctrl+O does not expand it. */}
          <text fg={UI_PALETTE.textMuted}>
            {`  … +${hiddenRows()} more line${plural(hiddenRows())}${props.maxRows === undefined ? " (Ctrl+O)" : ""}`}
          </text>
        </Show>
        <Show when={props.expanded && diff().truncated}>
          <text fg={UI_PALETTE.textMuted}>{`  … +${diff().truncatedRows} more — open ${diff().path} to view`}</text>
        </Show>
      </box>
    </box>
  )
}
