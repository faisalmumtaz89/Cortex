import type { RGBA } from "@opentui/core"
import { For } from "solid-js"
import { wrapText } from "../lib/text_rows"

/** `text` cut into rows of at most `columns` by Cortex (see lib/text_rows),
 * one <text> per row, those a line wraps onto `hang` columns in: the terminal
 * UI never wraps it, so a row breaks only at a space, every character kept. */
export function WrappedText(props: { text: string; columns: number; fg: RGBA; hang?: number }) {
  const rows = () =>
    props.text.split("\n").flatMap((line) => (line ? wrapText(line, props.columns, props.hang ?? 0) : [""]))
  return (
    <box flexDirection="column" flexShrink={0}>
      <For each={rows()}>{(row) => <text fg={props.fg}>{row}</text>}</For>
    </box>
  )
}
