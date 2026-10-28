import { For, Show } from "solid-js"
import type { PromptLayout } from "../lib/permission_prompt"
import type { PromptRow, PromptTone } from "../lib/text_rows"
import { UI_PALETTE } from "./ui_palette"

const TONES: Record<PromptTone, keyof typeof UI_PALETTE> = {
  text: "text",
  muted: "textMuted",
  addGutter: "diffAddGutter",
  delGutter: "diffDelGutter",
  delText: "diffDelFg",
}

function Row(props: { row: PromptRow }) {
  const band = () =>
    props.row.band === "add" ? UI_PALETTE.diffAddBg : props.row.band === "del" ? UI_PALETTE.diffDelBg : undefined
  return (
    <box flexShrink={0} backgroundColor={band()}>
      <text>
        <For each={props.row.spans}>
          {(span) => <span style={{ fg: span.color ?? UI_PALETTE[TONES[span.tone]] }}>{span.text}</span>}
        </For>
      </text>
    </box>
  )
}

/** The call a permission prompt asks about: the pinned rows, then the
 * window of the call's rows, an edit's diff indented under its file. */
export function PermissionCall(props: { layout: PromptLayout; diff: boolean }) {
  const shown = () => props.layout.rows.slice(props.layout.offset, props.layout.offset + props.layout.window)
  return (
    <box flexDirection="column" flexShrink={0}>
      <For each={props.layout.pinned}>{(row) => <Row row={row} />}</For>
      <box flexDirection="column" flexShrink={0} paddingLeft={props.diff ? 2 : 0}>
        <For each={shown()}>{(row) => <Row row={row} />}</For>
      </box>
    </box>
  )
}

/** The row under the call: where the window is, when the call has more
 * rows than it shows; else blank. */
export function PermissionPosition(props: { layout: PromptLayout }) {
  const layout = () => props.layout
  return (
    <Show when={layout().window < layout().rows.length} fallback={<text> </text>}>
      <text fg={UI_PALETTE.textMuted}>
        {`rows ${layout().offset + 1}–${layout().offset + layout().window} of ${layout().rows.length} · PgUp/PgDn, Home/End`}
      </text>
    </Show>
  )
}
