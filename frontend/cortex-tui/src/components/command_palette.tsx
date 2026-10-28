import { For, Show } from "solid-js"
import type { SlashCommand } from "../commands"
import type { ResultLayout } from "../lib/result_layout"
import { terminalColumns } from "../lib/terminal_size"
import { SelectionRows, listWindow, nameColumn, padPrimary } from "./selection_list"
import { UI_PALETTE } from "./ui_palette"

const SPINE_BORDER = {
  topLeft: "",
  bottomLeft: "",
  vertical: "┃",
  topRight: "",
  bottomRight: "",
  horizontal: " ",
  bottomT: "",
  topT: "",
  cross: "",
  leftT: "",
  rightT: "",
}

// Columns a palette row loses besides its name: the screen's side padding
// (2 + 2), the panel's rail and padding (1 + 2 + 1), and the "❯ " marker (2).
const ROW_CHROME = 10
// The palette's footer and the blank row above it.
const FOOTER_ROWS = 2

/** A description cut to the row, so a skill's long one stays on one line. */
function fitDescription(command: SlashCommand, column: number): string {
  const room = Math.max(10, terminalColumns() - padPrimary(`/${command.name}`, column).length - ROW_CHROME)
  const text = command.description.replace(/\s+/g, " ")
  return text.length > room ? `${text.slice(0, room - 1)}…` : text
}

/** Command list / arg-hint panel shown above the prompt while typing a slash
 * command, in at most `room` rows. The list body reuses the shared
 * SelectionRows so the palette, permission menu, and model picker are
 * visually identical. */
export function CommandPalette(props: {
  matches: SlashCommand[]
  selectedIndex: number
  argMode: boolean
  argCommand?: SlashCommand
  room: number
}) {
  const column = () => nameColumn(props.matches.map((command) => `/${command.name}`))
  return (
    <box
      flexDirection="column"
      flexShrink={0}
      backgroundColor={UI_PALETTE.panel}
      paddingLeft={2}
      paddingRight={1}
      border={["left"]}
      borderColor={UI_PALETTE.accent}
      customBorderChars={SPINE_BORDER}
    >
      <Show
        when={!props.argMode}
        fallback={
          <box flexDirection="column" flexShrink={0}>
            <text>
              <span style={{ fg: UI_PALETTE.accent }}>{`/${props.argCommand?.name ?? ""} `}</span>
              <span style={{ fg: UI_PALETTE.text }}>{props.argCommand?.argHint ?? ""}</span>
            </text>
            <text fg={UI_PALETTE.textMuted}>{props.argCommand?.description ?? ""}</text>
          </box>
        }
      >
        <Show
          when={props.matches.length > 0}
          fallback={<text fg={UI_PALETTE.textMuted}>No matching command — Enter to try anyway, Esc to cancel</text>}
        >
          <SelectionRows
            items={props.matches}
            selectedIndex={props.selectedIndex}
            getPrimary={(command) => padPrimary(`/${command.name}`, column())}
            getSecondary={(command) => fitDescription(command, column())}
            maxVisibleRows={listWindow(props.room, FOOTER_ROWS, props.matches.length)}
          />
        </Show>
      </Show>
      {/* Footer is mode-accurate (Tab/↑↓ don't apply in arg-hint mode) and
          width-adaptive so it never wraps mid-word on narrow terminals. One
          blank row above it keeps the list from feeling packed. */}
      <box flexShrink={0} marginTop={1}>
        <text fg={UI_PALETTE.textMuted}>
          {props.argMode
            ? "Enter run · Esc cancel"
            : terminalColumns() < 64
              ? "↑↓ · Tab · Enter run · Esc"
              : "↑↓ select · Tab complete · Enter run · Esc cancel"}
        </text>
      </box>
    </box>
  )
}

export function CommandResultPanel(props: { command: string; ok: boolean; layout: ResultLayout }) {
  const shown = () => props.layout.rows.slice(props.layout.offset, props.layout.offset + props.layout.window)
  const scrolls = () => props.layout.window < props.layout.rows.length

  return (
    <box
      flexDirection="column"
      flexShrink={0}
      backgroundColor={UI_PALETTE.panel}
      paddingLeft={2}
      paddingRight={1}
      border={["left"]}
      borderColor={props.ok ? UI_PALETTE.accent : UI_PALETTE.statusError}
      customBorderChars={SPINE_BORDER}
    >
      <box flexDirection="row" justifyContent="space-between" flexShrink={0}>
        <text>
          <span style={{ fg: props.ok ? UI_PALETTE.accent : UI_PALETTE.statusError }}>
            {props.ok ? "✓ " : "✕ "}
          </span>
          <span style={{ fg: UI_PALETTE.text }}>{props.command}</span>
        </text>
        <text fg={UI_PALETTE.textMuted}>Esc to dismiss</text>
      </box>
      <Show when={props.layout.rows.some((row) => row.trim())}>
        {/* Blank row between the ✓/✕ header and the body. */}
        <box flexDirection="column" flexShrink={0} marginTop={1}>
          <For each={shown()}>{(line) => <text fg={UI_PALETTE.text}>{line}</text>}</For>
          <Show when={scrolls()}>
            <text fg={UI_PALETTE.textMuted}>
              {`rows ${props.layout.offset + 1}–${props.layout.offset + props.layout.window} of ${props.layout.rows.length} · PgUp/PgDn, Home/End`}
            </text>
          </Show>
        </box>
      </Show>
    </box>
  )
}
