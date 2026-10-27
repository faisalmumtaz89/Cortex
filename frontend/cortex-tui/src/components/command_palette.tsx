import { For, Show } from "solid-js"
import type { SlashCommand } from "../commands"
import { terminalColumns, terminalRows } from "../lib/terminal_size"
import { SelectionRows, padPrimary } from "./selection_list"
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

/** A description cut to the row, so a skill's long one stays on one line. */
function fitDescription(command: SlashCommand): string {
  const room = Math.max(10, terminalColumns() - padPrimary(`/${command.name}`).length - 8)
  const text = command.description.replace(/\s+/g, " ")
  return text.length > room ? `${text.slice(0, room - 1)}…` : text
}

/** Command list / arg-hint panel shown above the prompt while typing a slash
 * command. The list body reuses the shared SelectionRows so the palette,
 * permission menu, and model picker are visually identical. */
export function CommandPalette(props: {
  matches: SlashCommand[]
  selectedIndex: number
  argMode: boolean
  argCommand?: SlashCommand
}) {
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
            getPrimary={(command) => padPrimary(`/${command.name}`)}
            getSecondary={(command) => fitDescription(command)}
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

const RESULT_MAX_ROWS = 24
// Rows the rest of the screen keeps while a result shows: header, a few
// transcript rows, the prompt and its footer.
const RESULT_RESERVED_ROWS = 14

export function CommandResultPanel(props: { command: string; ok: boolean; text: string }) {
  // Budgeted in screen rows, wrapped lines counted as the rows they take: a
  // taller panel pushes the prompt off screen and squeezes the transcript.
  const visible = () => {
    const lines = props.text.split("\n")
    const width = Math.max(10, terminalColumns() - 8) // margin, spine, padding
    let budget = Math.max(3, Math.min(RESULT_MAX_ROWS, terminalRows() - RESULT_RESERVED_ROWS))
    const shown: string[] = []
    for (const line of lines) {
      const rows = Math.max(1, Math.ceil(line.length / width))
      if (rows > budget) break
      shown.push(line)
      budget -= rows
    }
    if (shown.length < lines.length && shown.length > 0 && budget === 0) shown.pop()
    return { shown, overflow: lines.length - shown.length }
  }
  const shown = () => visible().shown
  const overflow = () => visible().overflow

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
      <Show when={props.text.trim().length > 0}>
        {/* Blank row between the ✓/✕ header and the body. */}
        <box flexDirection="column" flexShrink={0} marginTop={1}>
          <For each={shown()}>{(line) => <text fg={UI_PALETTE.text}>{line}</text>}</For>
          <Show when={overflow() > 0}>
            <text fg={UI_PALETTE.textMuted}>{`… +${overflow()} more lines: make the window taller to see them`}</text>
          </Show>
        </box>
      </Show>
    </box>
  )
}
