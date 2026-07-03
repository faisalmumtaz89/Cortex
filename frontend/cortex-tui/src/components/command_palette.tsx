import { For, Show } from "solid-js"
import type { SlashCommand } from "../commands"
import { terminalColumns } from "../lib/terminal_size"
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
            getSecondary={(command) => command.description}
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

const RESULT_MAX_ROWS = 16

export function CommandResultPanel(props: { command: string; ok: boolean; text: string }) {
  const lines = () => props.text.split("\n")
  const shown = () => lines().slice(0, RESULT_MAX_ROWS)
  const overflow = () => Math.max(0, lines().length - RESULT_MAX_ROWS)

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
            <text fg={UI_PALETTE.textMuted}>{`… (+${overflow()} lines)`}</text>
          </Show>
        </box>
      </Show>
    </box>
  )
}
