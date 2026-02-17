import type { KeyBinding, KeyEvent, RGBA, TextareaRenderable } from "@opentui/core"
import { Show } from "solid-js"
import { UI_PALETTE } from "./ui_palette"

const PROMPT_KEYBINDINGS: KeyBinding[] = [
  { name: "return", action: "submit" },
  { name: "return", shift: true, action: "newline" },
]
const PROMPT_TOP_SPACER_ROWS = 2
const PROMPT_BOTTOM_SPACER_ROWS = 1

const EMPTY_BORDER = {
  topLeft: "",
  bottomLeft: "",
  vertical: "",
  topRight: "",
  bottomRight: "",
  horizontal: " ",
  bottomT: "",
  topT: "",
  cross: "",
  leftT: "",
  rightT: "",
}

export function PromptPanel(props: {
  hasPendingPermission: boolean
  status: string
  statusColor: RGBA
  onPromptKeyDown: (event: KeyEvent) => void
  onSubmit: () => void
  setInputRef: (input: TextareaRenderable) => void
}) {
  return (
    <box flexShrink={0}>
      <box
        border={["left"]}
        borderColor={UI_PALETTE.accent}
        customBorderChars={{
          ...EMPTY_BORDER,
          vertical: "┃",
          bottomLeft: "╹",
        }}
      >
        <box
          paddingLeft={2}
          paddingRight={2}
          flexShrink={0}
          backgroundColor={UI_PALETTE.promptBackground}
          flexGrow={1}
        >
          <box flexDirection="column" flexShrink={0}>
            <box height={PROMPT_TOP_SPACER_ROWS} />
            <textarea
              ref={(r: TextareaRenderable) => {
                props.setInputRef(r)
              }}
              placeholder={props.hasPendingPermission ? "Permission modal active..." : "Ask Cortex..."}
              minHeight={1}
              maxHeight={6}
              textColor={UI_PALETTE.promptText}
              focusedTextColor={UI_PALETTE.promptText}
              cursorColor={UI_PALETTE.promptText}
              focusedBackgroundColor={UI_PALETTE.promptBackground}
              keyBindings={PROMPT_KEYBINDINGS}
              onKeyDown={props.onPromptKeyDown}
              onSubmit={props.onSubmit}
            />
            <box height={PROMPT_BOTTOM_SPACER_ROWS} />
          </box>
        </box>
      </box>
      <box
        height={1}
        border={["left"]}
        borderColor={UI_PALETTE.accent}
        customBorderChars={{
          ...EMPTY_BORDER,
          vertical: "╹",
        }}
      >
        <box
          height={1}
          border={["bottom"]}
          borderColor={UI_PALETTE.promptBackground}
          customBorderChars={{
            ...EMPTY_BORDER,
            horizontal: "▀",
          }}
        />
      </box>
      <box flexDirection="row" justifyContent="space-between">
        <text fg={UI_PALETTE.textMuted}>Enter submit · Esc reject permission</text>
        <Show when={props.status !== "idle"}>
          <text fg={props.statusColor}>{props.status}</text>
        </Show>
      </box>
    </box>
  )
}
