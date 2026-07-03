import type { KeyBinding, KeyEvent, RGBA, TextareaRenderable } from "@opentui/core"
import { Show } from "solid-js"
import { terminalColumns } from "../lib/terminal_size"
import { UI_PALETTE } from "./ui_palette"

// Hint segments in display order; `drop` is the order they are shed when width
// is tight (lowest first). "Enter send" survives longest (primary action).
const HINT_SEGMENTS: { text: string; drop: number }[] = [
  { text: "Enter send", drop: 3 },
  { text: "Shift+Enter newline", drop: 1 },
  { text: "/help commands", drop: 0 },
  { text: "Ctrl+C quit", drop: 2 },
]

/** Build the widest hint string that fits `available` columns, dropping the
 * least-important segments first so it never collides with the status. */
function buildHint(available: number): string {
  const dropOrder = [...HINT_SEGMENTS].sort((a, b) => a.drop - b.drop)
  const removed = new Set<string>()
  const render = () =>
    HINT_SEGMENTS.filter((segment) => !removed.has(segment.text))
      .map((segment) => segment.text)
      .join(" · ")
  for (const candidate of dropOrder) {
    if (render().length <= available) {
      break
    }
    removed.add(candidate.text)
  }
  const result = render()
  return result.length <= available ? result : ""
}

const PROMPT_KEYBINDINGS: KeyBinding[] = [
  { name: "return", action: "submit" },
  { name: "return", shift: true, action: "newline" },
  { name: "enter", action: "submit" },
  { name: "enter", shift: true, action: "newline" },
  { name: "kpenter", action: "submit" },
  { name: "kpenter", shift: true, action: "newline" },
  { name: "numpadenter", action: "submit" },
  { name: "numpadenter", shift: true, action: "newline" },
  { name: "linefeed", action: "submit" },
  { name: "linefeed", shift: true, action: "newline" },
]
// Symmetric padding rows above and below the text: the input band is exactly
// pad / text / pad, so the cursor sits perfectly centered vertically. The
// band's dark-gray background provides its own edges (no half-block floor row).
const PROMPT_TOP_SPACER_ROWS = 1
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
  onContentChange: () => void
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
              backgroundColor={UI_PALETTE.promptBackground}
              focusedBackgroundColor={UI_PALETTE.promptBackground}
              keyBindings={PROMPT_KEYBINDINGS}
              onKeyDown={props.onPromptKeyDown}
              onContentChange={props.onContentChange}
              onSubmit={props.onSubmit}
            />
            <box height={PROMPT_BOTTOM_SPACER_ROWS} />
          </box>
        </box>
      </box>
      {/* One blank row between the input band and the hint line. Adaptive
          hint: sheds segments so it never overlaps the status even at narrow
          widths. Reactive to terminalColumns() for live resize. */}
      <box flexDirection="row" justifyContent="space-between" marginTop={1}>
        <text fg={UI_PALETTE.textMuted}>
          {buildHint(
            terminalColumns() - 4 - (props.status !== "idle" ? props.status.length + 2 : 0),
          )}
        </text>
        <Show when={props.status !== "idle"}>
          <text fg={props.statusColor}>{props.status}</text>
        </Show>
      </box>
    </box>
  )
}
