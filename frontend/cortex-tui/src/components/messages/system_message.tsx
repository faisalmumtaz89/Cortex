import type { MessageRecord } from "../../context/store"
import { engineUpdateIndicatorLine } from "../../lib/progress_lines"
import { UI_PALETTE } from "../ui_palette"

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

const TERMINAL_PHASES = new Set(["ready", "failed"])

export function SystemMessage(props: { message: MessageRecord; index: number }) {
  // Everything below is a reactive accessor ON PURPOSE: component bodies run
  // once in Solid, so plain consts here would freeze the live/resolved gating
  // at mount, and a live row would keep spinning after its final frame.
  const progress = () => props.message.progress
  const phase = () => progress()?.phase?.trim().toLowerCase() ?? ""
  const isTerminalPhase = () => TERMINAL_PHASES.has(phase())
  const showLiveProgress = () =>
    Boolean(progress()) && !props.message.final && !isTerminalPhase()
  // While an operation is live, the scenario-specific indicator IS the
  // message; the plain content line takes over on the final frame.
  const showPrimaryContent = () =>
    typeof props.message.content === "string" &&
    props.message.content.length > 0 &&
    !showLiveProgress()
  const terminalFallbackMessage = () => {
    const record = progress()
    if (!record || !(props.message.final || isTerminalPhase()) || showPrimaryContent()) {
      return null
    }
    if (phase() === "failed") {
      return `Failed: ${record.repoID}`
    }
    return `Done: ${record.repoID}`
  }

  if (!props.message.content && !progress()) {
    return <></>
  }

  // Quiet by design: a dim spine + muted text on the shared rail, no "System"
  // label, no filled card. System notices should recede, not compete.
  return (
    <box
      flexDirection="column"
      flexShrink={0}
      marginTop={props.index === 0 ? 0 : 1}
      border={["left"]}
      borderColor={UI_PALETTE.textMuted}
      customBorderChars={SPINE_BORDER}
    >
      <box flexShrink={0} paddingLeft={2}>
        {showPrimaryContent() && <text fg={UI_PALETTE.textMuted}>{props.message.content}</text>}
        {!showPrimaryContent() && terminalFallbackMessage() && (
          <text fg={UI_PALETTE.textMuted}>{terminalFallbackMessage()}</text>
        )}
        {showLiveProgress() && (
          <text fg={UI_PALETTE.textMuted}>
            {engineUpdateIndicatorLine(progress()!, props.message.content)}
          </text>
        )}
      </box>
    </box>
  )
}
