import type { DownloadProgressRecord, MessageRecord } from "../../context/store"
import { spinnerFrame } from "../../lib/spinner"
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

function formatBytes(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return "0 B"
  }
  const units = ["B", "KB", "MB", "GB", "TB"]
  let amount = value
  let index = 0
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024
    index += 1
  }
  if (index === 0) {
    return `${Math.round(amount)} ${units[index]}`
  }
  return `${amount.toFixed(1)} ${units[index]}`
}

function formatRate(value: number | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 1) {
    return null
  }
  return `${formatBytes(value)}/s`
}

const TERMINAL_PHASES = new Set(["complete", "completed", "ready", "failed", "cancelled"])

/** GPU-memory load: exactly one minimal line. */
function loadIndicatorLine(progress: DownloadProgressRecord): string {
  return `${spinnerFrame()} Loading ${progress.repoID}…`
}

/** Download: one minimal line — transferred bytes (and rate) are the only
 * progress signal worth words. */
function downloadIndicatorLine(progress: DownloadProgressRecord): string {
  const parts: string[] = [`${spinnerFrame()} Downloading ${progress.repoID}…`]
  if (progress.bytesDownloaded > 0) {
    let bytesPart = formatBytes(progress.bytesDownloaded)
    if (typeof progress.bytesTotal === "number" && progress.bytesTotal > 0) {
      bytesPart += ` / ${formatBytes(progress.bytesTotal)}`
    }
    const rate = formatRate(progress.speedBps)
    if (rate) {
      bytesPart += ` (${rate})`
    }
    parts.push(bytesPart)
  }
  return parts.join(" · ")
}

export function SystemMessage(props: { message: MessageRecord; index: number }) {
  // Everything below is a reactive accessor ON PURPOSE: component bodies run
  // once in Solid, so plain consts here would freeze the live/resolved gating
  // at mount — the exact bug where a "Loading …" row kept spinning forever
  // after its final frame had already arrived.
  const progress = () => props.message.downloadProgress
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
    if (phase() === "cancelled") {
      return `Cancelled: ${record.repoID}`
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
        {showLiveProgress() && progress()?.kind === "model-load" && (
          <text fg={UI_PALETTE.textMuted}>{loadIndicatorLine(progress()!)}</text>
        )}
        {showLiveProgress() && progress()?.kind === "download" && (
          <text fg={UI_PALETTE.textMuted}>{downloadIndicatorLine(progress()!)}</text>
        )}
      </box>
    </box>
  )
}
