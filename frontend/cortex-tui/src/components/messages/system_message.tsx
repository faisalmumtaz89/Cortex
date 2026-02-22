import type { DownloadProgressRecord, MessageRecord } from "../../context/store"
import { UI_PALETTE } from "../ui_palette"

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

function formatETA(value: number | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null
  }
  const rounded = Math.max(1, Math.round(value))
  if (rounded < 60) {
    return `${rounded}s`
  }
  const minutes = Math.floor(rounded / 60)
  const seconds = rounded % 60
  if (minutes < 60) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`
  }
  const hours = Math.floor(minutes / 60)
  const remainderMinutes = minutes % 60
  return `${hours}h ${String(remainderMinutes).padStart(2, "0")}m`
}

function progressBar(percent: number | undefined, width = 24): string {
  const normalized = typeof percent === "number" && Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0
  const filled = Math.round((normalized / 100) * width)
  const left = "#".repeat(Math.max(filled, 0))
  const right = "-".repeat(Math.max(width - filled, 0))
  return `[${left}${right}]`
}

function progressHeadline(progress: DownloadProgressRecord): string {
  const percent = typeof progress.percent === "number" && Number.isFinite(progress.percent) ? progress.percent : undefined
  const phase = progress.phase.trim().toLowerCase()
  if (phase === "finalizing") {
    return `${progressBar(100)} 100.0% (finalizing)`
  }
  if (percent !== undefined) {
    return `${progressBar(percent)} ${percent.toFixed(1)}%`
  }
  return `${progressBar(undefined)} preparing`
}

function progressDetail(progress: DownloadProgressRecord): string {
  const parts: string[] = []
  const downloaded = formatBytes(progress.bytesDownloaded)
  if (typeof progress.bytesTotal === "number" && progress.bytesTotal > 0) {
    parts.push(`${downloaded} / ${formatBytes(progress.bytesTotal)}`)
  } else {
    parts.push(`${downloaded} downloaded`)
  }
  const rate = formatRate(progress.speedBps)
  if (rate) {
    parts.push(rate)
  }
  const phase = progress.phase.trim().toLowerCase()
  const transferComplete =
    typeof progress.bytesTotal === "number" &&
    progress.bytesTotal > 0 &&
    progress.bytesDownloaded >= progress.bytesTotal
  const eta = formatETA(progress.etaSeconds)
  if (eta && phase !== "finalizing" && !transferComplete) {
    parts.push(`ETA ${eta}`)
  }
  return parts.join(" | ")
}

function progressFooter(progress: DownloadProgressRecord): string | null {
  const parts: string[] = []
  if (typeof progress.filesCompleted === "number" && typeof progress.filesTotal === "number" && progress.filesTotal > 0) {
    parts.push(`${progress.filesCompleted.toFixed(1)}/${progress.filesTotal.toFixed(1)} files`)
  }
  if (progress.stalled) {
    parts.push("waiting for transfer")
  }
  if (parts.length === 0) {
    return null
  }
  return parts.join(" | ")
}

export function SystemMessage(props: { message: MessageRecord; index: number }) {
  const progress = props.message.downloadProgress
  const footer = progress ? progressFooter(progress) : null
  const phase = progress?.phase?.trim().toLowerCase() ?? ""
  const hasProgress = Boolean(progress)
  const isTerminalPhase = phase === "completed" || phase === "failed" || phase === "cancelled"
  const showLiveProgress = hasProgress && !props.message.final && !isTerminalPhase
  const showPrimaryContent =
    typeof props.message.content === "string" &&
    props.message.content.length > 0 &&
    (
      !showLiveProgress ||
      props.message.final ||
      phase === "completed" ||
      phase === "failed" ||
      phase === "cancelled"
    )
  const terminalFallbackMessage =
    progress && (props.message.final || isTerminalPhase)
      ? phase === "completed"
        ? `Download complete: ${progress.repoID}`
        : phase === "failed"
          ? `Download failed: ${progress.repoID}`
          : phase === "cancelled"
            ? `Download cancelled: ${progress.repoID}`
            : "Download update complete."
      : null

  if (!props.message.content && !progress) {
    return <></>
  }

  return (
    <box
      flexDirection="column"
      marginTop={props.index === 0 ? 0 : 1}
      border={["left"]}
      borderColor={UI_PALETTE.textMuted}
      paddingLeft={2}
      paddingTop={1}
      paddingBottom={1}
      backgroundColor={UI_PALETTE.panel}
    >
      <text fg={UI_PALETTE.textMuted}>System</text>
      {showPrimaryContent && <text fg={UI_PALETTE.textMuted}>{props.message.content}</text>}
      {!showPrimaryContent && terminalFallbackMessage && (
        <text fg={UI_PALETTE.textMuted}>{terminalFallbackMessage}</text>
      )}
      {showLiveProgress && progress && <text fg={UI_PALETTE.textMuted}>{progressHeadline(progress)}</text>}
      {showLiveProgress && progress && <text fg={UI_PALETTE.textMuted}>{progressDetail(progress)}</text>}
      {showLiveProgress && footer && <text fg={UI_PALETTE.textMuted}>{footer}</text>}
    </box>
  )
}
