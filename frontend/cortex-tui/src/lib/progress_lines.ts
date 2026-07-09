// Pure builders for the live system-progress indicator lines (model load,
// download, engine update). Extracted from system_message.tsx so the exact
// line composition is unit-testable (tests/progress_lines.test.ts) — the
// component only chooses WHICH builder to render.

import type { DownloadProgressRecord } from "../context/store"
import { spinnerFrame } from "./spinner"

export function formatBytes(value: number | undefined): string {
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

export function formatRate(value: number | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 1) {
    return null
  }
  return `${formatBytes(value)}/s`
}

/** GPU-memory load: exactly one minimal line. */
export function loadIndicatorLine(progress: DownloadProgressRecord): string {
  return `${spinnerFrame()} Loading ${progress.repoID}…`
}

/** Download: one minimal line — transferred bytes (and rate) are the only
 * progress signal worth words. */
export function downloadIndicatorLine(progress: DownloadProgressRecord): string {
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

// Installer output lines can be long; keep the indicator on ONE line.
const MAX_PHASE_DETAIL_CHARS = 64

/** Engine/self update: spinner + the operation's content line ("Updating
 * Lumen…") + the LATEST installer output line as the phase detail — a
 * multi-minute install must visibly progress, not sit on a static label. */
export function engineUpdateIndicatorLine(
  progress: DownloadProgressRecord,
  content: string,
): string {
  const label = content.trim().length > 0 ? content.trim() : `Updating ${progress.repoID}…`
  const parts = [`${spinnerFrame()} ${label}`]
  const phase = progress.phase.trim()
  if (phase.length > 0) {
    parts.push(
      phase.length > MAX_PHASE_DETAIL_CHARS
        ? `${phase.slice(0, MAX_PHASE_DETAIL_CHARS - 1)}…`
        : phase,
    )
  }
  return parts.join(" · ")
}
