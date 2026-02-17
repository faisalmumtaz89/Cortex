import { RGBA } from "@opentui/core"

export const UI_PALETTE = {
  accent: RGBA.fromHex("#10b981"),
  text: RGBA.fromHex("#abb2bf"),
  textMuted: RGBA.fromHex("#5c6370"),
  panel: RGBA.fromHex("#21252b"),
  statusIdle: RGBA.fromHex("#98c379"),
  statusBusy: RGBA.fromHex("#e5c07b"),
  statusRetry: RGBA.fromHex("#c678dd"),
  statusError: RGBA.fromHex("#e06c75"),
  promptBackground: RGBA.fromHex("#000000"),
  promptText: RGBA.fromHex("#ffffff"),
} as const

type SessionStatusColorInput = "idle" | "busy" | "retry" | string
type ToolStateColorInput = "pending" | "running" | "completed" | "error"

export function colorForSessionStatus(status: SessionStatusColorInput): RGBA {
  if (status === "busy") {
    return UI_PALETTE.statusBusy
  }
  if (status === "retry") {
    return UI_PALETTE.statusRetry
  }
  return UI_PALETTE.statusIdle
}

export function colorForToolState(state: ToolStateColorInput): RGBA {
  if (state === "running") {
    return UI_PALETTE.statusBusy
  }
  if (state === "error") {
    return UI_PALETTE.statusError
  }
  return UI_PALETTE.textMuted
}

export function formatTimestamp(tsMs?: number): string {
  if (!tsMs || !Number.isFinite(tsMs)) {
    return ""
  }

  const date = new Date(tsMs)
  if (Number.isNaN(date.getTime())) {
    return ""
  }

  const now = new Date()
  const isSameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()

  if (isSameDay) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  return date.toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function formatDuration(ms?: number): string {
  if (!ms || !Number.isFinite(ms) || ms <= 0) {
    return ""
  }

  if (ms < 1_000) {
    return `${Math.max(1, Math.round(ms))}ms`
  }

  const seconds = Math.round(ms / 100) / 10
  if (seconds < 60) {
    return `${seconds}s`
  }

  const totalSeconds = Math.round(seconds)
  const minutes = Math.floor(totalSeconds / 60)
  const remainder = totalSeconds % 60
  if (remainder === 0) {
    return `${minutes}m`
  }
  return `${minutes}m ${remainder}s`
}

export function titlecase(value?: string): string {
  if (!value) {
    return ""
  }
  const normalized = value.trim()
  if (!normalized) {
    return ""
  }
  return normalized[0].toUpperCase() + normalized.slice(1).toLowerCase()
}
