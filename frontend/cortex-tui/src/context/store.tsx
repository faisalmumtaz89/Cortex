import { batch, createContext, onCleanup, useContext, type ParentProps } from "solid-js"
import { createStore } from "solid-js/store"
import type { EventEnvelope, SessionStatus } from "../protocol"

type MessageRole = "user" | "assistant" | "system"
type ToolState = "pending" | "running" | "completed" | "error"

// A structured file-edit diff (from ToolResult.metadata.diff) for the green/red
// renderer. Mirrors cortex/tooling/builtin/diff_util.py's shape.
export type DiffRow =
  | { kind: "ctx"; text: string; oldNo: number; newNo: number; clipped?: boolean }
  | { kind: "del"; text: string; oldNo: number; clipped?: boolean }
  | { kind: "add"; text: string; newNo: number; clipped?: boolean }
export interface DiffHunk {
  oldStart: number
  newStart: number
  rows: DiffRow[]
}
export interface DiffData {
  path: string
  op: "edit" | "create" | "overwrite"
  language?: string
  added: number
  removed: number
  truncated: boolean
  truncatedRows: number
  hunks: DiffHunk[]
}

export type MessagePart =
  | { type: "text"; delta: string }
  | {
      type: "tool"
      call_id: string
      tool: string
      state: ToolState
      ok?: boolean
      input?: Record<string, unknown>
      output?: string
      error?: string
      diff?: DiffData
      /** Wall-clock ms when a LIVE terminal state was ingested. Absent on
       * historical fills (restored transcripts), letting the UI distinguish a
       * just-finished tool (animate the transition) from an old one (render
       * as-is). */
      completedAtMs?: number
    }

/** Defensively parse metadata.diff into a DiffData, or undefined on any shape
 * mismatch (a malformed diff must never break tool rendering). */
export function parseDiff(metadata: unknown): DiffData | undefined {
  if (!metadata || typeof metadata !== "object") return undefined
  const raw = (metadata as Record<string, unknown>).diff
  if (!raw || typeof raw !== "object") return undefined
  const d = raw as Record<string, unknown>
  const op = d.op
  if ((op !== "edit" && op !== "create" && op !== "overwrite") || !Array.isArray(d.hunks)) {
    return undefined
  }
  const hunks: DiffHunk[] = []
  for (const rawHunk of d.hunks) {
    if (!rawHunk || typeof rawHunk !== "object") continue
    const h = rawHunk as Record<string, unknown>
    if (!Array.isArray(h.rows)) continue
    const rows: DiffRow[] = []
    for (const rawRow of h.rows) {
      if (!rawRow || typeof rawRow !== "object") continue
      const r = rawRow as Record<string, unknown>
      const text = typeof r.text === "string" ? r.text : ""
      const clipped = r.clipped === true
      if (r.kind === "ctx" && typeof r.oldNo === "number" && typeof r.newNo === "number") {
        rows.push({ kind: "ctx", text, oldNo: r.oldNo, newNo: r.newNo, clipped })
      } else if (r.kind === "del" && typeof r.oldNo === "number") {
        rows.push({ kind: "del", text, oldNo: r.oldNo, clipped })
      } else if (r.kind === "add" && typeof r.newNo === "number") {
        rows.push({ kind: "add", text, newNo: r.newNo, clipped })
      }
    }
    if (rows.length > 0) {
      hunks.push({ oldStart: Number(h.oldStart) || 0, newStart: Number(h.newStart) || 0, rows })
    }
  }
  if (hunks.length === 0) return undefined
  return {
    path: String(d.path ?? ""),
    op,
    language: typeof d.language === "string" ? d.language : undefined,
    added: Number(d.added) || 0,
    removed: Number(d.removed) || 0,
    truncated: d.truncated === true,
    truncatedRows: Number(d.truncatedRows) || 0,
    hunks,
  }
}

export interface DownloadProgressRecord {
  kind: "download" | "model-load"
  repoID: string
  phase: string
  bytesDownloaded: number
  bytesTotal?: number
  percent?: number
  filesCompleted?: number
  filesTotal?: number
  speedBps?: number
  etaSeconds?: number
  elapsedSeconds?: number
  stalled: boolean
  /** Client-side: when this operation's first frame arrived (drives the live
   * elapsed display; preserved across update frames). */
}

export interface MessageRecord {
  id: string
  role: MessageRole
  content: string
  final: boolean
  parts: MessagePart[]
  createdTsMs?: number
  completedTsMs?: number
  elapsedMs?: number
  mode?: string
  backend?: string
  modelLabel?: string
  parentID?: string
  downloadProgress?: DownloadProgressRecord
  interrupted?: boolean
}

type State = {
  sessionID: string
  status: SessionStatus
  activeModelLabel: string
  activeBackend?: string
  localModelCount: number
  firstLocalModelName?: string
  error?: string
  pendingPermission?: {
    request_id: string
    permission: string
    patterns: string[]
    metadata: Record<string, unknown>
  }
  messages: Record<string, MessageRecord>
  orderedMessageIDs: string[]
  seqBySession: Record<string, number>
  toolDedupe: Record<string, true>
  busySinceMs?: number
  toolsExpanded: boolean
  queuedInputs: string[]
  lastInterrupted: boolean
  // Slash command palette + ephemeral command result (never enters the transcript).
  paletteOpen: boolean
  commandInFlight: boolean
  commandResult?: { command: string; ok: boolean; text: string; background?: boolean }
  // Full model lists (for the interactive /model picker) + permission-menu highlight.
  models: { local: Record<string, unknown>[]; cloud: Record<string, unknown>[] }
  // Selector of the local model currently downloading in the background (drives
  // the picker's "downloading…" tag); cleared on the final progress frame.
  activeDownloadRepoId?: string
  activeLoadSelector?: string
  permissionChoiceIndex: number
}

const StoreContext = createContext<{
  state: State
  applyEvent: (event: EventEnvelope) => void
  flushPendingEvents: () => void
  appendLocalMessage: (role: MessageRole, content: string) => void
  applySubmitResultFallback: (result: Record<string, unknown>) => void
  setModelStateFromList: (payload: Record<string, unknown>) => void
  setSessionID: (sessionID: string) => void
  setError: (message: string) => void
  clearError: () => void
  clearTranscript: () => void
  toggleToolsExpanded: () => void
  enqueueInput: (text: string) => void
  dequeueInput: () => string | undefined
  setPaletteOpen: (open: boolean) => void
  setPermissionChoice: (index: number) => void
  setCommandResult: (result: { command: string; ok: boolean; text: string; background?: boolean }) => void
  clearCommandResult: () => void
  armCommandInFlight: () => void
  disarmCommandInFlight: () => void
  registerPaletteDismiss: (callback: (() => void) | undefined) => void
  dismissPalette: () => void
}>()

const initialState: State = {
  sessionID: "",
  status: "idle",
  activeModelLabel: "No model loaded",
  activeBackend: undefined,
  localModelCount: 0,
  messages: {},
  orderedMessageIDs: [],
  seqBySession: {},
  toolDedupe: {},
  toolsExpanded: false,
  queuedInputs: [],
  lastInterrupted: false,
  paletteOpen: false,
  commandInFlight: false,
  models: { local: [], cloud: [] },
  activeDownloadRepoId: undefined,
  activeLoadSelector: undefined,
  permissionChoiceIndex: 0,
}

/** Normalize the worker's final part list: chronological text + one entry per
 * tool call (running/completed frames merged, arguments preserved). */
export function normalizeServerParts(rawParts: unknown[]): MessagePart[] {
  const ordered: MessagePart[] = []
  const toolIndexByCall: Record<string, number> = {}

  for (const raw of rawParts) {
    if (!raw || typeof raw !== "object") {
      continue
    }
    const typed = raw as Record<string, unknown>
    const type = String(typed.type ?? "")

    if (type === "text") {
      const delta = String(typed.delta ?? "")
      if (delta) {
        ordered.push({ type: "text", delta })
      }
      continue
    }

    if (type !== "tool") {
      continue
    }

    const part: Extract<MessagePart, { type: "tool" }> = {
      type: "tool",
      call_id: String(typed.call_id ?? ""),
      tool: String(typed.tool ?? "tool"),
      state: String(typed.state ?? "running") as ToolState,
      ok: typed.ok as boolean | undefined,
      input:
        typed.input && typeof typed.input === "object"
          ? (typed.input as Record<string, unknown>)
          : undefined,
      output: typed.output as string | undefined,
      error: typed.error as string | undefined,
      diff: parseDiff(typed.metadata),
    }

    const existingIndex = part.call_id ? toolIndexByCall[part.call_id] : undefined
    if (existingIndex === undefined) {
      if (part.call_id) {
        toolIndexByCall[part.call_id] = ordered.length
      }
      ordered.push(part)
      continue
    }
    const existing = ordered[existingIndex]
    ordered[existingIndex] =
      existing.type === "tool" && existing.input && !part.input
        ? { ...part, input: existing.input }
        : part
  }

  return ordered
}

export function StoreProvider(props: ParentProps) {
  const [state, setState] = createStore<State>(initialState)
  let queue: EventEnvelope[] = []
  let timer: ReturnType<typeof setTimeout> | undefined
  let lastFlushMs = 0

  const flush = () => {
    if (queue.length === 0) {
      timer = undefined
      return
    }
    const pending = queue
    queue = []
    timer = undefined
    lastFlushMs = Date.now()
    batch(() => {
      for (const event of pending) {
        applyOne(event)
      }
    })
  }

  const scheduleFlush = () => {
    if (timer) {
      return
    }
    const elapsed = Date.now() - lastFlushMs
    if (elapsed < 16) {
      timer = setTimeout(flush, 16)
      return
    }
    flush()
  }

  onCleanup(() => {
    if (timer) {
      clearTimeout(timer)
      timer = undefined
    }
  })

  const applyEvent = (event: EventEnvelope) => {
    queue.push(event)
    scheduleFlush()
  }

  const flushPendingEvents = () => {
    flush()
  }

  const parsePositiveInt = (value: unknown): number | undefined => {
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      return undefined
    }
    return Math.round(value)
  }

  const parseNonNegativeNumber = (value: unknown): number | undefined => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      return undefined
    }
    return value
  }

  const parseString = (value: unknown): string | undefined => {
    if (typeof value !== "string") {
      return undefined
    }
    const normalized = value.trim()
    return normalized.length > 0 ? normalized : undefined
  }

  const parseDownloadProgress = (raw: unknown): DownloadProgressRecord | undefined => {
    if (!raw || typeof raw !== "object") {
      return undefined
    }
    const payload = raw as Record<string, unknown>
    const kind = String(payload.kind ?? "")
    if (kind !== "download" && kind !== "model-load") {
      return undefined
    }

    const repoID = parseString(payload.repo_id)
    if (!repoID) {
      return undefined
    }

    // Bytes are download-only; model-load frames carry none (default 0).
    const bytesDownloaded = parseNonNegativeNumber(payload.bytes_downloaded) ?? 0

    const phase = parseString(payload.phase) ?? "preparing"
    const parsed: DownloadProgressRecord = {
      kind: kind as "download" | "model-load",
      repoID,
      phase,
      bytesDownloaded,
      stalled: Boolean(payload.stalled),
    }

    const bytesTotal = parsePositiveInt(payload.bytes_total)
    if (bytesTotal !== undefined) {
      parsed.bytesTotal = bytesTotal
    }
    const percent = parseNonNegativeNumber(payload.percent)
    if (percent !== undefined) {
      parsed.percent = Math.min(100, percent)
    }
    const filesCompleted = parseNonNegativeNumber(payload.files_completed)
    if (filesCompleted !== undefined) {
      parsed.filesCompleted = filesCompleted
    }
    const filesTotal = parseNonNegativeNumber(payload.files_total)
    if (filesTotal !== undefined && filesTotal > 0) {
      parsed.filesTotal = filesTotal
    }
    const speedBps = parseNonNegativeNumber(payload.speed_bps)
    if (speedBps !== undefined && speedBps > 0) {
      parsed.speedBps = speedBps
    }
    const etaSeconds = parseNonNegativeNumber(payload.eta_seconds)
    if (etaSeconds !== undefined && etaSeconds > 0) {
      parsed.etaSeconds = etaSeconds
    }
    const elapsedSeconds = parseNonNegativeNumber(payload.elapsed_seconds)
    if (elapsedSeconds !== undefined) {
      parsed.elapsedSeconds = elapsedSeconds
    }

    return parsed
  }

  const ensureMessage = (id: string, role: MessageRole = "assistant") => {
    if (state.messages[id]) {
      return
    }
    setState("messages", id, {
      id,
      role,
      content: "",
      final: false,
      parts: [],
    })
    setState("orderedMessageIDs", (ids) => [...ids, id])
  }

  const ingestMessageMetadata = (
    messageID: string,
    payload: Record<string, unknown>,
    eventTsMs: number,
    incomingFinal: boolean,
  ) => {
    const createdTsMs = parsePositiveInt(payload.created_ts_ms)
    if (createdTsMs !== undefined) {
      setState("messages", messageID, "createdTsMs", createdTsMs)
    } else if (!state.messages[messageID]?.createdTsMs) {
      setState("messages", messageID, "createdTsMs", eventTsMs)
    }

    const mode = parseString(payload.mode)
    if (mode !== undefined) {
      setState("messages", messageID, "mode", mode)
    }

    const modelLabel = parseString(payload.model_label)
    if (modelLabel !== undefined) {
      setState("messages", messageID, "modelLabel", modelLabel)
    }
    const backend = parseString(payload.backend)
    if (backend !== undefined) {
      setState("messages", messageID, "backend", backend)
    }

    const parentID = parseString(payload.parent_id)
    if (parentID !== undefined) {
      setState("messages", messageID, "parentID", parentID)
    }

    const elapsedMs = parsePositiveInt(payload.elapsed_ms)
    if (elapsedMs !== undefined) {
      setState("messages", messageID, "elapsedMs", elapsedMs)
    }

    const completedTsMs = parsePositiveInt(payload.completed_ts_ms)
    if (completedTsMs !== undefined) {
      setState("messages", messageID, "completedTsMs", completedTsMs)
    } else if (incomingFinal && !state.messages[messageID]?.completedTsMs) {
      setState("messages", messageID, "completedTsMs", eventTsMs)
    }

    if (incomingFinal && !state.messages[messageID]?.elapsedMs) {
      const created = state.messages[messageID]?.createdTsMs
      const completed = state.messages[messageID]?.completedTsMs
      if (created && completed && completed > created) {
        setState("messages", messageID, "elapsedMs", completed - created)
      }
    }
  }

  const upsertToolPart = (messageID: string, part: Extract<MessagePart, { type: "tool" }>) => {
    const parts = state.messages[messageID]?.parts ?? []
    const existingIndex = part.call_id
      ? parts.findIndex(
          (candidate) => candidate.type === "tool" && candidate.call_id === part.call_id,
        )
      : -1
    if (existingIndex < 0) {
      setState("messages", messageID, "parts", (previous) => [...previous, part])
      return
    }
    // Merge IN PLACE (path-level setState) so the part object keeps its
    // identity across state transitions. Replacing the object would remount
    // the tool row on every running→completed flip — restarting its fade-in
    // (visible flicker) and wiping its min-dwell display state.
    const updates: Partial<Extract<MessagePart, { type: "tool" }>> = {
      state: part.state,
      ok: part.ok,
      output: part.output,
      error: part.error,
    }
    // Result updates don't carry the call arguments; keep them from the
    // running-state part so fold summaries stay informative.
    if (part.input) {
      updates.input = part.input
    }
    if (part.diff) {
      updates.diff = part.diff
    }
    if (part.completedAtMs !== undefined) {
      updates.completedAtMs = part.completedAtMs
    }
    setState("messages", messageID, "parts", existingIndex, updates)
  }

  const ingestToolPart = (messageID: string, rawPart: Record<string, unknown>) => {
    const callId = String(rawPart.call_id ?? "")
    const tool = String(rawPart.tool ?? "tool")
    const status = String(rawPart.state ?? "running") as ToolState
    const statusHash = JSON.stringify({
      ok: rawPart.ok,
      output: rawPart.output,
      error: rawPart.error,
    })
    const callKey = callId || `anonymous:${tool}`
    const dedupeKey = `${messageID}:${callKey}:${status}:${statusHash}`
    if (state.toolDedupe[dedupeKey]) {
      return
    }

    const terminal = status === "completed" || status === "error"
    const part: Extract<MessagePart, { type: "tool" }> = {
      type: "tool",
      call_id: callId,
      tool,
      state: status,
      ok: rawPart.ok as boolean | undefined,
      input:
        rawPart.input && typeof rawPart.input === "object"
          ? (rawPart.input as Record<string, unknown>)
          : undefined,
      output: rawPart.output as string | undefined,
      error: rawPart.error as string | undefined,
      diff: parseDiff(rawPart.metadata),
      // Live-ingest timestamp for terminal states (NOT set on historical
      // fills) — lets the UI animate just-finished tools only.
      completedAtMs: terminal ? Date.now() : undefined,
    }
    setState("toolDedupe", dedupeKey, true)
    upsertToolPart(messageID, part)
  }

  const applyOne = (event: EventEnvelope) => {
    const lastSeq = state.seqBySession[event.session_id] ?? 0
    if (event.seq <= lastSeq) {
      return
    }
    setState("seqBySession", event.session_id, event.seq)

    switch (event.event_type) {
      case "session.status": {
        const status = String(event.payload.status ?? "idle") as SessionStatus
        setState("status", status)
        if (status === "busy") {
          if (!state.busySinceMs) {
            setState("busySinceMs", Date.now())
          }
          setState("lastInterrupted", false)
        } else if (status === "idle") {
          setState("busySinceMs", undefined)
          if (event.payload.interrupted === true) {
            setState("lastInterrupted", true)
          }
        }
        return
      }
      case "session.error": {
        setState("error", String(event.payload.error ?? "Unknown error"))
        setState("status", "idle")
        return
      }
      case "system.notice": {
        const message = String(event.payload.message ?? "")
        if (!message) {
          return
        }
        // A command is executing: drop its notice entirely. command.execute
        // returns the authoritative message (set in rpc.tsx), so transcribing
        // the duplicate notice would double it — and it must never reach the
        // scrollable transcript.
        if (state.commandInFlight) {
          return
        }
        const messageID = `system-notice:${event.session_id}:${event.seq}`
        ensureMessage(messageID, "system")
        setState("messages", messageID, "content", message)
        setState("messages", messageID, "final", true)
        setState("messages", messageID, "createdTsMs", event.ts_ms)
        setState("messages", messageID, "completedTsMs", event.ts_ms)
        return
      }
      case "permission.asked": {
        const patternsRaw = event.payload.patterns
        const patterns = Array.isArray(patternsRaw) ? patternsRaw.map((item) => String(item)) : []
        setState("pendingPermission", {
          request_id: String(event.payload.request_id ?? ""),
          permission: String(event.payload.permission ?? ""),
          patterns,
          metadata: (event.payload.metadata as Record<string, unknown>) ?? {},
        })
        setState("permissionChoiceIndex", 0) // default highlight = Allow once
        return
      }
      case "permission.replied": {
        setState("pendingPermission", undefined)
        setState("permissionChoiceIndex", 0)
        return
      }
      case "message.updated": {
        const id = String(event.payload.message_id ?? "")
        if (!id) {
          return
        }
        ensureMessage(id, String(event.payload.role ?? "assistant") as MessageRole)
        const incomingFinal = typeof event.payload.final === "boolean" ? event.payload.final : false

        if (typeof event.payload.content === "string") {
          const nextContent = event.payload.content
          setState("messages", id, "content", (current) => {
            if (nextContent === current) {
              return current
            }
            if (!incomingFinal && nextContent.length < current.length) {
              return current
            }
            return nextContent
          })
        }
        if (typeof event.payload.final === "boolean") {
          setState("messages", id, "final", event.payload.final)
        }
        if (event.payload.interrupted === true) {
          setState("messages", id, "interrupted", true)
        }
        const parsedProgress = parseDownloadProgress(event.payload.progress)
        if (parsedProgress) {
          setState("messages", id, "downloadProgress", parsedProgress)
          if (parsedProgress.kind === "download") {
            setState("activeDownloadRepoId", incomingFinal ? undefined : parsedProgress.repoID)
          } else if (parsedProgress.kind === "model-load") {
            setState("activeLoadSelector", incomingFinal ? undefined : parsedProgress.repoID)
          }
          if (incomingFinal && state.commandResult?.background) {
            // The op's acknowledgment panel ("Loading X…" / "Downloading X…")
            // is only meaningful while the op runs; once the operation's own
            // transcript row resolves (ready OR failed), a lingering panel
            // would contradict it — dismiss automatically.
            setState("commandResult", undefined)
          }
        }
        ingestMessageMetadata(id, event.payload, event.ts_ms, incomingFinal)
        const rawParts = event.payload.parts
        if (Array.isArray(rawParts)) {
          if (incomingFinal && rawParts.length > 0) {
            // The final frame carries the authoritative ordered part list.
            // RECONCILE it into the streamed accumulation instead of replacing
            // the array: replacement would give every part a new identity and
            // remount every tool row at turn end (restarting fades — a visible
            // flicker — and wiping their live completion timestamps). Tools
            // merge in place via upsertToolPart (backfilling any missed field);
            // wholesale fill only when nothing was streamed (e.g. reconnect).
            const existingParts = state.messages[id]?.parts ?? []
            if (existingParts.length === 0) {
              setState("messages", id, "parts", normalizeServerParts(rawParts))
            } else {
              for (const normalized of normalizeServerParts(rawParts)) {
                if (normalized.type === "tool") {
                  upsertToolPart(id, normalized)
                }
              }
            }
          } else {
            for (const rawPart of rawParts) {
              if (!rawPart || typeof rawPart !== "object") {
                continue
              }
              const typed = rawPart as Record<string, unknown>
              if (String(typed.type ?? "") === "tool") {
                ingestToolPart(id, typed)
              }
            }
          }
        }
        return
      }
      case "message.part.updated": {
        const id = String(event.payload.message_id ?? "")
        const rawPart = event.payload.part as Record<string, unknown> | undefined
        if (!id || !rawPart) {
          return
        }
        ensureMessage(id, "assistant")

        const type = String(rawPart.type ?? "")
        if (type === "text") {
          const delta = String(rawPart.delta ?? "")
          if (!delta) {
            return
          }
          setState("messages", id, "content", (current) => `${current}${delta}`)
          // Keep chronology: text lives in parts alongside tool calls so the
          // transcript can render turns in the order they happened.
          setState("messages", id, "parts", (parts) => [...parts, { type: "text" as const, delta }])
          return
        }

        if (type === "tool") {
          ingestToolPart(id, rawPart)
        }
      }
    }
  }

  const setSessionID = (sessionID: string) => {
    setState("sessionID", sessionID)
  }

  const clearTranscript = () => {
    // Reset the on-screen conversation (the worker starts a fresh conversation
    // on /clear); keep model/session identity and dedupe counters harmless.
    setState("messages", {})
    setState("orderedMessageIDs", [])
    setState("toolDedupe", {})
    setState("error", undefined)
    setState("queuedInputs", [])
    setState("lastInterrupted", false)
    // NOTE: intentionally does NOT clear commandResult — /clear must leave its
    // own "Conversation cleared." confirmation panel visible.
  }

  const setPaletteOpen = (open: boolean) => {
    setState("paletteOpen", open)
  }

  const setPermissionChoice = (index: number) => {
    setState("permissionChoiceIndex", index)
  }

  const setCommandResult = (result: { command: string; ok: boolean; text: string; background?: boolean }) => {
    setState("commandResult", result)
  }

  const clearCommandResult = () => {
    setState("commandResult", undefined)
  }

  const armCommandInFlight = () => {
    setState("commandInFlight", true)
  }

  const disarmCommandInFlight = () => {
    setState("commandInFlight", false)
  }

  // session.tsx owns the input ref, so it registers how to dismiss the palette
  // (clear input + local signals); app.tsx's global Esc handler calls it.
  let paletteDismiss: (() => void) | undefined
  const registerPaletteDismiss = (callback: (() => void) | undefined) => {
    paletteDismiss = callback
  }
  const dismissPalette = () => {
    paletteDismiss?.()
    setState("paletteOpen", false)
  }

  const appendLocalMessage = (role: MessageRole, content: string) => {
    const text = content.trim()
    if (!text) {
      return
    }
    const now = Date.now()
    const messageID = `local:${role}:${now}:${Math.random().toString(16).slice(2, 10)}`
    ensureMessage(messageID, role)
    setState("messages", messageID, "content", text)
    setState("messages", messageID, "final", true)
    setState("messages", messageID, "createdTsMs", now)
    setState("messages", messageID, "completedTsMs", now)
  }

  const applySubmitResultFallback = (result: Record<string, unknown>) => {
    const messageID = String(result.assistant_message_id ?? "")
    if (!messageID) {
      return
    }

    const text = typeof result.assistant_text === "string" ? result.assistant_text : ""
    ensureMessage(messageID, "assistant")
    if (text) {
      setState("messages", messageID, "content", (current) => (current.length >= text.length ? current : text))
    }
    setState("messages", messageID, "final", true)
    if (!state.messages[messageID]?.completedTsMs) {
      setState("messages", messageID, "completedTsMs", Date.now())
    }

    const elapsedSeconds = typeof result.elapsed_seconds === "number" ? result.elapsed_seconds : undefined
    if (elapsedSeconds && elapsedSeconds > 0) {
      setState("messages", messageID, "elapsedMs", Math.round(elapsedSeconds * 1000))
    }

    const modelLabel = parseString(result.active_model_label)
    if (modelLabel) {
      setState("messages", messageID, "modelLabel", modelLabel)
    }
  }

  const setModelStateFromList = (payload: Record<string, unknown>) => {
    const activeTargetRaw = payload.active_target
    const activeTarget =
      activeTargetRaw && typeof activeTargetRaw === "object"
        ? (activeTargetRaw as Record<string, unknown>)
        : {}

    const localRaw = Array.isArray(payload.local) ? payload.local : []
    const cloudRaw = Array.isArray(payload.cloud) ? payload.cloud : []
    const firstLocalEntry = localRaw.find(
      (item) => item && typeof item === "object" && typeof (item as Record<string, unknown>).name === "string",
    ) as Record<string, unknown> | undefined

    const firstLocalName =
      firstLocalEntry && typeof firstLocalEntry.name === "string" ? firstLocalEntry.name.trim() : undefined

    const activeLabel =
      typeof activeTarget.label === "string" && activeTarget.label.trim().length > 0
        ? activeTarget.label.trim()
        : "No model loaded"

    const asRecords = (list: unknown[]): Record<string, unknown>[] =>
      list.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")

    const activeBackend =
      typeof activeTarget.backend === "string" && activeTarget.backend.trim().length > 0
        ? activeTarget.backend.trim()
        : undefined

    batch(() => {
      setState("activeModelLabel", activeLabel)
      setState("activeBackend", activeLabel === "No model loaded" ? undefined : activeBackend)
      setState("localModelCount", localRaw.length)
      setState("firstLocalModelName", firstLocalName && firstLocalName.length > 0 ? firstLocalName : undefined)
      setState("models", { local: asRecords(localRaw), cloud: asRecords(cloudRaw) })
    })
  }

  const setError = (message: string) => {
    const normalized = message.trim()
    setState("error", normalized ? normalized : undefined)
    setState("status", "idle")
  }

  const clearError = () => {
    setState("error", undefined)
  }

  const toggleToolsExpanded = () => {
    setState("toolsExpanded", !state.toolsExpanded)
  }

  const enqueueInput = (text: string) => {
    setState("queuedInputs", [...state.queuedInputs, text])
  }

  const dequeueInput = (): string | undefined => {
    const [next, ...rest] = state.queuedInputs
    if (next === undefined) {
      return undefined
    }
    setState("queuedInputs", rest)
    return next
  }

  const value = {
      state,
      applyEvent,
      flushPendingEvents,
      appendLocalMessage,
      applySubmitResultFallback,
      setModelStateFromList,
      setSessionID,
      setError,
      clearError,
      clearTranscript,
      toggleToolsExpanded,
      enqueueInput,
      dequeueInput,
      setPaletteOpen,
      setPermissionChoice,
      setCommandResult,
      clearCommandResult,
      armCommandInFlight,
      disarmCommandInFlight,
      registerPaletteDismiss,
      dismissPalette,
  }

  return <StoreContext.Provider value={value}>{props.children}</StoreContext.Provider>
}

export function useStore() {
  const ctx = useContext(StoreContext)
  if (!ctx) {
    throw new Error("useStore must be used inside StoreProvider")
  }
  return ctx
}
