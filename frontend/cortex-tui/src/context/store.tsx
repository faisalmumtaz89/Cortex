import { batch, createContext, onCleanup, useContext, type ParentProps } from "solid-js"
import { createStore } from "solid-js/store"
import type { EventEnvelope, SessionStatus } from "../protocol"

type MessageRole = "user" | "assistant" | "system"
type ToolState = "pending" | "running" | "completed" | "error"

export type MessagePart =
  | { type: "text"; delta: string }
  | {
      type: "tool"
      call_id: string
      tool: string
      state: ToolState
      ok?: boolean
      output?: string
      error?: string
    }

export interface DownloadProgressRecord {
  kind: "download"
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
  modelLabel?: string
  parentID?: string
  downloadProgress?: DownloadProgressRecord
}

type State = {
  sessionID: string
  status: SessionStatus
  activeModelLabel: string
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
}>()

const initialState: State = {
  sessionID: "",
  status: "idle",
  activeModelLabel: "No model loaded",
  localModelCount: 0,
  messages: {},
  orderedMessageIDs: [],
  seqBySession: {},
  toolDedupe: {},
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
    if (String(payload.kind ?? "") !== "download") {
      return undefined
    }

    const repoID = parseString(payload.repo_id)
    if (!repoID) {
      return undefined
    }

    const bytesDownloaded = parseNonNegativeNumber(payload.bytes_downloaded)
    if (bytesDownloaded === undefined) {
      return undefined
    }

    const phase = parseString(payload.phase) ?? "preparing"
    const parsed: DownloadProgressRecord = {
      kind: "download",
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
    setState("messages", messageID, "parts", (parts) => {
      if (!part.call_id) {
        return [...parts, part]
      }
      const existingIndex = parts.findIndex(
        (candidate) => candidate.type === "tool" && candidate.call_id === part.call_id,
      )
      if (existingIndex < 0) {
        return [...parts, part]
      }
      const next = parts.slice()
      next[existingIndex] = part
      return next
    })
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

    const part: Extract<MessagePart, { type: "tool" }> = {
      type: "tool",
      call_id: callId,
      tool,
      state: status,
      ok: rawPart.ok as boolean | undefined,
      output: rawPart.output as string | undefined,
      error: rawPart.error as string | undefined,
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
        return
      }
      case "permission.replied": {
        setState("pendingPermission", undefined)
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
        const parsedProgress = parseDownloadProgress(event.payload.progress)
        if (parsedProgress) {
          setState("messages", id, "downloadProgress", parsedProgress)
        }
        ingestMessageMetadata(id, event.payload, event.ts_ms, incomingFinal)
        const rawParts = event.payload.parts
        if (Array.isArray(rawParts)) {
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
    const firstLocalEntry = localRaw.find(
      (item) => item && typeof item === "object" && typeof (item as Record<string, unknown>).name === "string",
    ) as Record<string, unknown> | undefined

    const firstLocalName =
      firstLocalEntry && typeof firstLocalEntry.name === "string" ? firstLocalEntry.name.trim() : undefined

    const activeLabel =
      typeof activeTarget.label === "string" && activeTarget.label.trim().length > 0
        ? activeTarget.label.trim()
        : "No model loaded"

    batch(() => {
      setState("activeModelLabel", activeLabel)
      setState("localModelCount", localRaw.length)
      setState("firstLocalModelName", firstLocalName && firstLocalName.length > 0 ? firstLocalName : undefined)
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
