import { MacOSScrollAccel, SyntaxStyle, type KeyEvent, type ScrollAcceleration, type ScrollBoxRenderable, type TextareaRenderable } from "@opentui/core"
import { For, Match, Show, Switch, createEffect, createMemo, createSignal, on, onCleanup } from "solid-js"
import { PromptPanel } from "../components/prompt_panel"
import { CommandPalette, CommandResultPanel } from "../components/command_palette"
import { SelectionList, SelectionRows, classifySelectionKey, type SelectionTag } from "../components/selection_list"
import { AssistantMessage } from "../components/messages/assistant_message"
import { SystemMessage } from "../components/messages/system_message"
import { UserMessage } from "../components/messages/user_message"
import { UI_PALETTE, colorForSessionStatus } from "../components/ui_palette"
import {
  commandTakesArgs,
  commandToken,
  findCommand,
  matchCommands,
  type SlashCommand,
} from "../commands"
import { useStore } from "../context/store"
import { readGitBranch } from "../lib/git_branch"
import { displayPath } from "../lib/paths"
import { spinnerFrame } from "../lib/spinner"
import { terminalColumns } from "../lib/terminal_size"

class CustomSpeedScroll implements ScrollAcceleration {
  constructor(private speed: number) {}

  tick(): number {
    return this.speed
  }

  reset(): void {}
}

const SLASH_COMMAND_ALIASES = new Set([
  "help",
  "status",
  "gpu",
  "model",
  "download",
  "login",
  "clear",
  "save",
  "benchmark",
  "template",
  "quit",
  "exit",
  "setup",
])

function isSlashCommandLike(raw: string): boolean {
  const trimmed = raw.trim()
  if (!trimmed) {
    return false
  }
  if (trimmed.startsWith("/")) {
    return true
  }
  const firstWord = trimmed.split(/\s+/, 1)[0]?.toLowerCase() || ""
  return SLASH_COMMAND_ALIASES.has(firstWord)
}

// Thick "┃" left rail used by every overlay (palette, result, permission, picker).
const OVERLAY_BORDER = {
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

// Permission choices, in highlight order; index maps to the reply string.
const PERMISSION_OPTIONS: { label: string; danger?: boolean }[] = [
  { label: "Allow once" },
  { label: "Allow always" },
  { label: "Reject", danger: true },
]
const PERMISSION_REPLIES = ["allow_once", "allow_always", "reject"] as const

/** Shorten a path to fit `maxLen`, keeping the first + last segments with a
 * "…/" in the middle so it never wraps the header. */
function truncatePathMiddle(path: string, maxLen: number): string {
  if (path.length <= maxLen) {
    return path
  }
  const parts = path.split("/")
  if (parts.length > 2) {
    const head = parts[0] // "~" for a home path, "" for an absolute root
    const tail = parts.slice(-2).join("/")
    const withTwo = head ? `${head}/…/${tail}` : `…/${tail}`
    if (withTwo.length <= maxLen) {
      return withTwo
    }
    const withOne = `…/${parts[parts.length - 1]}`
    if (withOne.length <= maxLen) {
      return withOne
    }
  }
  return `…${path.slice(path.length - maxLen + 1)}`
}

export function SessionRoute(props: {
  onSubmit: (value: string) => Promise<boolean>
  onInterrupt: () => void
}) {
  const store = useStore()
  let input: TextareaRenderable | undefined
  let transcriptScroll: ScrollBoxRenderable | undefined
  let lastSubmitFingerprint = ""
  let lastSubmitAtMs = 0
  const [isSubmitting, setIsSubmitting] = createSignal(false)
  let lastAssistantMessageID = ""
  let sawFirstAssistantDelta = false
  const debugLayout = process.env.CORTEX_TUI_DEBUG_LAYOUT === "1"

  const statusColor = () => {
    return colorForSessionStatus(store.state.status)
  }

  const scrollAcceleration = createMemo<ScrollAcceleration>(() => {
    if (process.platform === "darwin") {
      return new MacOSScrollAccel()
    }

    const raw = Number.parseFloat(process.env.CORTEX_TUI_SCROLL_SPEED ?? "")
    const speed = Number.isFinite(raw) && raw > 0 ? raw : 3
    return new CustomSpeedScroll(speed)
  })

  const messages = createMemo(() => store.state.orderedMessageIDs.map((id) => store.state.messages[id]).filter(Boolean))
  const hasPendingPermission = createMemo(() => Boolean(store.state.pendingPermission))

  // ---- Slash command palette (derived from the prompt contents) ----
  const [promptValue, setPromptValue] = createSignal("")
  const [selectedIndex, setSelectedIndex] = createSignal(0)

  const paletteOpen = createMemo(() => {
    const value = promptValue()
    // Literal leading '/', no space yet, single line, no permission modal.
    return /^\/\S*$/.test(value) && !value.includes("\n") && !hasPendingPermission()
  })
  // ARG-HINT mode: a recognized command followed by a space.
  const argCommand = createMemo<SlashCommand | undefined>(() => {
    const value = promptValue()
    if (!/^\/\S+\s/.test(value)) {
      return undefined
    }
    const command = findCommand(commandToken(value))
    return command && commandTakesArgs(command.name) ? command : undefined
  })
  const argMode = createMemo(() => Boolean(argCommand()))
  const matches = createMemo(() => (paletteOpen() ? matchCommands(promptValue()) : []))

  // Keep the highlight valid as the filter changes.
  createEffect(() => {
    const count = matches().length
    if (selectedIndex() >= count) {
      setSelectedIndex(count > 0 ? count - 1 : 0)
    }
  })

  // The panel is visible in both list mode and arg-hint mode.
  const paletteVisible = createMemo(() => paletteOpen() || argMode())

  // Mirror palette open-state into the store so app.tsx's global Esc handler
  // can enforce precedence, and register how to dismiss it (owns the input ref).
  createEffect(() => {
    store.setPaletteOpen(paletteVisible())
  })

  // ---- Live slash-command highlight: a VALID typed command token renders in
  // accent + bold inside the textarea (immediate "this will run" feedback). ----
  let commandStyleId: number | undefined
  const attachCommandHighlighter = (target: TextareaRenderable) => {
    try {
      const style = SyntaxStyle.fromStyles({ command: { fg: UI_PALETTE.accent, bold: true } })
      target.syntaxStyle = style
      commandStyleId = style.getStyleId("command") ?? undefined
    } catch {
      commandStyleId = undefined // degrade to uncolored input
    }
  }
  const updateCommandHighlight = () => {
    if (!input || input.isDestroyed || commandStyleId === undefined) {
      return
    }
    try {
      input.clearAllHighlights()
      const text = input.plainText
      const match = /^\/([A-Za-z]+)/.exec(text)
      if (match && findCommand(match[1])) {
        input.addHighlightByCharRange({ start: 0, end: match[0].length, styleId: commandStyleId })
      }
    } catch {
      // Highlighting is cosmetic — never let it break input handling.
    }
  }

  // SGR mouse reports (ESC[<b;x;yM) can occasionally leak into the textarea
  // when the wheel is scrolled over the prompt row; strip any such fragment
  // (full or prefix-truncated, e.g. "8;40M") so it never becomes typed text.
  const MOUSE_FRAGMENT = /(?:\x1b\[<|\[<|<)?\d{1,4}(?:;\d{1,4}){1,2}[Mm]/g
  const syncPromptValue = () => {
    if (!input || input.isDestroyed) {
      return
    }
    const raw = input.plainText
    const cleaned = raw.replace(MOUSE_FRAGMENT, "")
    if (cleaned !== raw) {
      input.setText(cleaned)
      input.gotoBufferEnd()
      setPromptValue(cleaned)
      updateCommandHighlight()
      return
    }
    setPromptValue(raw)
    updateCommandHighlight()
  }

  const closePaletteAndClearInput = () => {
    if (input && !input.isDestroyed) {
      input.clear()
    }
    setPromptValue("")
    setSelectedIndex(0)
  }

  // ---- Interactive /model picker: one tab per origin (Local / Cloud) ----
  interface ModelEntry {
    kind: "entry" | "divider"
    selector: string // what /model receives: a local name or "provider:model"
    primary: string
    size?: string // on-disk size for downloaded local models, e.g. "5.4 GB"
    tag?: SelectionTag
    active: boolean
  }
  type PickerTab = "local" | "cloud"
  const PICKER_TABS: PickerTab[] = ["local", "cloud"]
  const [modelPickerOpen, setModelPickerOpen] = createSignal(false)
  const [modelPickerIndex, setModelPickerIndex] = createSignal(0)
  const [modelPickerTab, setModelPickerTab] = createSignal<PickerTab>("local")

  const PICKER_DIVIDER: ModelEntry = { kind: "divider", selector: "", primary: "", active: false }

  const localPickerEntries = createMemo<ModelEntry[]>(() => {
    const downloaded: ModelEntry[] = []
    const downloadable: ModelEntry[] = []
    for (const item of store.state.models.local) {
      const name = String(item.name ?? "").trim()
      if (!name) continue
      const active = Boolean(item.active)
      const cached = item.cached !== false
      const loading =
        Boolean(item.loading) ||
        store.state.activeDownloadRepoId === name ||
        store.state.activeLoadSelector === name
      const entry: ModelEntry = {
        kind: "entry",
        selector: name,
        primary: name,
        size: cached ? String(item.size ?? "").trim() || undefined : undefined,
        active,
        tag: loading
          ? {
              text: store.state.activeDownloadRepoId === name ? "downloading…" : "loading…",
              color: UI_PALETTE.statusBusy,
            }
          : active
            ? { text: "active", color: UI_PALETTE.accent }
            : item.loaded
              ? { text: "loaded", color: UI_PALETTE.statusIdle }
              : !cached
                ? { text: "select to download", color: UI_PALETTE.statusBusy }
                : { text: "ready", color: UI_PALETTE.statusIdle },
      }
      if (cached) {
        downloaded.push(entry)
      } else {
        downloadable.push(entry)
      }
    }
    // Downloaded first; one blank divider before the download candidates —
    // the tab is the grouping, the tags carry per-row state.
    if (downloaded.length > 0 && downloadable.length > 0) {
      return [...downloaded, PICKER_DIVIDER, ...downloadable]
    }
    return [...downloaded, ...downloadable]
  })

  const cloudPickerEntries = createMemo<ModelEntry[]>(() => {
    const cloud: ModelEntry[] = []
    for (const item of store.state.models.cloud) {
      const selector = String(item.selector ?? "").trim()
      if (!selector) continue
      const active = Boolean(item.active)
      cloud.push({
        kind: "entry",
        selector,
        primary: selector,
        active,
        tag: active
          ? { text: "active", color: UI_PALETTE.accent }
          : item.authenticated
            ? { text: "ready", color: UI_PALETTE.statusIdle }
            : { text: "login required", color: UI_PALETTE.statusBusy },
      })
    }
    return cloud
  })

  const modelPickerEntries = createMemo<ModelEntry[]>(() =>
    modelPickerTab() === "local" ? localPickerEntries() : cloudPickerEntries(),
  )

  const isPickerDivider = (entry: ModelEntry) => entry.kind === "divider"

  /** Next selectable index in `dir`, skipping divider rows (wraps). */
  const stepPickerIndex = (from: number, dir: 1 | -1): number => {
    const entries = modelPickerEntries()
    const count = entries.length
    if (count === 0) return 0
    let index = from
    for (let hops = 0; hops < count; hops += 1) {
      index = (index + dir + count) % count
      if (!isPickerDivider(entries[index])) return index
    }
    return from
  }

  /** Selection lands on the active model when it lives on this tab, else the
   * first selectable row. */
  const resetPickerIndex = () => {
    const entries = modelPickerEntries()
    const activeIdx = entries.findIndex((entry) => entry.active)
    const firstSelectable = entries.findIndex((entry) => !isPickerDivider(entry))
    setModelPickerIndex(activeIdx >= 0 ? activeIdx : Math.max(firstSelectable, 0))
  }

  const switchPickerTab = () => {
    setModelPickerTab((tab) => (tab === "local" ? "cloud" : "local"))
    resetPickerIndex()
  }

  const openModelPicker = () => {
    // The store's model list is refreshed after every command, so it is current.
    // Open on the tab matching the active backend; an empty tab falls through
    // to the other so the picker never opens onto nothing.
    let tab: PickerTab = store.state.activeBackend === "cloud" ? "cloud" : "local"
    const entriesFor = (which: PickerTab) =>
      which === "local" ? localPickerEntries() : cloudPickerEntries()
    if (entriesFor(tab).length === 0 && entriesFor(tab === "local" ? "cloud" : "local").length > 0) {
      tab = tab === "local" ? "cloud" : "local"
    }
    setModelPickerTab(tab)
    resetPickerIndex()
    closePaletteAndClearInput()
    setModelPickerOpen(true)
  }
  const closeModelPicker = () => setModelPickerOpen(false)

  // ---- /login provider picker (bare /login) — same idiom as /model ----
  interface LoginEntry {
    provider: string
    tag?: SelectionTag
  }
  const LOGIN_PROVIDERS = ["openai", "anthropic", "azure"] as const
  const [loginPickerOpen, setLoginPickerOpen] = createSignal(false)
  const [loginPickerIndex, setLoginPickerIndex] = createSignal(0)

  const loginPickerEntries = createMemo<LoginEntry[]>(() =>
    LOGIN_PROVIDERS.map((provider) => {
      // Auth status flows from model.list's cloud rows (authenticated + source).
      const row = store.state.models.cloud.find(
        (item) => String(item.provider ?? "").trim() === provider,
      )
      const authenticated = Boolean(row?.authenticated)
      const source = String(row?.auth_source ?? "").trim()
      return {
        provider,
        tag: authenticated
          ? {
              text: source ? `logged in · ${source}` : "logged in",
              color: UI_PALETTE.statusIdle,
            }
          : { text: "not configured", color: UI_PALETTE.statusBusy },
      }
    }),
  )

  const openLoginPicker = () => {
    setLoginPickerIndex(0)
    closePaletteAndClearInput()
    setLoginPickerOpen(true)
  }
  const closeLoginPicker = () => setLoginPickerOpen(false)

  createEffect(() => {
    // Keep the pickers in the store's overlay precedence for global Esc.
    store.setPaletteOpen(paletteVisible() || modelPickerOpen() || loginPickerOpen())
  })

  // app.tsx's global Esc calls this; close whichever overlay is open.
  const dismissOverlays = () => {
    if (loginPickerOpen()) {
      closeLoginPicker()
      return
    }
    if (modelPickerOpen()) {
      closeModelPicker()
      return
    }
    closePaletteAndClearInput()
  }
  store.registerPaletteDismiss(dismissOverlays)
  onCleanup(() => store.registerPaletteDismiss(undefined))

  // Set the input text AND move the caret to the end. setText leaves the
  // cursor at offset 0, so without gotoBufferEnd() a subsequently-typed
  // argument would be inserted before the completed "/command " text.
  const setInputText = (text: string) => {
    if (input && !input.isDestroyed) {
      input.setText(text)
      input.gotoBufferEnd()
    }
    setPromptValue(text)
    updateCommandHighlight()
  }

  const runSlashCommand = (line: string) => {
    const trimmed = line.trim()
    if (!trimmed) {
      return
    }
    if (!store.state.sessionID) {
      store.setError("Still connecting to worker — try again in a moment.")
      return
    }
    if (store.state.commandInFlight) {
      return
    }
    closePaletteAndClearInput()
    void props.onSubmit(trimmed)
  }

  const completeSelected = (command: SlashCommand) => {
    setInputText(commandTakesArgs(command.name) ? `/${command.name} ` : `/${command.name}`)
  }

  const onPaletteEnter = () => {
    const list = matches()
    if (list.length === 0) {
      // Let the worker report "Unknown command" into the ephemeral panel.
      runSlashCommand(promptValue())
      return
    }
    const command = list[Math.min(selectedIndex(), list.length - 1)]
    // Bare /model opens the interactive picker (lists AND switches in one).
    if (command.name === "model") {
      openModelPicker()
      return
    }
    // Bare /login opens the provider picker; Enter there pre-fills the key prompt.
    if (command.name === "login") {
      openLoginPicker()
      return
    }
    if (commandTakesArgs(command.name)) {
      completeSelected(command) // drop into arg-hint mode; do not execute yet
    } else {
      runSlashCommand(`/${command.name}`)
    }
  }

  // Returns true when the key was consumed by the palette.
  const handlePaletteKey = (event: KeyEvent): boolean => {
    if (!paletteVisible()) {
      return false
    }
    const action = classifySelectionKey(event)

    // ARG-HINT mode: usage line only; Enter runs the full command line.
    if (argMode()) {
      if (action === "enter") {
        event.preventDefault()
        runSlashCommand(promptValue())
        return true
      }
      return false
    }

    const count = matches().length
    if (action === "up" && count > 0) {
      event.preventDefault()
      setSelectedIndex((index) => (index - 1 + count) % count)
      return true
    }
    if (action === "down" && count > 0) {
      event.preventDefault()
      setSelectedIndex((index) => (index + 1) % count)
      return true
    }
    if (action === "tab") {
      event.preventDefault()
      if (count > 0) {
        completeSelected(matches()[Math.min(selectedIndex(), count - 1)])
      }
      return true
    }
    if (action === "enter") {
      event.preventDefault()
      onPaletteEnter()
      return true
    }
    return false // Esc falls through to the centralized Esc handler
  }

  // Model picker is modal: it owns every key while open (select-one-of-N).
  const handleModelPickerKey = (event: KeyEvent): boolean => {
    if (!modelPickerOpen()) {
      return false
    }
    event.preventDefault()
    const action = classifySelectionKey(event)
    const rawKey = String(event.name ?? "").toLowerCase()
    const entries = modelPickerEntries()
    const count = entries.length
    if (action === "tab" || rawKey === "left" || rawKey === "right") {
      // Tab / ←→ flip between the Local and Cloud tabs (picker-only keys —
      // Tab in the slash palette still completes commands).
      switchPickerTab()
    } else if (action === "up" && count > 0) {
      setModelPickerIndex((index) => stepPickerIndex(index, -1))
    } else if (action === "down" && count > 0) {
      setModelPickerIndex((index) => stepPickerIndex(index, 1))
    } else if (action === "enter" && count > 0) {
      const entry = entries[Math.min(modelPickerIndex(), count - 1)]
      if (!entry || isPickerDivider(entry)) {
        return true
      }
      closeModelPicker()
      runSlashCommand(`/model ${entry.selector}`)
    } else if (action === "cancel") {
      closeModelPicker()
    }
    return true // swallow everything else (modal)
  }

  // Login picker is modal: it owns every key while open.
  const handleLoginPickerKey = (event: KeyEvent): boolean => {
    if (!loginPickerOpen()) {
      return false
    }
    event.preventDefault()
    const action = classifySelectionKey(event)
    const entries = loginPickerEntries()
    const count = entries.length
    if (action === "up" && count > 0) {
      setLoginPickerIndex((index) => (index - 1 + count) % count)
    } else if (action === "down" && count > 0) {
      setLoginPickerIndex((index) => (index + 1) % count)
    } else if (action === "enter" && count > 0) {
      const entry = entries[Math.min(loginPickerIndex(), count - 1)]
      closeLoginPicker()
      // Pre-fill the key prompt: arg-hint mode takes over ("/login openai ").
      setInputText(`/login ${entry.provider} `)
    } else if (action === "cancel") {
      closeLoginPicker()
    }
    return true // swallow everything else (modal)
  }

  // Animated working indicator (shared spinner + elapsed + tool count).
  const currentTurnToolCount = createMemo(() => {
    const list = messages()
    for (let i = list.length - 1; i >= 0; i--) {
      const message = list[i]
      if (message.role === "assistant") {
        return message.parts.filter((part) => part.type === "tool").length
      }
    }
    return 0
  })
  const workingLine = createMemo(() => {
    if (store.state.status !== "busy") {
      return ""
    }
    const frame = spinnerFrame()
    const since = store.state.busySinceMs
    const elapsed = since ? Math.max(0, Math.round((Date.now() - since) / 1000)) : 0
    const tools = currentTurnToolCount()
    const toolNote = tools > 0 ? ` · ${tools} tool call${tools === 1 ? "" : "s"}` : ""
    const expandNote = tools > 0 ? " · ctrl+o details" : ""
    return `${frame} Working… ${elapsed}s${toolNote} · Esc to interrupt${expandNote}`
  })

  // Input history (Up/Down when the prompt is empty or while navigating).
  const inputHistory: string[] = []
  let historyIndex = -1
  let historyDraft = ""
  const activeModelLabel = createMemo(() => {
    const label = store.state.activeModelLabel?.trim()
    return label && label.length > 0 ? label : "No model loaded"
  })
  const hasActiveModel = createMemo(() => activeModelLabel() !== "No model loaded")

  // ---- Single-line header: "⎇ branch ~/path         model · status" ----
  const gitBranch = createMemo(() => {
    void store.state.status // re-check on turn transitions (branch switches are rare)
    return readGitBranch(process.env.CORTEX_PROJECT_DIR || process.cwd())
  })
  const headerModel = createMemo(() => {
    if (!hasActiveModel()) {
      return "no model"
    }
    const origin = (store.state.activeBackend ?? "").trim()
    const label = origin ? `${origin} · ${activeModelLabel()}` : activeModelLabel()
    return label.length > 36 ? `${label.slice(0, 35)}…` : label
  })
  const headerRightLen = createMemo(() => headerModel().length + 3 + store.state.status.length)
  // Branch is capped (never wider than 24 or the remaining budget) and SHED
  // entirely when the row cannot fit branch + path floor + right side — the
  // header must stay a single collision-free line at every width.
  const headerBranch = createMemo(() => {
    const branch = gitBranch()
    if (!branch) {
      return undefined
    }
    const available = terminalColumns() - 6 - headerRightLen() - 16 - 3
    if (available < 8) {
      return undefined
    }
    const cap = Math.min(24, available)
    return branch.length > cap ? `${branch.slice(0, cap - 1)}…` : branch
  })
  const workingDir = createMemo(() => {
    const dir = process.env.CORTEX_PROJECT_DIR || process.cwd()
    const home = process.env.HOME
    const abbreviated = home && dir.startsWith(home) ? `~${dir.slice(home.length)}` : dir
    // Middle-ellipsize so branch + path + right side share one row without
    // wrapping or colliding. Reactive to terminalColumns() for live resize.
    const cols = terminalColumns()
    const branchLen = headerBranch() ? headerBranch()!.length + 3 : 0
    const budget = Math.max(16, Math.min(cols, 120) - 6 - branchLen - headerRightLen())
    return truncatePathMiddle(abbreviated, budget)
  })

  const emitLayoutDebug = (stage: string) => {
    if (!debugLayout) {
      return
    }
    const payload = {
      stage,
      rows: process.stdout.rows ?? null,
      cols: process.stdout.columns ?? null,
      status: store.state.status,
      message_count: messages().length,
      pending_permission: hasPendingPermission(),
    }
    process.stderr.write(`[layout] ${JSON.stringify(payload)}\n`)
  }

  const toBottom = () => {
    setTimeout(() => {
      if (!transcriptScroll || transcriptScroll.isDestroyed) {
        return
      }
      transcriptScroll.scrollTo(transcriptScroll.scrollHeight)
    }, 50)
  }

  // Patterns/paths display repo-relative when inside the repo (display only).
  const permissionTarget = () =>
    store.state.pendingPermission?.patterns.map(displayPath).join(", ") || "*"

  const permissionSummary = () => {
    const pending = store.state.pendingPermission
    if (!pending) {
      return ""
    }

    // Tool metadata is {tool, arguments: {...}} — the details live in arguments.
    const meta = (pending.metadata || {}) as Record<string, unknown>
    const args = (meta.arguments ?? {}) as Record<string, unknown>
    const tool = String(meta.tool ?? "")
    const argPath = displayPath(String(args.path ?? pending.patterns[0] ?? ""))

    if (pending.permission === "read") {
      return `Read file: ${argPath}`
    }
    if (pending.permission === "list") {
      return `List directory: ${argPath}`
    }
    if (pending.permission === "grep") {
      return `Search pattern: ${String(args.query ?? args.pattern ?? "")}`
    }
    if (pending.permission === "bash") {
      return `Run command: ${String(args.command ?? "")}`
    }
    if (pending.permission === "edit") {
      const action = tool === "write_file" ? "Write file" : "Edit file"
      return `${action}: ${argPath}`
    }
    if (pending.permission === "external_directory") {
      return `Access external directory: ${String(args.parentDir ?? args.filepath ?? pending.patterns[0] ?? "")}`
    }
    if (pending.permission === "doom_loop") {
      return "Continue after repeated identical tool calls."
    }
    return `Permission: ${pending.permission}${tool ? ` (${tool})` : ""}`
  }

  const submitFromPrompt = async () => {
    if (!input) {
      return
    }
    // The palette owns Enter while it is open; never fall through to submit.
    if (paletteVisible()) {
      return
    }
    const value = input.plainText
    const fingerprint = value.trim()
    if (!fingerprint) {
      return
    }
    const commandLike = isSlashCommandLike(fingerprint)
    // Recall history: prompts only (not slash commands), no consecutive dupes,
    // capped so a long session can't grow it unbounded.
    if (!commandLike && inputHistory[inputHistory.length - 1] !== fingerprint) {
      inputHistory.push(fingerprint)
      if (inputHistory.length > 200) {
        inputHistory.shift()
      }
    }
    historyIndex = -1
    historyDraft = ""
    // Commands never queue: run now (ephemeral) or report we're still connecting.
    if (commandLike) {
      if (!store.state.sessionID) {
        store.setError("Still connecting to worker — try again in a moment.")
        return
      }
      input.clear()
      setPromptValue("")
      await props.onSubmit(value)
      return
    }
    // A new chat turn dismisses any lingering command result panel.
    store.clearCommandResult()
    // Not connected yet (bootstrap in flight): queue chat input.
    if (!store.state.sessionID) {
      store.enqueueInput(fingerprint)
      input.clear()
      return
    }
    // Busy turn: queue the follow-up and send it when the turn finishes.
    if (!commandLike && (store.state.status === "busy" || isSubmitting())) {
      store.enqueueInput(fingerprint)
      input.clear()
      return
    }
    if (isSubmitting() && !commandLike) {
      return
    }
    const now = Date.now()
    if (fingerprint === lastSubmitFingerprint && now - lastSubmitAtMs < 250) {
      return
    }
    lastSubmitFingerprint = fingerprint
    lastSubmitAtMs = now
    if (!commandLike) {
      setIsSubmitting(true)
    }
    input.clear()

    let submitted = false
    try {
      submitted = await props.onSubmit(value)
    } finally {
      if (!commandLike) {
        setIsSubmitting(false)
      }
      lastSubmitFingerprint = ""
      lastSubmitAtMs = 0
    }
    if (submitted) {
      toBottom()
      emitLayoutDebug("submit_success")
      return
    }

    if (input && !input.isDestroyed && input.plainText.trim().length === 0) {
      input.setText(value)
      input.gotoBufferEnd()
    }
  }

  const onPromptKeyDown = (event: KeyEvent) => {
    if (!hasPendingPermission()) {
      // Pickers are modal and own all keys while open.
      if (handleModelPickerKey(event)) {
        return
      }
      if (handleLoginPickerKey(event)) {
        return
      }
      // Palette navigation/completion/execution wins over submit and history.
      if (handlePaletteKey(event)) {
        return
      }
      const key = String(event.name ?? "").toLowerCase()
      const shift = Boolean((event as { shift?: boolean }).shift)

      // Esc handled here (the focused textarea otherwise swallows it before the
      // global handler): palette → clear; result panel → dismiss; busy → interrupt.
      if (key === "escape" || key === "esc") {
        if (paletteVisible()) {
          event.preventDefault()
          closePaletteAndClearInput()
          return
        }
        if (store.state.commandResult) {
          event.preventDefault()
          store.clearCommandResult()
          return
        }
        if (store.state.status === "busy") {
          event.preventDefault()
          props.onInterrupt()
          return
        }
        return
      }

      // Keyboard scrolling of the transcript (mouse reporting is disabled).
      if ((key === "pageup" || key === "pagedown") && transcriptScroll && !transcriptScroll.isDestroyed) {
        event.preventDefault()
        transcriptScroll.scrollBy({ x: 0, y: key === "pageup" ? -10 : 10 })
        return
      }
      const isEnterKey =
        key === "return" ||
        key === "linefeed" ||
        key === "kpenter" ||
        key === "numpadenter" ||
        key.includes("enter")
      if (isEnterKey && !shift) {
        event.preventDefault()
        void submitFromPrompt()
        return
      }

      // Prompt history: Up/Down when the input is empty or mid-recall.
      if (input && (key === "up" || key === "down") && inputHistory.length > 0) {
        const navigating = historyIndex >= 0
        const empty = input.plainText.trim().length === 0
        if (!navigating && !empty) {
          return
        }
        event.preventDefault()
        if (key === "up") {
          if (!navigating) {
            historyDraft = input.plainText
            historyIndex = inputHistory.length - 1
          } else if (historyIndex > 0) {
            historyIndex -= 1
          }
          input.setText(inputHistory[historyIndex] ?? "")
          input.gotoBufferEnd()
        } else {
          if (!navigating) {
            return
          }
          if (historyIndex < inputHistory.length - 1) {
            historyIndex += 1
            input.setText(inputHistory[historyIndex] ?? "")
            input.gotoBufferEnd()
          } else {
            historyIndex = -1
            input.setText(historyDraft)
            input.gotoBufferEnd()
            historyDraft = ""
          }
        }
      }
      return
    }
    event.preventDefault()
  }

  // Dispatch queued follow-ups once the session is connected and idle.
  createEffect(() => {
    if (
      store.state.sessionID &&
      store.state.status === "idle" &&
      !hasPendingPermission() &&
      store.state.queuedInputs.length > 0 &&
      !isSubmitting()
    ) {
      const next = store.dequeueInput()
      if (next) {
        setIsSubmitting(true)
        void props.onSubmit(next).finally(() => setIsSubmitting(false))
      }
    }
  })

  createEffect(() => {
    if (!hasPendingPermission() && input && !input.isDestroyed) {
      input.focus()
    }
  })

  createEffect(
    on(
      () => store.state.sessionID,
      () => {
        toBottom()
        emitLayoutDebug("session_changed")
      },
      { defer: true },
    ),
  )

  createEffect(
    on(
      () => store.state.status,
      () => {
        emitLayoutDebug("status_changed")
      },
      { defer: true },
    ),
  )

  createEffect(() => {
    emitLayoutDebug("messages_changed")
    const lastAssistant = [...messages()].reverse().find((message) => message.role === "assistant")
    if (!lastAssistant) {
      return
    }
    if (lastAssistant.id !== lastAssistantMessageID) {
      lastAssistantMessageID = lastAssistant.id
      sawFirstAssistantDelta = false
    }
    if (!sawFirstAssistantDelta && lastAssistant.content.length > 0) {
      sawFirstAssistantDelta = true
      toBottom()
      emitLayoutDebug("assistant_first_delta")
    }
  })

  const hasConversation = createMemo(() =>
    messages().some((message) => message.role === "user" || message.role === "assistant"),
  )

  const TranscriptBody = () => (
    <>
      <Show when={!hasConversation()}>
        {/* marginBottom keeps the first system notice from touching the help text. */}
        <box flexDirection="column" flexShrink={0} gap={1} marginBottom={1}>
          <text fg={UI_PALETTE.text}>Ask anything about this repository, or give Cortex a task.</text>
          <text fg={UI_PALETTE.textMuted}>
            It can read, search, edit files, and run commands here — edits and commands ask first.
          </text>
          <text fg={UI_PALETTE.textMuted}>
            /model picks a local or cloud model · /download fetches one · /help lists commands
          </text>
        </box>
      </Show>
      <Show when={messages().length > 0}>
        <For each={messages()}>
          {(message, index) => (
            <Switch>
              <Match when={message.role === "user"}>
                <UserMessage message={message} index={index()} />
              </Match>
              <Match when={message.role === "system"}>
                <SystemMessage message={message} index={index()} />
              </Match>
              <Match when={message.role === "assistant"}>
                <AssistantMessage message={message} index={index()} toolsExpanded={store.state.toolsExpanded} />
              </Match>
            </Switch>
          )}
        </For>
      </Show>
    </>
  )

  return (
    <box flexDirection="row" width="100%">
      <box
        flexDirection="column"
        flexGrow={1}
        paddingTop={1}
        paddingBottom={1}
        paddingLeft={2}
        paddingRight={2}
        gap={1}
      >
        {/* Single-line header (reference-CLI style): branch + path on the
            left, model · status on the right. All chrome, kept quiet. */}
        <box flexDirection="row" justifyContent="space-between" flexShrink={0}>
          {/* Spans stay mounted (empty when the branch is shed) — a <Show>
              inside <text> re-appends its spans on toggle, scrambling order. */}
          <text>
            <span style={{ fg: UI_PALETTE.accent }}>{headerBranch() ? "⎇ " : ""}</span>
            <span style={{ fg: UI_PALETTE.text }}>{headerBranch() ? `${headerBranch()} ` : ""}</span>
            <span style={{ fg: UI_PALETTE.textMuted }}>{workingDir()}</span>
          </text>
          <text flexShrink={0} marginLeft={2}>
            <span style={{ fg: hasActiveModel() ? UI_PALETTE.textMuted : UI_PALETTE.statusBusy }}>
              {headerModel()}
            </span>
            <span style={{ fg: UI_PALETTE.textMuted }}>{" · "}</span>
            <span style={{ fg: statusColor() }}>{store.state.status}</span>
          </text>
        </box>

        <scrollbox
          ref={(r: ScrollBoxRenderable) => {
            transcriptScroll = r
          }}
          flexGrow={1}
          stickyScroll
          stickyStart="bottom"
          viewportOptions={{ paddingRight: 0 }}
          verticalScrollbarOptions={{ visible: false }}
          scrollAcceleration={scrollAcceleration()}
        >
          {/* Bottom-anchor: a short conversation sits just above the prompt
              (empty space collects at the TOP, which reads naturally) instead
              of stranding the input under a large void. Long content overflows
              and scrolls as before. */}
          <box flexGrow={1} flexDirection="column" justifyContent="flex-end">
            <TranscriptBody />
          </box>
        </scrollbox>

        <Show when={store.state.error}>
          <box>
            <text fg={UI_PALETTE.statusError}>Error: {store.state.error}</text>
          </box>
        </Show>

        <Show when={workingLine()}>
          <box flexShrink={0}>
            <text fg={UI_PALETTE.statusBusy}>{workingLine()}</text>
          </box>
        </Show>

        <Show when={store.state.status === "idle" && store.state.lastInterrupted}>
          <box flexShrink={0}>
            <text fg={UI_PALETTE.textMuted}>✕ Interrupted.</text>
          </box>
        </Show>

        <Show when={store.state.queuedInputs.length > 0}>
          <box flexDirection="column" flexShrink={0}>
            <For each={store.state.queuedInputs}>
              {(queued) => <text fg={UI_PALETTE.textMuted}>↳ queued: {queued}</text>}
            </For>
          </box>
        </Show>

        <Show when={store.state.pendingPermission}>
          <box
            flexDirection="column"
            flexShrink={0}
            paddingLeft={2}
            border={["left"]}
            borderColor={UI_PALETTE.statusBusy}
            customBorderChars={OVERLAY_BORDER}
          >
            <text fg={UI_PALETTE.statusBusy}>Permission required</text>
            <text fg={UI_PALETTE.text}>{permissionSummary()}</text>
            {/* Only show Target when it adds info beyond the summary line. */}
            <Show when={permissionTarget() && !permissionSummary().includes(permissionTarget())}>
              <text fg={UI_PALETTE.textMuted}>{`Target: ${permissionTarget()}`}</text>
            </Show>
            {/* Blank row above the options and above the footer (menu rhythm). */}
            <box flexShrink={0} marginTop={1}>
              <SelectionRows
                items={PERMISSION_OPTIONS}
                selectedIndex={store.state.permissionChoiceIndex}
                getPrimary={(option) => option.label}
                getDanger={(option) => Boolean(option.danger)}
              />
            </box>
            {/* Width-adaptive so the footer never wraps mid-word on narrow terminals. */}
            <box flexShrink={0} marginTop={1}>
              <text fg={UI_PALETTE.textMuted}>
                {terminalColumns() < 56 ? "↑↓ · Enter confirm · Esc reject" : "↑↓ select · Enter confirm · Esc reject"}
              </text>
            </box>
          </box>
        </Show>

        {/* Interactive /model picker — same overlay slot, mutually exclusive.
            One origin per tab (Local / Cloud); Tab or ←→ switch. Primary names
            are width-budgeted so they ellipsize instead of colliding with the
            right-aligned status tag on narrow terminals. */}
        <Show when={modelPickerOpen()}>
          <SelectionList
            items={modelPickerEntries()}
            selectedIndex={modelPickerIndex()}
            getPrimary={(entry) => {
              const budget = Math.max(12, terminalColumns() - 24)
              return entry.primary.length > budget ? `${entry.primary.slice(0, budget - 1)}…` : entry.primary
            }}
            getSecondary={(entry) => (entry.size ? `  ${entry.size}` : undefined)}
            getTag={(entry) => entry.tag}
            isHeader={isPickerDivider}
            title="Select a model"
            tabs={{
              labels: ["Local", "Cloud"],
              activeIndex: PICKER_TABS.indexOf(modelPickerTab()),
            }}
            footer={
              terminalColumns() < 64
                ? `↑↓ · Tab ${modelPickerTab() === "local" ? "cloud" : "local"} · Enter · Esc`
                : modelPickerTab() === "local"
                  ? "↑↓ select · Tab cloud · Enter load · Esc cancel"
                  : "↑↓ select · Tab local · Enter use · Esc cancel"
            }
            emptyLabel={
              modelPickerTab() === "local"
                ? "No local models — Lumen not detected. Tab for cloud."
                : "No cloud models — /login openai|anthropic|azure. Tab for local."
            }
          />
        </Show>

        {/* /login provider picker — Enter pre-fills "/login <provider> " so
            the arg-hint prompt asks for the API key. */}
        <Show when={loginPickerOpen()}>
          <SelectionList
            items={loginPickerEntries()}
            selectedIndex={loginPickerIndex()}
            getPrimary={(entry) => entry.provider}
            getTag={(entry) => entry.tag}
            title="Log in to a provider"
            footer="↑↓ select · Enter choose · Esc cancel"
            emptyLabel="No providers available."
          />
        </Show>

        {/* Slash command palette and its ephemeral result share this slot,
            directly above the prompt, and are mutually exclusive. */}
        <Show when={paletteVisible()}>
          <CommandPalette
            matches={matches()}
            selectedIndex={selectedIndex()}
            argMode={argMode()}
            argCommand={argCommand()}
          />
        </Show>
        <Show when={store.state.commandResult && !paletteVisible()}>
          <CommandResultPanel
            command={store.state.commandResult!.command}
            ok={store.state.commandResult!.ok}
            text={store.state.commandResult!.text}
          />
        </Show>

        <PromptPanel
          hasPendingPermission={hasPendingPermission()}
          status={store.state.status}
          statusColor={statusColor()}
          setInputRef={(r: TextareaRenderable) => {
            input = r
            attachCommandHighlighter(r)
            if (!hasPendingPermission() && !r.isDestroyed) {
              r.focus()
            }
          }}
          onPromptKeyDown={onPromptKeyDown}
          onContentChange={syncPromptValue}
          onSubmit={() => {
            void submitFromPrompt()
          }}
        />
      </box>
    </box>
  )
}
