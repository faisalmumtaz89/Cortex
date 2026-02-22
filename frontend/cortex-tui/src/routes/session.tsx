import { MacOSScrollAccel, type KeyEvent, type ScrollAcceleration, type ScrollBoxRenderable, type TextareaRenderable } from "@opentui/core"
import { For, Match, Show, Switch, createEffect, createMemo, on } from "solid-js"
import { PromptPanel } from "../components/prompt_panel"
import { AssistantMessage } from "../components/messages/assistant_message"
import { SystemMessage } from "../components/messages/system_message"
import { UserMessage } from "../components/messages/user_message"
import { UI_PALETTE, colorForSessionStatus } from "../components/ui_palette"
import { useStore } from "../context/store"

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
  "models",
  "download",
  "login",
  "clear",
  "save",
  "benchmark",
  "template",
  "finetune",
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

export function SessionRoute(props: {
  onSubmit: (value: string) => Promise<boolean>
}) {
  const store = useStore()
  let input: TextareaRenderable | undefined
  let transcriptScroll: ScrollBoxRenderable | undefined
  let lastSubmitFingerprint = ""
  let lastSubmitAtMs = 0
  let isSubmitting = false
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
  const activeModelLabel = createMemo(() => {
    const label = store.state.activeModelLabel?.trim()
    return label && label.length > 0 ? label : "No model loaded"
  })
  const hasActiveModel = createMemo(() => activeModelLabel() !== "No model loaded")

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

  const permissionSummary = () => {
    const pending = store.state.pendingPermission
    if (!pending) {
      return ""
    }

    const meta = pending.metadata || {}
    if (pending.permission === "read") {
      return `Read file: ${String(meta.path ?? pending.patterns[0] ?? "")}`
    }
    if (pending.permission === "list") {
      return `List directory: ${String(meta.path ?? pending.patterns[0] ?? "")}`
    }
    if (pending.permission === "grep") {
      return `Search pattern: ${String(meta.pattern ?? "")}`
    }
    if (pending.permission === "bash") {
      return `Run command: ${String(meta.command ?? "")}`
    }
    if (pending.permission === "external_directory") {
      return `Access external directory: ${String(meta.parentDir ?? meta.filepath ?? pending.patterns[0] ?? "")}`
    }
    if (pending.permission === "doom_loop") {
      return "Continue after repeated identical tool calls."
    }
    return `Permission: ${pending.permission}`
  }

  const submitFromPrompt = async () => {
    if (!input) {
      return
    }
    const value = input.plainText
    const fingerprint = value.trim()
    if (!fingerprint) {
      return
    }
    const commandLike = isSlashCommandLike(fingerprint)
    if (isSubmitting && !commandLike) {
      return
    }
    const now = Date.now()
    if (fingerprint === lastSubmitFingerprint && now - lastSubmitAtMs < 250) {
      return
    }
    lastSubmitFingerprint = fingerprint
    lastSubmitAtMs = now
    if (!commandLike) {
      isSubmitting = true
    }
    input.clear()

    let submitted = false
    try {
      submitted = await props.onSubmit(value)
    } finally {
      if (!commandLike) {
        isSubmitting = false
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
    }
  }

  const onPromptKeyDown = (event: KeyEvent) => {
    if (!hasPendingPermission()) {
      const key = String(event.name ?? "").toLowerCase()
      const shift = Boolean((event as { shift?: boolean }).shift)
      const isEnterKey =
        key === "return" ||
        key === "linefeed" ||
        key === "kpenter" ||
        key === "numpadenter" ||
        key.includes("enter")
      if (isEnterKey && !shift) {
        event.preventDefault()
        void submitFromPrompt()
      }
      return
    }
    event.preventDefault()
  }

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

  const TranscriptBody = () => (
    <>
      <Show when={messages().length === 0}>
        <text fg={UI_PALETTE.textMuted}>No messages yet.</text>
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
                <AssistantMessage message={message} index={index()} />
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
        <box flexDirection="column" gap={0}>
          <box flexDirection="row" gap={1}>
            <text fg={UI_PALETTE.accent}>Cortex</text>
            <text fg={UI_PALETTE.textMuted}>status:</text>
            <text fg={statusColor()}>{store.state.status}</text>
          </box>

          <box height={1} />

          <box flexDirection="row" backgroundColor={UI_PALETTE.panel}>
            <text fg={hasActiveModel() ? UI_PALETTE.accent : UI_PALETTE.statusBusy}>│</text>
            <box flexDirection="row" gap={1} paddingLeft={2}>
              <text fg={UI_PALETTE.textMuted}>Model:</text>
              <text fg={hasActiveModel() ? UI_PALETTE.text : UI_PALETTE.statusError}>{activeModelLabel()}</text>
            </box>
          </box>
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
          <TranscriptBody />
        </scrollbox>

        <Show when={store.state.error}>
          <box>
            <text fg={UI_PALETTE.statusError}>Error: {store.state.error}</text>
          </box>
        </Show>

        <Show when={store.state.pendingPermission}>
          <box flexDirection="column" border={["left"]} borderColor={UI_PALETTE.statusBusy} paddingLeft={2} gap={0}>
            <text fg={UI_PALETTE.statusBusy}>Permission required</text>
            <text fg={UI_PALETTE.text}>{permissionSummary()}</text>
            <text fg={UI_PALETTE.textMuted}>
              Target: {store.state.pendingPermission?.patterns.join(", ") || "*"}
            </text>
            <text fg={UI_PALETTE.textMuted}>[1] Allow once  [2] Allow always  [3] Reject  [Esc] Reject</text>
          </box>
        </Show>

        <PromptPanel
          hasPendingPermission={hasPendingPermission()}
          status={store.state.status}
          statusColor={statusColor()}
          setInputRef={(r: TextareaRenderable) => {
            input = r
            if (!hasPendingPermission() && !r.isDestroyed) {
              r.focus()
            }
          }}
          onPromptKeyDown={onPromptKeyDown}
          onSubmit={() => {
            void submitFromPrompt()
          }}
        />
      </box>
    </box>
  )
}
