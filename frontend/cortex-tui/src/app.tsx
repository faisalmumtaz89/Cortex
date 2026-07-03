import { useKeyboard, useRenderer } from "@opentui/solid"
import type { KeyEvent } from "@opentui/core"
import { ErrorBoundary } from "solid-js"
import { RpcProvider, useRpc } from "./context/rpc"
import { StoreProvider, useStore } from "./context/store"
import { SessionRoute } from "./routes/session"
import { classifySelectionKey } from "./components/selection_list"
import { UI_PALETTE } from "./components/ui_palette"

const PERMISSION_REPLIES = ["allow_once", "allow_always", "reject"] as const

function SessionScreen() {
  const rpc = useRpc()
  const store = useStore()
  const renderer = useRenderer()

  renderer.disableStdoutInterception()

  const submit = async (value: string): Promise<boolean> => {
    const trimmed = value.trim()
    if (!trimmed) {
      return false
    }
    return rpc.submitInput(trimmed)
  }

  useKeyboard((evt: KeyEvent) => {
    const rawKey = String(evt.name ?? "")
    const key = rawKey.toLowerCase()
    const ctrl = Boolean((evt as { ctrl?: boolean }).ctrl)
    if (store.state.pendingPermission) {
      // The prompt textarea is unfocused while a permission is pending, so the
      // global handler owns these keys. Arrow-select + Enter, no numbers.
      const requestId = store.state.pendingPermission.request_id
      const action = classifySelectionKey(evt)
      if (action === "up") {
        store.setPermissionChoice((store.state.permissionChoiceIndex + PERMISSION_REPLIES.length - 1) % PERMISSION_REPLIES.length)
        return
      }
      if (action === "down") {
        store.setPermissionChoice((store.state.permissionChoiceIndex + 1) % PERMISSION_REPLIES.length)
        return
      }
      if (action === "enter") {
        void rpc.replyPermission(requestId, PERMISSION_REPLIES[store.state.permissionChoiceIndex])
        return
      }
      if (action === "cancel") {
        void rpc.replyPermission(requestId, "reject") // Esc always rejects
        return
      }
      // Permission prompt is modal: ignore all other keys until resolved.
      return
    }

    const isEsc = key === "escape" || key === "esc"

    // Esc precedence (highest first): palette → result panel → busy-interrupt.
    if (isEsc && store.state.paletteOpen) {
      store.dismissPalette()
      return
    }
    if (isEsc && store.state.commandResult) {
      store.clearCommandResult()
      return
    }

    // Ctrl+O toggles tool detail, but not while a command overlay is showing.
    if (ctrl && key === "o") {
      if (!store.state.paletteOpen && !store.state.commandResult) {
        store.toggleToolsExpanded()
      }
      return
    }

    if (
      isEsc &&
      store.state.status === "busy" &&
      !store.state.paletteOpen &&
      !store.state.commandResult &&
      !store.state.commandInFlight
    ) {
      void rpc.interrupt()
      return
    }
  })

  return <SessionRoute onSubmit={submit} onInterrupt={() => void rpc.interrupt()} />
}

export function App() {
  return (
    <StoreProvider>
      <ErrorBoundary fallback={(error) => <text fg={UI_PALETTE.statusError}>UI runtime error: {String(error)}</text>}>
        <RpcProvider>
          <SessionScreen />
        </RpcProvider>
      </ErrorBoundary>
    </StoreProvider>
  )
}
