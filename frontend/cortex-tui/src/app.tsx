import { useKeyboard, useRenderer } from "@opentui/solid"
import type { KeyEvent } from "@opentui/core"
import { ErrorBoundary } from "solid-js"
import { RpcProvider, useRpc } from "./context/rpc"
import { StoreProvider, useStore } from "./context/store"
import { SessionRoute } from "./routes/session"
import { UI_PALETTE } from "./components/ui_palette"

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
    if (store.state.pendingPermission) {
      if (key === "1") {
        void rpc.replyPermission(store.state.pendingPermission.request_id, "allow_once")
        return
      }
      if (key === "2") {
        void rpc.replyPermission(store.state.pendingPermission.request_id, "allow_always")
        return
      }
      if (key === "3" || key === "escape" || key === "esc") {
        void rpc.replyPermission(store.state.pendingPermission.request_id, "reject")
        return
      }
      // Permission prompt is modal: ignore all other keys until resolved.
      return
    }
  })

  return <SessionRoute onSubmit={submit} />
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
