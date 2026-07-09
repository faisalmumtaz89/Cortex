// Context/provider binding for the session store. ALL store logic lives in
// ./session_store (pure .ts, unit-testable under bun without a JSX
// transform); this file only mounts it into Solid's context tree and
// re-exports the module so existing "../context/store" imports keep working.
import { createContext, useContext, type ParentProps } from "solid-js"
import { createSessionStore, type SessionStore } from "./session_store"

export * from "./session_store"

const StoreContext = createContext<SessionStore>()

export function StoreProvider(props: ParentProps) {
  return <StoreContext.Provider value={createSessionStore()}>{props.children}</StoreContext.Provider>
}

export function useStore() {
  const ctx = useContext(StoreContext)
  if (!ctx) {
    throw new Error("useStore must be used inside StoreProvider")
  }
  return ctx
}
