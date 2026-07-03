import { createRoot, createSignal } from "solid-js"

// Shared spinner: ONE app-lifetime 100ms ticker drives every spinner in the
// UI (footer working line, running tool rows), so all spinners animate in
// phase and adding more spinning rows never adds timers. A Solid signal with
// no active observers triggers nothing, so the always-on tick is free while
// nothing on screen is spinning.

export const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const

const tick = createRoot(() => {
  const [value, setValue] = createSignal(0)
  setInterval(() => setValue((v) => v + 1), 100)
  return value
})

/** Current spinner glyph — reactive; read it inside JSX or memos. */
export function spinnerFrame(): string {
  return SPINNER_FRAMES[tick() % SPINNER_FRAMES.length]
}
