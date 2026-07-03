import { render } from "@opentui/solid"
import { App } from "./app"

process.on("unhandledRejection", (error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error)
  process.stderr.write(`[frontend] unhandled rejection: ${message}\n`)
})

process.on("uncaughtException", (error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error)
  process.stderr.write(`[frontend] uncaught exception: ${message}\n`)
})

void render(() => <App />, {
  targetFps: 60,
  gatherStats: false,
  exitOnCtrlC: true,
  autoFocus: false,
  useAlternateScreen: true,
  // Mouse ON so the transcript scrollbox scrolls with the wheel (the expected
  // gesture). Movement reporting stays off (it is the noisiest source of stray
  // sequences); any SGR fragment that still leaks into the prompt is stripped
  // in session.tsx's input handler. PageUp/PageDown also scroll.
  useMouse: true,
  enableMouseMovement: false,
})
