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
})
