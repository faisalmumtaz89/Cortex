import { appendFileSync, mkdirSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import { render } from "@opentui/solid"
import { App } from "./app"
import { exitCortex } from "./lib/exit"

// The screen belongs to the app, so errors go to a log file next to the
// worker's, where they can be read afterwards.
const ERROR_LOG = join(homedir(), ".cortex", "cortex-tui.log")

function logError(kind: string, error: unknown): void {
  const message = error instanceof Error ? error.stack ?? error.message : String(error)
  try {
    mkdirSync(join(homedir(), ".cortex"), { recursive: true })
    appendFileSync(ERROR_LOG, `${new Date().toISOString()} ${kind}: ${message}\n`)
  } catch {
    // Nowhere left to report it.
  }
}

process.on("unhandledRejection", (error) => logError("unhandled rejection", error))
process.on("uncaughtException", (error) => logError("uncaught exception", error))

// stdout and stderr are the terminal. A failed write means it is gone; exit
// instead of reporting the failure to the same dead terminal.
process.stdout.on("error", () => process.exit(1))
process.stderr.on("error", () => process.exit(1))

// The terminal closed. Registered before render() so it runs ahead of
// OpenTUI's own SIGHUP handler, which only destroys the renderer. A read
// that fails (EIO on macOS, where the hangup can come after it) means the
// same.
process.on("SIGHUP", () => exitCortex(129))
process.stdin.on("error", () => exitCortex(129))

void render(() => <App />, {
  targetFps: 60,
  gatherStats: false,
  // OpenTUI's exitOnCtrlC (and its SIGINT/SIGTERM handlers) only DESTROY the
  // renderer — they never exit the process, and the shared spinner interval +
  // worker pipes keep the event loop alive, which made Ctrl+C need a second
  // press. app.tsx owns Ctrl+C/SIGINT/SIGTERM via lib/exit.ts instead.
  exitOnCtrlC: false,
  autoFocus: false,
  screenMode: "alternate-screen",
  // Writes to stdout go straight to the terminal, never into the frame.
  externalOutputMode: "passthrough",
  // Mouse ON so the transcript scrollbox scrolls with the wheel (the expected
  // gesture). Movement reporting stays off (it is the noisiest source of stray
  // sequences); any SGR fragment that still leaks into the prompt is stripped
  // in session.tsx's input handler. PageUp/PageDown also scroll.
  useMouse: true,
  enableMouseMovement: false,
  // An error stays in the log: OpenTUI's console overlay would cover the app.
  openConsoleOnError: false,
})
