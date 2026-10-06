import { render } from "@opentui/solid"
import { App } from "./app"
import { exitCortex } from "./lib/exit"

process.on("unhandledRejection", (error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error)
  process.stderr.write(`[frontend] unhandled rejection: ${message}\n`)
})

process.on("uncaughtException", (error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error)
  process.stderr.write(`[frontend] uncaught exception: ${message}\n`)
})

// stdout and stderr are the terminal. A failed write means it is gone; exit
// instead of reporting the failure to the same dead terminal.
process.stdout.on("error", () => process.exit(1))
process.stderr.on("error", () => process.exit(1))

// The terminal closed. Registered before render() so it runs ahead of
// OpenTUI's own SIGHUP handler, which only destroys the renderer.
process.on("SIGHUP", () => exitCortex(129))

void render(() => <App />, {
  targetFps: 60,
  gatherStats: false,
  // OpenTUI's exitOnCtrlC (and its SIGINT/SIGTERM handlers) only DESTROY the
  // renderer — they never exit the process, and the shared spinner interval +
  // worker pipes keep the event loop alive, which made Ctrl+C need a second
  // press. app.tsx owns Ctrl+C/SIGINT/SIGTERM via lib/exit.ts instead.
  exitOnCtrlC: false,
  autoFocus: false,
  useAlternateScreen: true,
  // Mouse ON so the transcript scrollbox scrolls with the wheel (the expected
  // gesture). Movement reporting stays off (it is the noisiest source of stray
  // sequences); any SGR fragment that still leaks into the prompt is stripped
  // in session.tsx's input handler. PageUp/PageDown also scroll.
  useMouse: true,
  enableMouseMovement: false,
})
