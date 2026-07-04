// Single exit path for the sidecar. OpenTUI's Ctrl+C / signal handling only
// destroys the renderer (restores the terminal) — it never exits the process,
// and our app legitimately holds the event loop open (shared spinner interval,
// worker stdio pipes). So every exit trigger (Ctrl+C key, SIGINT/SIGTERM from
// a parent, /quit) funnels here: destroy the renderer first, then exit on a
// short delay so an in-flight render pass can finalize the terminal restore.
// Exiting closes the worker's stdin → the worker shuts down → its atexit stops
// any managed lumen-server, so the whole tree tears down from one press.

let exiting = false
let destroyRenderer: (() => void) | null = null
let killWorker: (() => void) | null = null

export function registerRendererDestroy(destroy: () => void): void {
  destroyRenderer = destroy
}

/** Registered by the RPC layer: SIGTERMs the worker so the lumen-server
 * teardown starts immediately instead of waiting for stdin EOF. */
export function registerWorkerKill(kill: () => void): void {
  killWorker = kill
}

export function exitCortex(code = 0): void {
  if (exiting) {
    return
  }
  exiting = true
  try {
    killWorker?.()
  } catch {
    // EOF on our exit still tears the worker down; this is the fast path.
  }
  try {
    destroyRenderer?.()
  } catch {
    // Terminal restore is best-effort; exiting matters more.
  }
  setTimeout(() => process.exit(code), 50)
}
