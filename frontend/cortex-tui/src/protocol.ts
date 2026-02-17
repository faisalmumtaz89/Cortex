import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import readline from "node:readline"

export type SessionStatus = "idle" | "busy" | "retry"
export type EventType =
  | "session.status"
  | "message.updated"
  | "message.part.updated"
  | "permission.asked"
  | "permission.replied"
  | "session.error"
  | "system.notice"

export interface EventEnvelope {
  session_id: string
  seq: number
  ts_ms: number
  event_type: EventType
  payload: Record<string, unknown>
}

interface RpcSuccess {
  jsonrpc: "2.0"
  id: number
  result: Record<string, unknown>
}

interface RpcFailure {
  jsonrpc: "2.0"
  id?: number
  error: {
    code: number
    message: string
    data?: Record<string, unknown>
  }
}

interface RpcEventFrame {
  jsonrpc: "2.0"
  method: "event"
  params: EventEnvelope
}

type RpcIncoming = RpcSuccess | RpcFailure | RpcEventFrame

type PendingRequest = {
  resolve: (value: Record<string, unknown>) => void
  reject: (error: Error) => void
  timer: NodeJS.Timeout
}

export class RpcClient {
  private process: ChildProcessWithoutNullStreams
  private lineReader: readline.Interface
  private nextId = 1
  private pending = new Map<number, PendingRequest>()
  private listeners = new Set<(event: EventEnvelope) => void>()
  private readonly requestTimeoutMs: number
  private closed = false
  private workerFailed = false
  private readonly debugWorkerStderr = process.env.CORTEX_TUI_DEBUG_WORKER_STDERR === "1"
  private readonly stderrRingLimit = 80
  private stderrRing: string[] = []

  constructor(workerCommand: string, workerArgs: string[], cwd: string) {
    const timeoutFromEnv = Number.parseInt(process.env.CORTEX_WORKER_REQUEST_TIMEOUT_MS ?? "", 10)
    this.requestTimeoutMs = Number.isFinite(timeoutFromEnv) && timeoutFromEnv > 0 ? timeoutFromEnv : 120_000

    this.process = spawn(workerCommand, workerArgs, {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    })

    this.lineReader = readline.createInterface({ input: this.process.stdout })
    this.lineReader.on("line", (line) => this.handleLine(line))

    this.process.stderr.on("data", (chunk) => {
      const text = String(chunk || "")
      this.captureWorkerStderr(text)
    })
    this.process.on("error", (error) => {
      this.failWorker(new Error(`Failed to spawn worker process: ${error.message}`))
    })
    this.process.on("exit", (code, signal) => {
      if (this.closed) {
        return
      }
      const summary = signal
        ? `Worker exited by signal ${signal}`
        : `Worker exited with code ${String(code ?? "unknown")}`
      this.failWorker(new Error(summary))
    })
  }

  onEvent(listener: (event: EventEnvelope) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  async request(method: string, params: Record<string, unknown>, timeoutMs?: number): Promise<Record<string, unknown>> {
    if (this.closed || this.workerFailed) {
      throw new Error("Worker process is not available")
    }

    const id = this.nextId++
    const payload = JSON.stringify({
      jsonrpc: "2.0",
      id,
      method,
      params,
    })

    const promise = new Promise<Record<string, unknown>>((resolve, reject) => {
      const timeout = timeoutMs && timeoutMs > 0 ? timeoutMs : this.requestTimeoutMs
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`RPC request timed out after ${timeout}ms: ${method}`))
      }, timeout)
      this.pending.set(id, { resolve, reject, timer })
    })

    this.process.stdin.write(`${payload}\n`, (error) => {
      if (!error) {
        return
      }
      const pending = this.pending.get(id)
      if (!pending) {
        return
      }
      this.pending.delete(id)
      clearTimeout(pending.timer)
      pending.reject(new Error(`Failed to send RPC request (${method}): ${error.message}`))
    })
    return promise
  }

  close(): void {
    this.closed = true
    this.lineReader.close()
    this.rejectAllPending(new Error("RPC client closed"))
    this.process.stdin.end()
    this.process.kill()
  }

  private handleLine(line: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch {
      process.stderr.write(`[worker] invalid JSON frame: ${line}\n`)
      return
    }

    if (!parsed || typeof parsed !== "object") {
      process.stderr.write(`[worker] ignored non-object frame\n`)
      return
    }

    const frame = parsed as Record<string, unknown>
    const method = typeof frame.method === "string" ? frame.method : ""
    if (method === "event") {
      const params = frame.params
      if (!params || typeof params !== "object") {
        process.stderr.write("[worker] ignored malformed event frame\n")
        return
      }
      for (const listener of this.listeners) {
        try {
          listener(params as EventEnvelope)
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          process.stderr.write(`[worker] event listener error: ${message}\n`)
        }
      }
      return
    }

    const id = frame.id
    if (typeof id !== "number") {
      return
    }

    const pending = this.pending.get(id)
    if (!pending) {
      return
    }
    this.pending.delete(id)
    clearTimeout(pending.timer)

    if ("result" in frame) {
      const result = frame.result
      if (result && typeof result === "object") {
        pending.resolve(result as Record<string, unknown>)
        return
      }
      pending.resolve({})
      return
    }

    const rawError = frame.error
    if (!rawError || typeof rawError !== "object") {
      pending.reject(new Error("Malformed RPC response: missing error payload"))
      return
    }

    const errorObj = rawError as Record<string, unknown>
    const base = typeof errorObj.message === "string" && errorObj.message.trim() ? errorObj.message : "RPC request failed"
    const data = errorObj.data
    const nested = data && typeof data === "object" ? (data as Record<string, unknown>).error : undefined
    const detail = typeof nested === "string" ? nested.trim() : ""
    pending.reject(new Error(detail ? `${base}: ${detail}` : base))
  }

  private captureWorkerStderr(text: string): void {
    const lines = text
      .split(/\r?\n/g)
      .map((line) => line.trim())
      .filter(Boolean)
    if (lines.length === 0) {
      return
    }

    for (const line of lines) {
      this.stderrRing.push(line)
      if (this.stderrRing.length > this.stderrRingLimit) {
        this.stderrRing = this.stderrRing.slice(-this.stderrRingLimit)
      }
      if (this.debugWorkerStderr) {
        process.stderr.write(`[worker] ${line}\n`)
      }
    }
  }

  private failWorker(error: Error): void {
    if (this.workerFailed) {
      return
    }
    this.workerFailed = true
    this.rejectAllPending(error)
    if (!this.debugWorkerStderr && this.stderrRing.length > 0) {
      const excerpt = this.stderrRing.slice(-8).join("\n")
      process.stderr.write(`[worker] ${error.message}\n${excerpt}\n`)
      return
    }
    process.stderr.write(`[worker] ${error.message}\n`)
  }

  private rejectAllPending(error: Error): void {
    for (const [id, pending] of this.pending.entries()) {
      clearTimeout(pending.timer)
      pending.reject(error)
      this.pending.delete(id)
    }
  }
}
