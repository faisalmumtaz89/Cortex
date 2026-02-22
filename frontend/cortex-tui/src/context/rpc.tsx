import { createContext, onCleanup, onMount, useContext, type ParentProps } from "solid-js"
import { RpcClient } from "../protocol"
import { useStore } from "./store"

type RpcContextValue = {
  client: RpcClient
  bootstrap: () => Promise<void>
  runCommand: (command: string) => Promise<boolean>
  submitInput: (text: string) => Promise<boolean>
  replyPermission: (requestID: string, reply: "allow_once" | "allow_always" | "reject") => Promise<boolean>
}

const RpcContext = createContext<RpcContextValue>()

const SLASH_COMMAND_ALIASES = new Set([
  "help",
  "status",
  "gpu",
  "model",
  "models",
  "download",
  "login",
  "clear",
  "save",
  "benchmark",
  "template",
  "finetune",
  "quit",
  "exit",
  "setup",
])

function splitCommandArgs(raw: string): string[] {
  const result: string[] = []
  let current = ""
  let quote: '"' | "'" | "" = ""
  let escaping = false

  for (const char of raw) {
    if (escaping) {
      current += char
      escaping = false
      continue
    }
    if (char === "\\") {
      escaping = true
      continue
    }
    if (quote) {
      if (char === quote) {
        quote = ""
        continue
      }
      current += char
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (current.length > 0) {
        result.push(current)
        current = ""
      }
      continue
    }
    current += char
  }

  if (current.length > 0) {
    result.push(current)
  }
  return result
}

function parseWorkerCommand(): { cmd: string; args: string[]; cwd: string } {
  const cwd = process.cwd()
  const envCmd = process.env.CORTEX_WORKER_CMD
  const envArgs = process.env.CORTEX_WORKER_ARGS

  if (envCmd) {
    const args = envArgs ? splitCommandArgs(envArgs) : []
    return { cmd: envCmd, args, cwd }
  }

  return {
    cmd: process.env.PYTHON || "python3",
    args: ["-m", "cortex", "--worker-stdio"],
    cwd,
  }
}

function parseTimeoutMs(raw: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(raw ?? "", 10)
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed
  }
  return fallback
}

function normalizeSlashCommand(raw: string): string | null {
  const trimmed = raw.trim()
  if (!trimmed) {
    return null
  }
  if (trimmed.startsWith("/")) {
    return trimmed
  }

  const parts = splitCommandArgs(trimmed)
  const first = (parts[0] || "").toLowerCase()
  if (!SLASH_COMMAND_ALIASES.has(first)) {
    return null
  }
  return `/${trimmed}`
}

export function RpcProvider(props: ParentProps) {
  const store = useStore()
  const worker = parseWorkerCommand()
  const client = new RpcClient(worker.cmd, worker.args, worker.cwd)

  client.onEvent((event) => store.applyEvent(event))

  const bootstrap = async () => {
    await client.request("app.handshake", {
      protocol_version: "1.0.0",
      client_name: "cortex-tui",
    })
    const session = await client.request("session.create_or_resume", {})
    store.setSessionID(String(session.session_id ?? ""))
    await refreshModels()
  }

  const refreshModels = async (): Promise<void> => {
    try {
      const models = await client.request("model.list", {})
      store.setModelStateFromList(models as Record<string, unknown>)
    } catch {
      // Non-fatal: model bar can stay on previous state if the list call fails.
    }
  }

  const runCommand = async (command: string): Promise<boolean> => {
    const sessionID = store.state.sessionID
    const normalized = command.trim()
    if (!sessionID || !normalized) {
      return false
    }

    const slashCommand = normalizeSlashCommand(normalized) || (normalized.startsWith("/") ? normalized : `/${normalized}`)
    const lowerCommand = slashCommand.toLowerCase()
    const commandTimeoutMs = parseTimeoutMs(
      process.env.CORTEX_WORKER_COMMAND_TIMEOUT_MS ?? process.env.CORTEX_WORKER_TURN_TIMEOUT_MS,
      1_800_000,
    )
    const downloadTimeoutMs = parseTimeoutMs(process.env.CORTEX_WORKER_DOWNLOAD_TIMEOUT_MS, commandTimeoutMs)
    const timeoutMs = lowerCommand.startsWith("/download")
      ? Math.max(downloadTimeoutMs, commandTimeoutMs)
      : commandTimeoutMs

    store.clearError()
    store.appendLocalMessage("user", slashCommand)
    try {
      const result = await client.request(
        "command.execute",
        {
          session_id: sessionID,
          command: slashCommand,
        },
        timeoutMs,
      )
      if (result.exit === true) {
        process.exit(0)
      }
      await refreshModels()
      return true
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      store.setError(message)
      return false
    }
  }

  const submitInput = async (text: string): Promise<boolean> => {
    const sessionID = store.state.sessionID
    const normalized = text.trim()
    if (!sessionID || !normalized) {
      return false
    }
    store.clearError()
    try {
      const maybeCommand = normalizeSlashCommand(normalized)
      if (maybeCommand) {
        return runCommand(maybeCommand)
      }

      if (store.state.activeModelLabel === "No model loaded") {
        const firstLocal = (store.state.firstLocalModelName || "").trim()
        if (firstLocal) {
          const loaded = await runCommand(`/model ${firstLocal}`)
          if (!loaded) {
            return false
          }
        } else {
          store.setError(
            "No model loaded. Run: download mlx-community/Nanbeige4.1-3B-bf16 --load",
          )
          return false
        }
      }

      const result = await client.request(
        "session.submit_user_input",
        {
          session_id: sessionID,
          user_input: normalized,
        },
        Number.parseInt(process.env.CORTEX_WORKER_TURN_TIMEOUT_MS ?? "1800000", 10),
      )
      if (result.ok === false) {
        const message = typeof result.error === "string" ? result.error : String(result.message ?? "Request failed")
        store.setError(message)
        return false
      }
      store.flushPendingEvents()
      store.applySubmitResultFallback(result)
      return true
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      store.setError(message)
      return false
    }
  }

  const replyPermission = async (
    requestID: string,
    reply: "allow_once" | "allow_always" | "reject",
  ): Promise<boolean> => {
    const sessionID = store.state.sessionID
    if (!sessionID || !requestID) {
      return false
    }
    store.clearError()
    try {
      await client.request("permission.reply", {
        session_id: sessionID,
        request_id: requestID,
        reply,
      })
      return true
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      store.setError(message)
      return false
    }
  }

  onMount(async () => {
    try {
      await bootstrap()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      store.setError(`Failed to initialize worker connection: ${message}`)
    }
  })

  onCleanup(() => {
    client.close()
  })

  return (
    <RpcContext.Provider
      value={{
        client,
        bootstrap,
        runCommand,
        submitInput,
        replyPermission,
      }}
    >
      {props.children}
    </RpcContext.Provider>
  )
}

export function useRpc() {
  const ctx = useContext(RpcContext)
  if (!ctx) {
    throw new Error("useRpc must be used inside RpcProvider")
  }
  return ctx
}
