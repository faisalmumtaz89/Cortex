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

    const slashCommand = normalized.startsWith("/") ? normalized : `/${normalized}`
    store.clearError()
    try {
      const result = await client.request("command.execute", {
        session_id: sessionID,
        command: slashCommand,
      })
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
    if (!sessionID || !text.trim()) {
      return false
    }
    store.clearError()
    try {
      if (text.startsWith("/")) {
        return runCommand(text.trim())
      }

      const result = await client.request(
        "session.submit_user_input",
        {
          session_id: sessionID,
          user_input: text.trim(),
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
