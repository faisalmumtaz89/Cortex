// Slash command registry shared by the palette and the command router.
// Descriptions and arg hints mirror cortex/app/command_service.py.

export interface SlashCommand {
  name: string // canonical, without leading slash
  description: string
  argHint?: string // usage shown in ARG-HINT mode when the command takes arguments
}

// Canonical order (help first); the palette shows this order for an empty query.
export const SLASH_COMMANDS: SlashCommand[] = [
  { name: "help", description: "List available commands" },
  { name: "status", description: "Show model & session status" },
  { name: "gpu", description: "Show GPU / memory status" },
  { name: "model", description: "Pick a model (or /model <name | provider:model>)", argHint: "<name | provider:model>" },
  { name: "download", description: "Download a local model (Lumen)", argHint: "<model[:quant]>  \u00b7  /download cancel" },
  { name: "login", description: "Log in to a cloud provider (picker, then paste the key)", argHint: "<provider> <api_key>" },
  { name: "benchmark", description: "Benchmark local token throughput", argHint: "[tokens] [--prompt <text>]" },
  { name: "clear", description: "Clear the conversation" },
  { name: "save", description: "Save the conversation to disk" },
  { name: "setup", description: "Load the first local model" },
  { name: "update", description: "Check for updates, or update Lumen / Cortex", argHint: "[lumen|cortex]" },
  { name: "quit", description: "Exit Cortex" },
]

const BY_NAME = new Map(SLASH_COMMANDS.map((command) => [command.name, command]))

/** Every accepted command/alias name (for router validation). */
export const SLASH_COMMAND_NAMES: ReadonlySet<string> = new Set([
  ...SLASH_COMMANDS.map((command) => command.name),
  "exit", // alias for quit
])

export function findCommand(name: string): SlashCommand | undefined {
  return BY_NAME.get(name.replace(/^\//, "").toLowerCase())
}

export function commandTakesArgs(name: string): boolean {
  return Boolean(findCommand(name)?.argHint)
}

/** The leading command token of an input like "/model azure:x" → "model". */
export function commandToken(input: string): string {
  return input.replace(/^\//, "").toLowerCase().split(/\s+/, 1)[0] ?? ""
}

/**
 * Commands matching the current "/prefix" (prefix-only, exact-match first,
 * canonical order as tiebreak). Empty query returns all in canonical order.
 */
export function matchCommands(query: string): SlashCommand[] {
  const term = commandToken(query)
  if (!term) {
    return SLASH_COMMANDS
  }
  const prefix = SLASH_COMMANDS.filter((command) => command.name.startsWith(term))
  return prefix.sort((a, b) => {
    const aExact = a.name === term ? 0 : 1
    const bExact = b.name === term ? 0 : 1
    if (aExact !== bExact) {
      return aExact - bExact
    }
    return SLASH_COMMANDS.indexOf(a) - SLASH_COMMANDS.indexOf(b)
  })
}
