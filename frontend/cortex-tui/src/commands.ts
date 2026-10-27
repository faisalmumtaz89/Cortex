// Slash command registry shared by the palette and the command router.
// Descriptions and arg hints mirror cortex/app/command_service.py.

import { createSignal } from "solid-js"

export interface SlashCommand {
  name: string // canonical, without leading slash
  description: string
  argHint?: string // usage shown in ARG-HINT mode when the command takes arguments
  skill?: boolean // a skill: "/name request" is sent to the model as a message
}

// Canonical order (help first); the palette shows this order for an empty query.
export const SLASH_COMMANDS: SlashCommand[] = [
  { name: "help", description: "List available commands" },
  { name: "status", description: "Show the active model and the one that answered last" },
  { name: "model", description: "Pick a model (or /model <provider:model>)", argHint: "<provider:model>" },
  { name: "login", description: "Log in to a cloud provider (picker, then paste the key)", argHint: "<provider> <api_key> [url]" },
  { name: "clear", description: "Clear the conversation" },
  { name: "save", description: "Save the conversation to disk" },
  { name: "update", description: "Check for updates, or update Cortex", argHint: "[cortex]" },
  { name: "quit", description: "Exit Cortex" },
]

const BY_NAME = new Map(SLASH_COMMANDS.map((command) => [command.name, command]))

// Skills the worker found (skill.list); a command of the same name wins.
const [skillCommands, setSkillCommands] = createSignal<SlashCommand[]>([])

export function setSkills(skills: { name: string; description: string }[]): void {
  setSkillCommands(
    skills
      .filter((skill) => !BY_NAME.has(skill.name) && skill.name !== "exit")
      .map((skill) => ({ name: skill.name, description: skill.description, skill: true })),
  )
}

export function skills(): SlashCommand[] {
  return skillCommands()
}

function allCommands(): SlashCommand[] {
  return [...SLASH_COMMANDS, ...skillCommands()]
}

/** Every accepted command/alias name (for router validation). */
export const SLASH_COMMAND_NAMES: ReadonlySet<string> = new Set([
  ...SLASH_COMMANDS.map((command) => command.name),
  "exit", // alias for quit
])

export function findCommand(name: string): SlashCommand | undefined {
  const key = name.replace(/^\//, "").toLowerCase()
  return BY_NAME.get(key) ?? skillCommands().find((command) => command.name === key)
}

/** Whether the input runs a skill: "/<skill name>" alone or before a request. */
export function isSkillInput(input: string): boolean {
  const match = /^\/(\S+)(\s|$)/.exec(input.trim())
  return Boolean(match && findCommand(match[1])?.skill && match[1] === match[1].toLowerCase())
}

export function commandTakesArgs(name: string): boolean {
  return Boolean(findCommand(name)?.argHint)
}

/** Whether the command also runs with no arguments ("/update [cortex]"). */
export function argsOptional(name: string): boolean {
  return Boolean(findCommand(name)?.argHint?.startsWith("["))
}

// Providers whose /login also takes the URL requests go to (mirrors the worker).
const LOGIN_URL_ARGUMENT: Record<string, { hint: string; example: string }> = {
  azure: { hint: "<endpoint>", example: "https://<resource>.openai.azure.com" },
  "openai-compatible": { hint: "<base_url>", example: "https://api.deepseek.com/v1" },
}

/** The arg hint for an input in ARG-HINT mode; /login's depends on the provider typed. */
// What to type after "/model <provider>:" for providers whose models you name.
const NAMED_MODEL_HINT: Record<string, { argHint: string; description: string }> = {
  "openai-compatible": {
    argHint: "openai-compatible:<model>",
    description: "Type the model name as your endpoint calls it, e.g. deepseek/deepseek-v4.1-flash",
  },
  azure: {
    argHint: "azure:<deployment>",
    description: "Type the name of your Azure OpenAI deployment",
  },
}

export function argHintFor(
  command: SlashCommand,
  input: string,
  urlSaved: (provider: string) => boolean = () => false,
): SlashCommand {
  if (command.name === "model") {
    const provider = (input.trim().split(/\s+/)[1] ?? "").toLowerCase().split(":")[0] ?? ""
    const hint = NAMED_MODEL_HINT[provider]
    return hint ? { ...command, ...hint } : command
  }
  if (command.name !== "login") {
    return command
  }
  const provider = input.trim().split(/\s+/)[1]?.toLowerCase() ?? ""
  const url = LOGIN_URL_ARGUMENT[provider]
  if (url && urlSaved(provider)) {
    return {
      ...command,
      argHint: `${provider} <api_key> [${url.hint.replace(/[<>]/g, "")}]`,
      description: "Paste the key and press Enter: the saved URL is used (add one to change it)",
    }
  }
  if (url) {
    return {
      ...command,
      argHint: `${provider} <api_key> ${url.hint}`,
      description: `Paste the key, a space, then the URL, e.g. ${url.example}`,
    }
  }
  if (provider) {
    return { ...command, argHint: `${provider} <api_key>`, description: "Paste the key and press Enter" }
  }
  return command
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
  const commands = allCommands()
  if (!term) {
    return commands
  }
  const prefix = commands.filter((command) => command.name.startsWith(term))
  return prefix.sort((a, b) => {
    const aExact = a.name === term ? 0 : 1
    const bExact = b.name === term ? 0 : 1
    if (aExact !== bExact) {
      return aExact - bExact
    }
    return commands.indexOf(a) - commands.indexOf(b)
  })
}
