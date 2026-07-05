import type { RGBA } from "@opentui/core"
import { For, Show } from "solid-js"
import type { DiffData } from "../context/store"
import { createFadeIn, createMinDwell, mixRgba } from "../lib/animation"
import { displayPath } from "../lib/paths"
import { spinnerFrame } from "../lib/spinner"
import { terminalColumns } from "../lib/terminal_size"
import { DiffView } from "./diff_view"
import { UI_PALETTE } from "./ui_palette"

type ToolState = "pending" | "running" | "completed" | "error"

// TitleCase verbs for every tool so headlines are polished and stable across
// the running→completed lifecycle.
const TOOL_VERB: Record<string, string> = {
  read_file: "Read",
  list_dir: "List",
  bash: "Bash",
  search: "Search",
  edit_file: "Update",
  write_file: "Write",
}

function toolVerb(tool: string): string {
  return TOOL_VERB[tool] ?? tool
}

const SUMMARY_MAX_CHARS = 72
const PATH_MAX_CHARS = 56

function truncate(text: string, max: number = SUMMARY_MAX_CHARS): string {
  const flat = text.replace(/\s+/g, " ").trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

/** Keep the informative tail (filename + parents) of a long path on one line. */
function truncatePath(path: string, max: number): string {
  return path.length > max ? `…${path.slice(path.length - max + 1)}` : path
}

/** First pipeline stage of a shell command, quote-aware: `git status && …`
 * reads as `git status …`. Chains of &&/|/; are execution detail — the first
 * stage identifies the action without the punctuation soup. */
function firstShellStage(command: string): string {
  let quote: string | undefined
  for (let i = 0; i < command.length; i++) {
    const ch = command[i]
    if (quote) {
      if (ch === quote) {
        quote = undefined
      } else if (ch === "\\" && quote === '"') {
        i += 1 // escapes only apply inside double quotes
      }
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      continue
    }
    if (ch === "\\") {
      i += 1
      continue
    }
    if (ch === "|" || ch === ";" || ch === "&") {
      const stage = command.slice(0, i).trim()
      return stage ? `${stage} …` : command
    }
  }
  return command
}

/** Path budget for the headline, derived from the live terminal width (minus
 * the ◆/verb/()/·hint/rail overhead) and capped so it stays compact when wide. */
function pathBudget(): number {
  return Math.max(16, Math.min(terminalColumns() - 30, PATH_MAX_CHARS))
}

/** Compact, Claude Code-style argument summary for a tool call. Absolute
 * paths under the repo root display repo-relative (display only — the tool's
 * actual input is untouched). */
export function toolCallSummary(tool: string, input?: Record<string, unknown>): string {
  const args = input ?? {}
  const path = typeof args.path === "string" ? displayPath(args.path) : ""
  const budget = pathBudget()
  switch (tool) {
    case "bash":
      return truncate(firstShellStage(String(args.command ?? "")))
    case "read_file":
      return truncatePath(path, budget)
    case "list_dir":
      return truncatePath(path || ".", budget)
    case "search":
      return truncate(String(args.query ?? ""))
    case "edit_file":
      return truncatePath(path, budget)
    case "write_file":
      return truncatePath(path, budget)
    default: {
      const raw = Object.entries(args)
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(" ")
      return truncate(raw)
    }
  }
}

function glyphColor(state: ToolState): RGBA {
  if (state === "running" || state === "pending") {
    return UI_PALETTE.statusBusy
  }
  if (state === "error") {
    return UI_PALETTE.statusError
  }
  return UI_PALETTE.accent
}

/** One-word result hint shown inline on the headline for completed tools. */
function resultHint(tool: string, output?: string): string {
  const text = String(output ?? "")
  if (!text.trim()) {
    return ""
  }
  const lines = text.replace(/\r\n/g, "\n").split("\n")
  if (lines.length && lines[lines.length - 1] === "") {
    lines.pop() // drop the single trailing-newline empty line
  }
  if (lines.length <= 1) {
    return truncate(lines[0] ?? "", 40)
  }
  const unit = tool === "list_dir" ? "entries" : "lines"
  return `${lines.length} ${unit}`
}

export function ToolBlock(props: {
  tool: string
  state: ToolState
  input?: Record<string, unknown>
  output?: string
  error?: string
  expanded: boolean
  diff?: DiffData
  completedAtMs?: number
}) {
  // DISPLAY state: mirrors props.state but holds each live transition for a
  // minimum dwell. Fast tools complete in milliseconds — faster than a frame
  // (sometimes within the same tick as the row's mount) — so without this the
  // amber "running" diamond was never actually visible. A row that mounts
  // already-terminal presents the running phase first ONLY when the completion
  // is fresh (completedAtMs within the dwell window); historical rows
  // (restored transcripts, re-revealed folds) render their final state as-is.
  const STATE_DWELL_MS = 450
  const mountFreshTerminal =
    (props.state === "completed" || props.state === "error") &&
    props.completedAtMs !== undefined &&
    Date.now() - props.completedAtMs < STATE_DWELL_MS
  const displayState = createMinDwell<ToolState>(() => props.state, STATE_DWELL_MS, {
    initial: mountFreshTerminal ? "running" : undefined,
  })

  // Mount fade: row colors ease in from the background instead of popping.
  const fade = createFadeIn(240)
  const eased = (target: RGBA) => mixRgba(UI_PALETTE.codeBackground, target, fade())

  // A completed edit/write with a structured diff renders as a green/red diff.
  const showDiff = () =>
    Boolean(props.diff) &&
    displayState() === "completed" &&
    (props.tool === "edit_file" || props.tool === "write_file")

  // Reference-CLI style "Verb subject" — no paren wrapping; on long paths and
  // shell commands the parens read as pure punctuation noise.
  const headline = () => {
    if (showDiff() && props.diff) {
      // Use the tool's stable verb (edit→Update, write→Write) rather than the
      // diff op, so the label never flips between the running and completed
      // states (write_file would otherwise go Write→Create for a new file).
      return `${toolVerb(props.tool)} ${props.diff.path}`
    }
    const verb = toolVerb(props.tool)
    const summary = toolCallSummary(props.tool, props.input)
    return summary ? `${verb} ${summary}` : verb
  }

  const running = () => displayState() === "running" || displayState() === "pending"
  const errored = () => displayState() === "error" || (Boolean(props.error) && !running())

  const expandedOutput = () => {
    if (!props.expanded || displayState() !== "completed" || !props.output) {
      return undefined
    }
    const lines = String(props.output).split("\n")
    if (lines.length <= 40) {
      return props.output
    }
    return `${lines.slice(0, 40).join("\n")}\n… (${lines.length - 40} more lines)`
  }

  return (
    <box flexDirection="column" flexShrink={0}>
      {/* Every tool is a STABLE one-line row (no sub-line appearing and
          vanishing — that shifts layout mid-turn). Activity is DIM (textMuted)
          so the bright answer prose below reads as the output; the glyph alone
          carries the status: an amber SPINNER while running (motion makes even
          a brief run perceptible), a green ◆ when done, red ◆ on error. */}
      <text>
        <span style={{ fg: eased(glyphColor(displayState())) }}>
          {`${running() ? spinnerFrame() : "◆"} `}
        </span>
        <span style={{ fg: eased(UI_PALETTE.textMuted) }}>{headline()}</span>
        <Show when={!running() && !errored() && !showDiff() && resultHint(props.tool, props.output)}>
          <span style={{ fg: eased(UI_PALETTE.textMuted) }}>{` · ${resultHint(props.tool, props.output)}`}</span>
        </Show>
      </text>

      <Show when={errored()}>
        <box paddingLeft={2}>
          <text fg={UI_PALETTE.statusError}>{`⎿ ${truncate(String(props.error ?? "failed"), 120)}`}</text>
        </box>
      </Show>

      {/* A diff replaces the generic expanded-output block for edit/write. */}
      <Show when={showDiff() && props.diff} fallback={
        <Show when={expandedOutput()}>
          <box paddingLeft={2} backgroundColor={UI_PALETTE.codeBackground}>
            <For each={String(expandedOutput()).split("\n")}>
              {(line, i) => (
                <text fg={UI_PALETTE.textMuted}>{i() === 0 ? `⎿ ${line}` : `  ${line}`}</text>
              )}
            </For>
          </box>
        </Show>
      }>
        <DiffView diff={props.diff!} expanded={props.expanded} />
      </Show>
    </box>
  )
}
