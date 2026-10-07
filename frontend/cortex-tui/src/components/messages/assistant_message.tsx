import { For, Match, Show, Switch, createMemo } from "solid-js"
import type { MessagePart, MessageRecord } from "../../context/store"
import { Markdown } from "../markdown"
import { ToolBlock } from "../tool_block"
import { UI_PALETTE, formatDuration, titlecase } from "../ui_palette"

type ToolPart = Extract<MessagePart, { type: "tool" }>
type Segment =
  | { kind: "text"; text: string }
  | { kind: "tool"; part: ToolPart }
  | { kind: "collapsed"; count: number }

// Above this many tool calls in one turn, older ones fold to a summary line
// (Ctrl+O reveals them all) so long turns don't bury the answer.
const MAX_VISIBLE_TOOLS = 6

/** One blank row when narration (prose) meets a tool group, so interleaved
 * text and tool calls don't read as one dense wall. Consecutive one-line tools
 * pack tight; multi-line tool blocks (Ctrl+O expanded output, or an inline
 * diff) get a separating blank so blocks don't abut. */
function segmentMarginTop(segments: Segment[], index: number, toolsExpanded: boolean): number {
  if (index <= 0) {
    return 0
  }
  const previous = segments[index - 1]
  const current = segments[index]
  const group = (kind: Segment["kind"]) => (kind === "text" ? "prose" : "tool")
  if (group(previous.kind) !== group(current.kind)) {
    return 1
  }
  if (previous.kind === "tool" && current.kind === "tool") {
    const previousHasDiff =
      Boolean(previous.part.diff) && previous.part.state === "completed"
    if (toolsExpanded || previousHasDiff) {
      return 1
    }
  }
  return 0
}

// Tool segment wrappers are cached BY PART IDENTITY so re-running toSegments
// (every text delta / new part) hands <For> the same wrapper object for an
// unchanged tool — otherwise every recompute would remount every tool row,
// restarting its fade-in and wiping its min-dwell display state. Parts keep a
// stable identity because the store merges updates in place (upsertToolPart).
const TOOL_SEGMENT_CACHE = new WeakMap<object, Segment>()

function toolSegment(part: Extract<MessagePart, { type: "tool" }>): Segment {
  let segment = TOOL_SEGMENT_CACHE.get(part)
  if (!segment) {
    segment = { kind: "tool", part }
    TOOL_SEGMENT_CACHE.set(part, segment)
  }
  return segment
}

/** Collapse the ordered part stream into renderable segments: consecutive
 * text deltas merge into one block; tool calls stay in place. */
function toSegments(parts: MessagePart[]): Segment[] {
  const segments: Segment[] = []
  for (const part of parts) {
    if (part.type === "text") {
      const last = segments[segments.length - 1]
      if (last && last.kind === "text") {
        last.text += part.delta
      } else {
        segments.push({ kind: "text", text: part.delta })
      }
      continue
    }
    if (part.type === "tool") {
      segments.push(toolSegment(part))
    }
  }
  return segments.filter((segment) => segment.kind !== "text" || segment.text.trim().length > 0)
}

export function AssistantMessage(props: {
  message: MessageRecord
  index: number
  toolsExpanded: boolean
}) {
  const segments = createMemo(() => {
    const fromParts = toSegments(props.message.parts)
    if (fromParts.length > 0) {
      return fromParts
    }
    // Fallback for turns without part data (e.g. RPC fallback fill).
    const content = props.message.content
    return content ? ([{ kind: "text", text: content }] satisfies Segment[]) : []
  })

  // Fold older tool calls into a single "N earlier steps" line when the turn
  // ran many tools, unless the user expanded with Ctrl+O.
  const displaySegments = createMemo<Segment[]>(() => {
    const all = segments()
    const toolCount = all.filter((segment) => segment.kind === "tool").length
    if (props.toolsExpanded || toolCount <= MAX_VISIBLE_TOOLS) {
      return all
    }
    const hideBefore = toolCount - MAX_VISIBLE_TOOLS
    const result: Segment[] = []
    let seenTools = 0
    let hidden = 0
    let markerPlaced = false
    for (const segment of all) {
      if (segment.kind === "tool") {
        seenTools += 1
        if (seenTools <= hideBefore) {
          hidden += 1
          continue
        }
      }
      if (hidden > 0 && !markerPlaced) {
        result.push({ kind: "collapsed", count: hidden })
        markerPlaced = true
      }
      result.push(segment)
    }
    if (hidden > 0 && !markerPlaced) {
      result.push({ kind: "collapsed", count: hidden })
    }
    return result
  })

  const placeholder = () => {
    if (segments().length > 0) {
      return ""
    }
    if (props.message.final) {
      // Narrow markers only: wide emoji (⚠/⏹) corrupt when the palette overlays.
      return props.message.interrupted
        ? "✕ Interrupted before any output."
        : "✕ No response — the model call failed (see error below)."
    }
    return "Thinking..."
  }

  const modeLabel = () => titlecase(props.message.mode ?? "chat")
  const modelLabel = () => props.message.modelLabel ?? "model unavailable"
  const durationLabel = () => formatDuration(props.message.elapsedMs)

  // No top role label — the reply is identified by its indent onto the shared
  // rail and the metadata footer below (opencode-style). This keeps the answer
  // reading as clean prose rather than a labelled block.
  return (
    <box
      flexDirection="column"
      flexShrink={0}
      marginTop={props.index === 0 ? 0 : 1}
      paddingLeft={3}
      // Cap prose width on wide terminals for readability (opencode has no cap,
      // so answers run edge-to-edge on ultrawide displays); ignored when the
      // terminal is already narrower than this.
      maxWidth={104}
    >
      <Show when={placeholder().length > 0}>
        <text fg={UI_PALETTE.text}>{placeholder()}</text>
      </Show>
      {/* Segments live in their own box so <For> re-creation can never
          reorder them past the metadata footer below. */}
      <box flexDirection="column" flexShrink={0}>
        <For each={displaySegments()}>
          {(segment, index) => (
            <box flexShrink={0} marginTop={segmentMarginTop(displaySegments(), index(), props.toolsExpanded)}>
              <Switch>
                <Match when={segment.kind === "text"}>
                  <Markdown content={(segment as { kind: "text"; text: string }).text.trim()} />
                </Match>
                <Match when={segment.kind === "collapsed"}>
                  <text fg={UI_PALETTE.textMuted}>
                    {`⋮ ${(segment as { kind: "collapsed"; count: number }).count} earlier steps (ctrl+o to expand)`}
                  </text>
                </Match>
                <Match when={segment.kind === "tool"}>
                  <ToolBlock
                    tool={(segment as { kind: "tool"; part: ToolPart }).part.tool}
                    state={(segment as { kind: "tool"; part: ToolPart }).part.state}
                    input={(segment as { kind: "tool"; part: ToolPart }).part.input}
                    output={(segment as { kind: "tool"; part: ToolPart }).part.output}
                    error={(segment as { kind: "tool"; part: ToolPart }).part.error}
                    diff={(segment as { kind: "tool"; part: ToolPart }).part.diff}
                    completedAtMs={(segment as { kind: "tool"; part: ToolPart }).part.completedAtMs}
                    expanded={props.toolsExpanded}
                  />
                </Match>
              </Switch>
            </box>
          )}
        </For>
      </box>
      <Show when={props.message.final}>
        <box flexShrink={0} marginTop={1}>
          <text fg={UI_PALETTE.textMuted}>
            <span style={{ fg: UI_PALETTE.accent }}>▣ </span>
            <span style={{ fg: UI_PALETTE.text }}>{modeLabel()}</span>
            <span style={{ fg: UI_PALETTE.textMuted }}> · {modelLabel()}</span>
            <Show when={durationLabel()}>
              <span style={{ fg: UI_PALETTE.textMuted }}> · {durationLabel()}</span>
            </Show>
            <Show when={props.message.interrupted}>
              <span style={{ fg: UI_PALETTE.statusBusy }}> · interrupted</span>
            </Show>
          </text>
        </box>
      </Show>
    </box>
  )
}
