import { For, Show } from "solid-js"
import type { MessagePart, MessageRecord } from "../../context/store"
import { ToolBlock } from "../tool_block"
import { UI_PALETTE, formatDuration, titlecase } from "../ui_palette"

export function AssistantMessage(props: {
  message: MessageRecord
  index: number
}) {
  const displayText = () => {
    if (props.message.content) {
      return props.message.content
    }
    return props.message.final ? "[No response generated]" : "Thinking..."
  }

  const modeLabel = () => titlecase(props.message.mode ?? "chat")
  const modelLabel = () => props.message.modelLabel ?? "model unavailable"
  const durationLabel = () => formatDuration(props.message.elapsedMs)

  const toolParts = () =>
    props.message.parts.filter((part): part is Extract<MessagePart, { type: "tool" }> => part.type === "tool")

  return (
    <box flexDirection="column" marginTop={props.index === 0 ? 0 : 1} gap={1}>
      <box flexDirection="column" paddingLeft={3}>
        <text fg={UI_PALETTE.textMuted}>Cortex</text>
        <text fg={UI_PALETTE.text}>{displayText()}</text>
      </box>
      <Show when={toolParts().length > 0}>
        <For each={toolParts()}>
          {(part) => <ToolBlock tool={part.tool} state={part.state} output={part.output} error={part.error} />}
        </For>
      </Show>
      <Show when={props.message.final || props.message.content.length > 0}>
        <box paddingLeft={3}>
          <text fg={UI_PALETTE.textMuted}>
            <span style={{ fg: UI_PALETTE.accent }}>▣ </span>
            <span style={{ fg: UI_PALETTE.text }}>{modeLabel()}</span>
            <span style={{ fg: UI_PALETTE.textMuted }}> · {modelLabel()}</span>
            <Show when={durationLabel()}>
              <span style={{ fg: UI_PALETTE.textMuted }}> · {durationLabel()}</span>
            </Show>
          </text>
        </box>
      </Show>
    </box>
  )
}
