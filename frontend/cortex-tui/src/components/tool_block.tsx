import { Show } from "solid-js"
import { ToolInline } from "./tool_inline"
import { UI_PALETTE } from "./ui_palette"

type ToolState = "pending" | "running" | "completed" | "error"

export function ToolBlock(props: {
  tool: string
  state: ToolState
  output?: string
  error?: string
}) {
  const outputPreview = () => {
    if (!props.output || props.state !== "completed") {
      return undefined
    }
    const lines = String(props.output).split("\n")
    if (lines.length <= 10) {
      return props.output
    }
    const hidden = lines.length - 10
    return `${lines.slice(0, 10).join("\n")}\n… (${hidden} more lines)`
  }

  return (
    <box flexDirection="column" paddingLeft={3}>
      <ToolInline tool={props.tool} state={props.state} error={props.error} />
      <Show when={outputPreview()}>
        <text fg={UI_PALETTE.textMuted}>{outputPreview()}</text>
      </Show>
    </box>
  )
}
