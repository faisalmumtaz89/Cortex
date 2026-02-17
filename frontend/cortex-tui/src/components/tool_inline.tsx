import { colorForToolState } from "./ui_palette"

type ToolState = "pending" | "running" | "completed" | "error"

export function ToolInline(props: {
  tool: string
  state: ToolState
  error?: string
}) {
  const icon = () => {
    if (props.state === "running") return "~"
    if (props.state === "completed") return "✓"
    if (props.state === "error") return "✗"
    return "·"
  }
  const color = () => {
    return colorForToolState(props.state)
  }
  const statusLabel = () => {
    if (props.state === "running") return "running"
    if (props.state === "completed") return "completed"
    if (props.state === "error") return "failed"
    return "pending"
  }

  return (
    <text fg={color()}>
      {icon()} {props.tool} ({statusLabel()})
      {props.error ? `: ${props.error}` : ""}
    </text>
  )
}
