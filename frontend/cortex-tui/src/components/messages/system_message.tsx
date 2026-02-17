import type { MessageRecord } from "../../context/store"
import { UI_PALETTE } from "../ui_palette"

export function SystemMessage(props: { message: MessageRecord; index: number }) {
  if (!props.message.content) {
    return <></>
  }

  return (
    <box
      flexDirection="column"
      marginTop={props.index === 0 ? 0 : 1}
      border={["left"]}
      borderColor={UI_PALETTE.textMuted}
      paddingLeft={2}
      paddingTop={1}
      paddingBottom={1}
      backgroundColor={UI_PALETTE.panel}
    >
      <text fg={UI_PALETTE.textMuted}>System</text>
      <text fg={UI_PALETTE.textMuted}>{props.message.content}</text>
    </box>
  )
}
