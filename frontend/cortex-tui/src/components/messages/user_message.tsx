import type { MessageRecord } from "../../context/store"
import { screenColumns } from "../../lib/terminal_size"
import { UI_PALETTE } from "../ui_palette"
import { WrappedText } from "../wrapped_text"

// Columns a message's rows lose to its rail and the padding around them (1 + 2 + 2).
const MESSAGE_INDENT = 5

// A single left "spine" (┃) instead of a full box — one consistent role marker.
const SPINE_BORDER = {
  topLeft: "",
  bottomLeft: "",
  vertical: "┃",
  topRight: "",
  bottomRight: "",
  horizontal: " ",
  bottomT: "",
  topT: "",
  cross: "",
  leftT: "",
  rightT: "",
}

export function UserMessage(props: { message: MessageRecord; index: number }) {
  const text = () => props.message.content

  if (!text()) {
    return <></>
  }

  // The accent spine + subtle panel fill distinguish the user's turn; no role
  // label or timestamp keeps it quiet. Content lands on the shared left rail.
  return (
    <box
      flexDirection="column"
      flexShrink={0}
      marginTop={props.index === 0 ? 0 : 1}
      border={["left"]}
      borderColor={UI_PALETTE.accent}
      customBorderChars={SPINE_BORDER}
    >
      <box flexShrink={0} paddingLeft={2} paddingRight={2} backgroundColor={UI_PALETTE.panel}>
        <WrappedText text={text()} columns={screenColumns() - MESSAGE_INDENT} fg={UI_PALETTE.text} />
      </box>
    </box>
  )
}
