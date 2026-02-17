import { Show } from "solid-js"
import type { MessageRecord } from "../../context/store"
import { UI_PALETTE, formatTimestamp } from "../ui_palette"

const USER_BORDER = {
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
  const metadata = () => formatTimestamp(props.message.createdTsMs)

  if (!text()) {
    return <></>
  }

  return (
    <box
      flexDirection="column"
      marginTop={props.index === 0 ? 0 : 1}
      border={["left"]}
      borderColor={UI_PALETTE.accent}
      customBorderChars={USER_BORDER}
    >
      <box
        flexDirection="column"
        paddingTop={1}
        paddingBottom={1}
        paddingLeft={2}
        paddingRight={2}
        backgroundColor={UI_PALETTE.panel}
      >
        <text fg={UI_PALETTE.accent}>You</text>
        <text fg={UI_PALETTE.text}>{text()}</text>
        <Show when={metadata()}>
          <text fg={UI_PALETTE.textMuted}>{metadata()}</text>
        </Show>
      </box>
    </box>
  )
}
