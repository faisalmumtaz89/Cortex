import type { KeyEvent, RGBA } from "@opentui/core"
import { For, Show } from "solid-js"
import { UI_PALETTE } from "./ui_palette"

// The single "pick one of N" idiom shared by the slash palette, the permission
// menu, and the model picker: Up/Down (+ Ctrl+P/N) navigate, ❯ caret +
// highlight band mark the selection, Enter confirms, Esc cancels. No numbers.

export type SelectionAction = "up" | "down" | "enter" | "tab" | "cancel" | null

/** Classify a key event into the one selection idiom. Callers own the index
 * math (wraparound) and the enter/cancel actions. */
export function classifySelectionKey(event: KeyEvent): SelectionAction {
  const key = String(event.name ?? "").toLowerCase()
  const shift = Boolean((event as { shift?: boolean }).shift)
  const ctrl = Boolean((event as { ctrl?: boolean }).ctrl)
  if (key === "up" || (ctrl && key === "p")) return "up"
  if (key === "down" || (ctrl && key === "n")) return "down"
  if (key === "tab") return "tab"
  if (
    !shift &&
    (key === "return" || key === "linefeed" || key === "kpenter" || key === "numpadenter" || key.includes("enter"))
  ) {
    return "enter"
  }
  if (key === "escape" || key === "esc") return "cancel"
  return null
}

export interface SelectionTag {
  text: string
  color: RGBA
}

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

const NAME_COLUMN = 12

/** Left-pad a primary label to the shared name column so descriptions align. */
export function padPrimary(label: string): string {
  return label.length >= NAME_COLUMN ? `${label} ` : label + " ".repeat(NAME_COLUMN - label.length)
}

interface RowProps<T> {
  items: T[]
  selectedIndex: number
  getPrimary: (item: T) => string
  getSecondary?: (item: T) => string | undefined
  getTag?: (item: T) => SelectionTag | undefined
  getDanger?: (item: T) => boolean
  /** Non-selectable section-header rows (muted label, no caret/tag). Callers
   * own the index math and must skip these when moving the selection. */
  isHeader?: (item: T) => boolean
  maxVisibleRows?: number
}

/** Just the windowed selectable rows (no panel chrome) — used directly by the
 * permission menu inside its own box. */
export function SelectionRows<T>(props: RowProps<T>) {
  const maxRows = () => props.maxVisibleRows ?? 8
  const windowStart = () =>
    props.selectedIndex < maxRows() ? 0 : props.selectedIndex - maxRows() + 1
  const visible = () => props.items.slice(windowStart(), windowStart() + maxRows())
  const hiddenTail = () => Math.max(0, props.items.length - (windowStart() + maxRows()))

  return (
    <box flexDirection="column" flexShrink={0}>
      <For each={visible()}>
        {(item, i) => {
          const absoluteIndex = () => windowStart() + i()
          const selected = () => absoluteIndex() === props.selectedIndex
          const secondary = () => props.getSecondary?.(item)
          const tag = () => props.getTag?.(item)
          const danger = () => Boolean(props.getDanger?.(item))
          const primaryColor = () =>
            danger() ? UI_PALETTE.statusError : selected() ? UI_PALETTE.accent : UI_PALETTE.text
          if (props.isHeader?.(item)) {
            return (
              <box flexShrink={0}>
                <text fg={UI_PALETTE.textMuted}>{props.getPrimary(item)}</text>
              </box>
            )
          }
          return (
            <box
              flexShrink={0}
              flexDirection="row"
              justifyContent="space-between"
              backgroundColor={selected() ? UI_PALETTE.codeBackground : undefined}
            >
              <text>
                <span style={{ fg: selected() ? UI_PALETTE.accent : UI_PALETTE.textMuted }}>
                  {selected() ? "❯ " : "  "}
                </span>
                <span style={{ fg: primaryColor() }}>{props.getPrimary(item)}</span>
                <Show when={secondary()}>
                  <span style={{ fg: selected() ? UI_PALETTE.text : UI_PALETTE.textMuted }}>{secondary()}</span>
                </Show>
              </text>
              <Show when={tag()}>
                {/* Hard 1-col gutter so a wide primary can never glue to the tag. */}
                <text flexShrink={0} marginLeft={1} fg={tag()!.color}>
                  {tag()!.text}
                </text>
              </Show>
            </box>
          )
        }}
      </For>
      <Show when={hiddenTail() > 0}>
        <text fg={UI_PALETTE.textMuted}>{`  +${hiddenTail()} more`}</text>
      </Show>
    </box>
  )
}

export interface SelectionTabs {
  labels: string[]
  activeIndex: number
}

/** Full overlay: the ┃ panel chrome + optional title + rows + footer hint.
 * Used by the slash palette (list mode) and the model picker. An optional tab
 * bar renders on the title row (active tab accent+bold, others muted); the
 * CALLER owns tab state and which rows are visible. */
export function SelectionList<T>(
  props: RowProps<T> & {
    title?: string
    footer: string
    borderColor?: RGBA
    emptyLabel?: string
    tabs?: SelectionTabs
  },
) {
  return (
    <box
      flexDirection="column"
      flexShrink={0}
      backgroundColor={UI_PALETTE.panel}
      paddingLeft={2}
      paddingRight={1}
      border={["left"]}
      borderColor={props.borderColor ?? UI_PALETTE.accent}
      customBorderChars={SPINE_BORDER}
    >
      {/* One blank row between title / rows / footer so the menu breathes. */}
      <Show when={props.title}>
        <box
          flexShrink={0}
          marginBottom={1}
          flexDirection="row"
          justifyContent="space-between"
        >
          <text fg={UI_PALETTE.textMuted}>{props.title}</text>
          <Show when={props.tabs}>
            <text flexShrink={0} marginLeft={1}>
              <For each={props.tabs!.labels}>
                {(label, index) => (
                  <>
                    <Show when={index() > 0}>
                      <span style={{ fg: UI_PALETTE.textMuted }}>{" ─ "}</span>
                    </Show>
                    <Show
                      when={index() === props.tabs!.activeIndex}
                      fallback={<span style={{ fg: UI_PALETTE.textMuted }}>{label}</span>}
                    >
                      <strong>
                        <span style={{ fg: UI_PALETTE.accent }}>{label}</span>
                      </strong>
                    </Show>
                  </>
                )}
              </For>
            </text>
          </Show>
        </box>
      </Show>
      <Show
        when={props.items.length > 0}
        fallback={<text fg={UI_PALETTE.textMuted}>{props.emptyLabel ?? "No matches"}</text>}
      >
        <SelectionRows
          items={props.items}
          selectedIndex={props.selectedIndex}
          getPrimary={props.getPrimary}
          getSecondary={props.getSecondary}
          getTag={props.getTag}
          getDanger={props.getDanger}
          isHeader={props.isHeader}
          maxVisibleRows={props.maxVisibleRows}
        />
      </Show>
      <box flexShrink={0} marginTop={1}>
        <text fg={UI_PALETTE.textMuted}>{props.footer}</text>
      </box>
    </box>
  )
}
