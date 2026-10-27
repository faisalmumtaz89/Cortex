// The /login key as the input shows it (src/lib/secret_input.ts).
import { describe, expect, test } from "bun:test"
import { applyEdit, masked } from "../src/lib/secret_input"

// Types, deletes or pastes like the textarea: returns the new text and shown text.
function edit(text: string, shown: string, shownAfter: string, cursor: number) {
  const after = applyEdit(text, shown, shownAfter, cursor)
  return { text: after, shown: masked(after) }
}

describe("the /login key in the input", () => {
  test("is shown as one • per character, the provider and URL as typed", () => {
    expect(masked("/login openai-compatible sk-abc https://x/v1")).toBe(
      "/login openai-compatible •••••• https://x/v1",
    )
    expect(masked("/login openai")).toBe("/login openai")
    expect(masked("/model openai:gpt-5.5")).toBe("/model openai:gpt-5.5")
  })

  test("typing the key keeps every character, though only bullets are shown", () => {
    let state = { text: "/login openai ", shown: "/login openai " }
    for (const ch of "sk-12") {
      const shownAfter = state.shown + ch
      state = edit(state.text, state.shown, shownAfter, shownAfter.length)
    }
    expect(state).toEqual({ text: "/login openai sk-12", shown: "/login openai •••••" })
  })

  test("backspace, delete and typing in the middle change the character at the cursor", () => {
    const text = "/login openai abcdef"
    const shown = masked(text) // "/login openai ••••••"
    // Backspace after the third bullet removes "c".
    expect(edit(text, shown, shown.slice(0, 16) + shown.slice(17), 16).text).toBe("/login openai abdef")
    // Delete at the third bullet removes it too.
    expect(edit(text, shown, shown.slice(0, 16) + shown.slice(17), 16).text).toBe("/login openai abdef")
    // Typing "X" after the second bullet.
    const typed = shown.slice(0, 16) + "X" + shown.slice(16)
    expect(edit(text, shown, typed, 17)).toEqual({ text: "/login openai abXcdef", shown: "/login openai •••••••" })
  })

  test("a pasted line is hidden at once, and clearing the input clears the key", () => {
    const pasted = edit("", "", "/login anthropic sk-ant-123", 27)
    expect(pasted.shown).toBe("/login anthropic ••••••••••")
    expect(edit(pasted.text, pasted.shown, "", 0).text).toBe("")
  })

  test("adding the URL after the key keeps the key", () => {
    const text = "/login azure key1"
    const shown = masked(text)
    const after = shown + " https://r.example"
    expect(edit(text, shown, after, after.length).text).toBe("/login azure key1 https://r.example")
  })
})
