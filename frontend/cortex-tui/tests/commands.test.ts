// Which inputs run a slash command (src/commands.ts). A message that only
// starts with a command's name must reach the model: "Clear up the docstring"
// once cleared the conversation.
// Run via `bun test` (wired into the pytest gate by tests/test_frontend_unit.py).
import { expect, test } from "bun:test"
import { slashCommandFor } from "../src/commands"

test("a slash input is a command as typed", () => {
  expect(slashCommandFor("/clear")).toBe("/clear")
  expect(slashCommandFor("  /model openai:gpt-5.5 ")).toBe("/model openai:gpt-5.5")
})

test("a command name typed alone runs it", () => {
  expect(slashCommandFor("quit")).toBe("/quit")
  expect(slashCommandFor("Help")).toBe("/help")
  expect(slashCommandFor(" exit ")).toBe("/exit")
})

test("a message that starts with a command name is a message", () => {
  for (const message of [
    "Clear up the docstring wording in cli.py",
    "Update the module docstring",
    "Help me write a docstring",
    "status of the tests?",
    "Model the conversions as a dataclass?",
    "quit using globals in core.py",
    "clearly wrong",
  ]) {
    expect(slashCommandFor(message)).toBeNull()
  }
})
