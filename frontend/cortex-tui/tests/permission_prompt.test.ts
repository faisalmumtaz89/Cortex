import { describe, expect, test } from "bun:test"
import { permissionLines, permissionShown, permissionShownInFull } from "../src/lib/permission_prompt"

const bash = (command: string) => ({
  permission: "bash",
  patterns: [command],
  metadata: { tool: "bash", arguments: { command } },
})

describe("permission prompt", () => {
  test("shows a bash call as its command, once", () => {
    expect(permissionLines(bash("make test"))).toEqual(["Run command: make test"])
  })

  test("a short call that fits is shown in full", () => {
    expect(permissionShownInFull(bash("make test"), 80, 24)).toBe(true)
  })

  test("a command just under the text limit, made too long by its prefix, is not", () => {
    // 80x20 leaves rows for no more than the 240 characters always shown.
    expect(permissionShownInFull(bash("echo " + "A".repeat(230)), 80, 20)).toBe(false)
  })

  test("a long command is shown whole when the free rows hold it", () => {
    const command = "python3 -c \"" + "x = 1; ".repeat(120) + "print(x)\""
    const shown = permissionShown(bash(command), 120, 40)
    expect(shown.full).toBe(true)
    expect(shown.lines[0]).toBe(`Run command: ${command}`)
  })

  test("a command longer than the free rows keeps its ends and is not shown in full", () => {
    const command = "echo " + "A".repeat(5000) + "; touch LONG.txt"
    const shown = permissionShown(bash(command), 80, 24)
    expect(shown.full).toBe(false)
    expect(shown.lines[0]).toContain("characters not shown⟩")
    expect(shown.lines[0].endsWith("touch LONG.txt")).toBe(true)
    expect(shown.lines[0].length).toBeLessThan(80 * 10)
  })

  test("a call that fits in text but not on a short screen is not", () => {
    const command = "echo " + "漢".repeat(100)
    expect(permissionShownInFull(bash(command), 200, 60)).toBe(true)
    expect(permissionShownInFull(bash(command), 80, 16)).toBe(false)
    expect(permissionShownInFull(bash("echo hi"), 30, 10)).toBe(false)
  })
})
