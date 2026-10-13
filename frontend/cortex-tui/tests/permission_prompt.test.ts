import { describe, expect, test } from "bun:test"
import { permissionLines, permissionShownInFull } from "../src/lib/permission_prompt"

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
    expect(permissionShownInFull(bash("echo " + "A".repeat(230)), 200, 60)).toBe(false)
  })

  test("a call that fits in text but not on a short screen is not", () => {
    const command = "echo " + "漢".repeat(100)
    expect(permissionShownInFull(bash(command), 200, 60)).toBe(true)
    expect(permissionShownInFull(bash(command), 80, 16)).toBe(false)
    expect(permissionShownInFull(bash("echo hi"), 30, 10)).toBe(false)
  })
})
