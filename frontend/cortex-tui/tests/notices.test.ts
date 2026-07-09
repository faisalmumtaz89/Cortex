// Unit tests for the system.notice transcription gate (src/lib/notices.ts).
// The load-bearing case: the worker's async update-check notice is emitted
// exactly ONCE per session — if the store dropped it because an unrelated
// slash command was in flight, it would be lost forever. Out-of-band origins
// must therefore bypass the command-in-flight drop.
import { describe, expect, test } from "bun:test"
import { isOutOfBandNotice } from "../src/lib/notices"

describe("isOutOfBandNotice", () => {
  test("update-check notices are out-of-band (never dropped mid-command)", () => {
    expect(
      isOutOfBandNotice({ message: "Lumen 0.4.0 available", origin: "update-check" }),
    ).toBe(true)
  })

  test("plain command notices are NOT out-of-band (dropped as result mirrors)", () => {
    expect(isOutOfBandNotice({ message: "Saved conversation: /tmp/x.json" })).toBe(false)
  })

  test("unknown or malformed origins stay in-band", () => {
    expect(isOutOfBandNotice({ message: "x", origin: "something-else" })).toBe(false)
    expect(isOutOfBandNotice({ message: "x", origin: 42 as unknown as string })).toBe(false)
    expect(isOutOfBandNotice({})).toBe(false)
  })
})
