// Unit tests for the live-indicator line builder (src/lib/progress_lines.ts).
// The latest installer output line (phase) must appear in the rendered line,
// so a multi-minute update visibly progresses.
import { describe, expect, test } from "bun:test"
import type { ProgressRecord } from "../src/context/store"
import { engineUpdateIndicatorLine } from "../src/lib/progress_lines"

function record(overrides: Partial<ProgressRecord>): ProgressRecord {
  return {
    kind: "engine-update",
    repoID: "cortex",
    phase: "",
    ...overrides,
  }
}

describe("engineUpdateIndicatorLine", () => {
  test("renders content plus the latest installer phase line", () => {
    const line = engineUpdateIndicatorLine(
      record({ phase: "installing cortex_llm-9.9.9.whl" }),
      "Updating Cortex…",
    )
    expect(line).toContain("Updating Cortex…")
    expect(line).toContain(" · installing cortex_llm-9.9.9.whl")
  })

  test("phase updates change the line (narration is live, not static)", () => {
    const first = engineUpdateIndicatorLine(
      record({ phase: "downloading cortex_llm-9.9.9.whl" }),
      "Updating Cortex…",
    )
    const second = engineUpdateIndicatorLine(
      record({ phase: "verifying checksum" }),
      "Updating Cortex…",
    )
    expect(first).toContain("downloading cortex_llm-9.9.9.whl")
    expect(second).toContain("verifying checksum")
    expect(second).not.toContain("downloading")
  })

  test("falls back to a repo-derived label when content is empty", () => {
    const line = engineUpdateIndicatorLine(record({ phase: "" }), "")
    expect(line).toContain("Updating cortex…")
    expect(line.endsWith("Updating cortex…")).toBe(true) // no dangling separator
  })

  test("long installer lines are ellipsized to keep the indicator on one line", () => {
    const phase = "x".repeat(200)
    const line = engineUpdateIndicatorLine(record({ phase }), "Updating Cortex…")
    expect(line).toContain("…")
    expect(line.length).toBeLessThan(120)
  })
})
