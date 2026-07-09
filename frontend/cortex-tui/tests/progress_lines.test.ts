// Unit tests for the live-indicator line builders (src/lib/progress_lines.ts).
// The engine-update case is the regression pin for the "static 'Updating
// Lumen…' for minutes" finding: the latest installer output line (phase)
// must appear in the rendered line.
import { describe, expect, test } from "bun:test"
import type { DownloadProgressRecord } from "../src/context/store"
import {
  downloadIndicatorLine,
  engineUpdateIndicatorLine,
  formatBytes,
  loadIndicatorLine,
} from "../src/lib/progress_lines"

function record(overrides: Partial<DownloadProgressRecord>): DownloadProgressRecord {
  return {
    kind: "engine-update",
    repoID: "lumen",
    phase: "",
    bytesDownloaded: 0,
    stalled: false,
    ...overrides,
  }
}

describe("engineUpdateIndicatorLine", () => {
  test("renders content plus the latest installer phase line", () => {
    const line = engineUpdateIndicatorLine(
      record({ phase: "Installing Lumen v0.4.0..." }),
      "Updating Lumen…",
    )
    expect(line).toContain("Updating Lumen…")
    expect(line).toContain(" · Installing Lumen v0.4.0...")
  })

  test("phase updates change the line (narration is live, not static)", () => {
    const first = engineUpdateIndicatorLine(record({ phase: "stopping server" }), "Updating Lumen…")
    const second = engineUpdateIndicatorLine(
      record({ phase: "verifying installed version" }),
      "Updating Lumen…",
    )
    expect(first).toContain("stopping server")
    expect(second).toContain("verifying installed version")
    expect(second).not.toContain("stopping server")
  })

  test("falls back to a repo-derived label when content is empty", () => {
    const line = engineUpdateIndicatorLine(record({ repoID: "cortex", phase: "" }), "")
    expect(line).toContain("Updating cortex…")
    expect(line.endsWith("Updating cortex…")).toBe(true) // no dangling separator
  })

  test("long installer lines are ellipsized to keep the indicator on one line", () => {
    const phase = "x".repeat(200)
    const line = engineUpdateIndicatorLine(record({ phase }), "Updating Lumen…")
    expect(line).toContain("…")
    expect(line.length).toBeLessThan(120)
  })
})

describe("load and download lines", () => {
  test("load line names the selector", () => {
    const line = loadIndicatorLine(record({ kind: "model-load", repoID: "qwen3-5-9b:q4_0" }))
    expect(line).toContain("Loading qwen3-5-9b:q4_0…")
  })

  test("download line carries bytes and rate only when present", () => {
    const bare = downloadIndicatorLine(record({ kind: "download", repoID: "m:q" }))
    expect(bare).toContain("Downloading m:q…")
    expect(bare).not.toContain(" · ")

    const withBytes = downloadIndicatorLine(
      record({ kind: "download", repoID: "m:q", bytesDownloaded: 3 * 1024 * 1024, speedBps: 2 * 1024 * 1024 }),
    )
    expect(withBytes).toContain("3.0 MB")
    expect(withBytes).toContain("(2.0 MB/s)")
  })

  test("formatBytes handles edges", () => {
    expect(formatBytes(undefined)).toBe("0 B")
    expect(formatBytes(-5)).toBe("0 B")
    expect(formatBytes(512)).toBe("512 B")
    expect(formatBytes(1536)).toBe("1.5 KB")
  })
})
