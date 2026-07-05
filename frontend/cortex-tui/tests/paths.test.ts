// Unit tests for the display-only path normalization (src/lib/paths.ts).
// These feed the permission modal and tool rows, so every branch matters:
// a wrong mapping either leaks an absolute path or, worse, RELABELS a path
// outside the repo as a harmless-looking relative one.
// Run via `bun test` (wired into the pytest gate by tests/test_frontend_unit.py).
import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import { displayPath } from "../src/lib/paths"

const ROOT = "/repo/project"
let savedProjectDir: string | undefined

beforeEach(() => {
  savedProjectDir = process.env.CORTEX_PROJECT_DIR
  process.env.CORTEX_PROJECT_DIR = ROOT
})

afterEach(() => {
  if (savedProjectDir === undefined) {
    delete process.env.CORTEX_PROJECT_DIR
  } else {
    process.env.CORTEX_PROJECT_DIR = savedProjectDir
  }
})

describe("displayPath", () => {
  test("absolute path inside the repo becomes repo-relative", () => {
    expect(displayPath(`${ROOT}/src/a.ts`)).toBe("src/a.ts")
    expect(displayPath(`${ROOT}/deeply/nested/dir/file.py`)).toBe("deeply/nested/dir/file.py")
  })

  test("the repo root itself displays as '.'", () => {
    expect(displayPath(ROOT)).toBe(".")
    expect(displayPath(`${ROOT}/`)).toBe(".")
  })

  test("sibling-prefix directories are NOT inside the repo", () => {
    // "/repo/project2" shares the string prefix with "/repo/project" but is a
    // different directory — it must pass through unchanged, never relabeled.
    expect(displayPath("/repo/project2/file.ts")).toBe("/repo/project2/file.ts")
    expect(displayPath("/repo/project-backup/a.py")).toBe("/repo/project-backup/a.py")
  })

  test("absolute paths outside the repo pass through unchanged", () => {
    expect(displayPath("/etc/hosts")).toBe("/etc/hosts")
    expect(displayPath("/")).toBe("/")
  })

  test("relative paths, globs, and empty input pass through unchanged", () => {
    expect(displayPath("src/a.ts")).toBe("src/a.ts")
    expect(displayPath("*")).toBe("*")
    expect(displayPath("**/*.py")).toBe("**/*.py")
    expect(displayPath("")).toBe("")
  })

  test("a trailing slash on the configured root is tolerated", () => {
    process.env.CORTEX_PROJECT_DIR = `${ROOT}/`
    expect(displayPath(`${ROOT}/src/a.ts`)).toBe("src/a.ts")
    expect(displayPath(ROOT)).toBe(".")
  })

  test("a filesystem-root project dir disables mapping instead of relabeling", () => {
    // Root "/" collapses to "" after trailing-slash strip; every absolute
    // path must then pass through unchanged (never sliced into a bogus
    // "relative" form that would mislabel the permission modal).
    process.env.CORTEX_PROJECT_DIR = "/"
    expect(displayPath("/etc/hosts")).toBe("/etc/hosts")
    expect(displayPath("/")).toBe("/")
  })

  test("surrounding whitespace is trimmed before matching", () => {
    expect(displayPath(`  ${ROOT}/src/a.ts  `)).toBe("src/a.ts")
  })
})
