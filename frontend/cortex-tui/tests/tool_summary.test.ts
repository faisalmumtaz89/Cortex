import { describe, expect, test } from "bun:test"
import { shellSummary } from "../src/lib/shell_summary"

describe("bash row summary", () => {
  test("names the command, not the cd in front of it", () => {
    expect(shellSummary("cd /tmp/some/long/project && python3 -m tests.test_account"))
      .toBe("python3 -m tests.test_account")
    expect(shellSummary('cd "/a b" ; cd sub && make test')).toBe("make test")
  })

  test("keeps a redirect such as 2>&1 inside its stage", () => {
    expect(shellSummary("python3 -m tests 2>&1 | tail -5"))
      .toBe("python3 -m tests 2>&1 …")
  })

  test("folds line breaks into spaces", () => {
    expect(shellSummary("git commit -F - <<'EOF'\nSubject line\n\nBody\nEOF"))
      .toBe("git commit -F - <<'EOF' Subject line Body EOF")
  })

  test("a lone cd is still shown", () => {
    expect(shellSummary("cd src")).toBe("cd src")
  })
})
