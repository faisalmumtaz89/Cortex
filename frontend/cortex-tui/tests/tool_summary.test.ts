import { describe, expect, test } from "bun:test"
import { alwaysCovers } from "../src/lib/permission_prompt"
import { toolCallSummary, toolVerb } from "../src/components/tool_block"
import { shellSummary } from "../src/lib/shell_summary"
import { gitSummary } from "../src/lib/git_summary"

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

describe("git row summary", () => {
  test("reads as the git command that runs", () => {
    expect(gitSummary({ command: "status" })).toBe("status")
    expect(gitSummary({ command: "diff", staged: true })).toBe("diff --staged")
    expect(gitSummary({ command: "diff", revision: "main...HEAD", stat: true })).toBe("diff main...HEAD --stat")
    expect(gitSummary({ command: "log", patch: true, paths: ["src/app.py"] })).toBe("log -p -- src/app.py")
  })
})

describe("what Allow always would cover", () => {
  const prompt = (permission: string, pattern: string) => ({ permission, patterns: [pattern], metadata: {} })

  test("is the exact command, or the file", () => {
    expect(alwaysCovers(prompt("bash", "ls | wc"), "Allow always", { columns: 80, rows: 24 }))
      .toBe(" · this exact command")
    expect(alwaysCovers(prompt("edit", "a.py"), "Allow always", { columns: 80, rows: 24 })).toBe(" · this file")
  })

  test("is not said when no rule would be saved, or for another kind of call", () => {
    expect(alwaysCovers(prompt("bash", "*"), "Allow always", { columns: 80, rows: 24 })).toBeUndefined()
    expect(alwaysCovers(prompt("doom_loop", "x"), "Allow always", { columns: 80, rows: 24 })).toBeUndefined()
  })

  test("is not said where it would make the choice wrap", () => {
    expect(alwaysCovers(prompt("bash", "ls"), "Allow always", { columns: 42, rows: 24 })).toBe(" · this exact command")
    expect(alwaysCovers(prompt("bash", "ls"), "Allow always", { columns: 41, rows: 24 })).toBeUndefined()
    expect(alwaysCovers(prompt("edit", "a.py"), "Allow always", { columns: 33, rows: 24 })).toBe(" · this file")
    expect(alwaysCovers(prompt("edit", "a.py"), "Allow always", { columns: 32, rows: 24 })).toBeUndefined()
  })
})

describe("a tool row's headline", () => {
  const hostile = "a\u001b[2Jb\u202ec\u200bd\u0007e\rf"
  const calls: [string, Record<string, unknown>][] = [
    ["bash", { command: `echo ${hostile}` }],
    ["read_file", { path: hostile }],
    ["list_dir", { path: hostile }],
    ["search", { query: hostile }],
    ["edit_file", { path: hostile }],
    ["write_file", { path: hostile }],
    ["git", { command: "diff", paths: [hostile] }],
    ["skill", { name: hostile }],
    ["other_tool", { anything: hostile }],
  ]

  test("shows nothing invisible, whichever tool it is", () => {
    for (const [tool, input] of calls) {
      const shown = toolCallSummary(tool, input)
      expect(shown).not.toMatch(/[\p{Cc}\p{Cf}\p{Cs}]/u)
      expect(shown).toContain("⟨U+001B⟩")
    }
  })

  test("is one line: a line break or a tab reads as a space", () => {
    expect(toolCallSummary("read_file", { path: "a\nb\tc" })).toBe("a b c")
    expect(toolCallSummary("search", { query: "a\nb\tc" })).toBe("a b c")
  })

  test("marks a marker character once, and plain text not at all", () => {
    expect(toolCallSummary("search", { query: "a⟨b" })).toBe("a⟨U+27E8⟩b")
    expect(toolCallSummary("bash", { command: "ls -la src" })).toBe("ls -la src")
  })

  test("marks a blank that is not a space: only spaces, tabs and line breaks fold", () => {
    expect(toolCallSummary("bash", { command: "rm -rf a\u00a0b" })).toBe("rm -rf a⟨U+00A0⟩b")
    expect(toolCallSummary("read_file", { path: "calc.py\ufeff" })).toBe("calc.py⟨U+FEFF⟩")
    expect(toolCallSummary("search", { query: "a\u2028b\u3000" })).toBe("a⟨U+2028⟩b⟨U+3000⟩")
    // Also where a command is cut to its first stage, and after a leading cd.
    expect(toolCallSummary("bash", { command: "\ufeffls\u00a0| cat" })).toBe("⟨U+FEFF⟩ls⟨U+00A0⟩ …")
    expect(toolCallSummary("bash", { command: "cd foo &&\u00a0ls" })).toBe("⟨U+00A0⟩ls")
    expect(toolCallSummary("bash", { command: "cd\u00a0foo && ls" })).toBe("cd⟨U+00A0⟩foo …")
  })

  test("shows the invisible characters that join an emoji, as the prompt does", () => {
    expect(toolCallSummary("search", { query: "⚠\ufe0f 👨\u200d👩" })).toBe("⚠⟨U+FE0F⟩ 👨⟨U+200D⟩👩")
  })

  test("names a tool that is none of Cortex's by its marked name", () => {
    expect(toolVerb("bash")).toBe("Bash")
    expect(toolVerb("zz\u001b]0;T")).toBe("zz⟨U+001B⟩]0;T")
    expect(toolVerb("ba\u200bsh")).toBe("ba⟨U+200B⟩sh")
    expect(toolVerb("constructor")).toBe("constructor")
    expect([...toolVerb("Q".repeat(500))].length).toBe(40)
  })

  test("is cut between characters, and never inside a marker", () => {
    expect(toolCallSummary("search", { query: `${"x".repeat(70)}😀😀😀` })).toBe(`${"x".repeat(70)}😀…`)
    // The marker would end past the cut: it is left out whole.
    expect(toolCallSummary("search", { query: `${"x".repeat(68)}\u202etail` })).toBe(`${"x".repeat(68)}…`)
    // A path keeps its end, which starts at a character or a whole marker.
    const path = toolCallSummary("read_file", { path: `${"d/".repeat(40)}\u202e${"y".repeat(10)}` })
    expect(path).toMatch(/^…[d/]*⟨U\+202E⟩y{10}$/)
    expect(toolCallSummary("read_file", { path: `${"d/".repeat(40)}\u202e${"y".repeat(60)}` })).toMatch(/^…y+$/)
  })
})
