import { readFileSync } from "node:fs"
import { isAbsolute, join } from "node:path"

// Read the current git branch without spawning a subprocess: parse .git/HEAD
// directly (works in the compiled sidecar, no git binary needed). Returns the
// branch name, a short SHA for a detached HEAD, or undefined outside a repo.
export function readGitBranch(dir: string): string | undefined {
  try {
    const dotGit = join(dir, ".git")
    let head: string
    try {
      head = readFileSync(join(dotGit, "HEAD"), "utf8")
    } catch {
      // .git can be a FILE pointing at the real gitdir (worktrees/submodules).
      const indirection = readFileSync(dotGit, "utf8")
      const match = /^gitdir:\s*(.+)\s*$/m.exec(indirection)
      if (!match) {
        return undefined
      }
      const gitDir = match[1].trim()
      head = readFileSync(join(isAbsolute(gitDir) ? gitDir : join(dir, gitDir), "HEAD"), "utf8")
    }
    const ref = /^ref:\s*refs\/heads\/(.+)\s*$/m.exec(head)
    if (ref) {
      return ref[1].trim()
    }
    const sha = head.trim()
    return sha.length >= 7 ? sha.slice(0, 7) : undefined
  } catch {
    return undefined
  }
}
