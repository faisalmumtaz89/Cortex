// Display-only path normalization: models sometimes pass ABSOLUTE paths in
// tool arguments; rows and modals should read repo-relative when the path is
// inside the repo. Tool inputs themselves are never rewritten — this is a
// presentation concern only (the backend already sandbox-resolves inputs).

/** Repo root used for display normalization — same source as the header's
 * working-dir line (routes/session.tsx). */
function repoRoot(): string {
  return process.env.CORTEX_PROJECT_DIR || process.cwd()
}

/**
 * Repo-relative form of `path` when it is an absolute path under the repo
 * root; any other value (already-relative, outside the repo, globs like "*")
 * passes through unchanged.
 */
export function displayPath(path: string): string {
  const raw = path.trim()
  if (!raw.startsWith("/")) {
    return path
  }
  const root = repoRoot().replace(/\/+$/, "")
  if (!root) {
    return path
  }
  if (raw === root) {
    return "."
  }
  if (raw.startsWith(`${root}/`)) {
    return raw.slice(root.length + 1) || "."
  }
  return path
}
