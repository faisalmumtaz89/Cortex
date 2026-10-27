// How a git tool call reads in its transcript row: as the git command it
// runs, written the way a user would type it ("diff --staged", "log -p").

export function gitSummary(args: Record<string, unknown>): string {
  const words = [String(args.command ?? "")]
  if (args.staged) words.push("--staged")
  if (typeof args.revision === "string" && args.revision.trim()) words.push(args.revision.trim())
  if (args.stat) words.push("--stat")
  if (args.patch) words.push("-p")
  if (Array.isArray(args.paths) && args.paths.length > 0) words.push("--", ...args.paths.map(String))
  return words.join(" ")
}
