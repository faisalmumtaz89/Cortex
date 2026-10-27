/** Where the first stage of a shell command ends (quote-aware), or -1. The
 * `&` of a redirect such as `2>&1` or `&>` is not a separator. */
function firstStageEnd(command: string): number {
  let quote: string | undefined
  for (let i = 0; i < command.length; i++) {
    const ch = command[i]
    if (quote) {
      if (ch === quote) {
        quote = undefined
      } else if (ch === "\\" && quote === '"') {
        i += 1 // escapes only apply inside double quotes
      }
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      continue
    }
    if (ch === "\\") {
      i += 1
      continue
    }
    if (ch === "&" && (command[i - 1] === ">" || command[i - 1] === "<" || command[i + 1] === ">")) {
      continue
    }
    if (ch === "|" || ch === ";" || ch === "&") {
      return i
    }
  }
  return -1
}

const LEADING_CD = /^cd\s+("[^"]*"|'[^']*'|[^\s;&|]+)\s*(&&|;)\s*/

/** What a bash row names: the first stage of the command, after any leading
 * `cd <dir> &&` steps, with line breaks folded into spaces. Chains of
 * &&/|/; are execution detail; the first real stage identifies the action. */
export function shellSummary(command: string): string {
  let rest = command.replace(/\s+/g, " ").trim()
  for (let match = LEADING_CD.exec(rest); match && match[0].length < rest.length; match = LEADING_CD.exec(rest)) {
    rest = rest.slice(match[0].length)
  }
  const end = firstStageEnd(rest)
  if (end < 0) return rest
  const stage = rest.slice(0, end).trim()
  return stage ? `${stage} …` : rest
}
