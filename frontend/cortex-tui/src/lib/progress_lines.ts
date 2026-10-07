// Pure builder for the live system-progress indicator line (Cortex
// self-update). Extracted from system_message.tsx so the exact line
// composition is unit-testable (tests/progress_lines.test.ts).

import type { ProgressRecord } from "../context/store"
import { spinnerFrame } from "./spinner"

// Installer output lines can be long; keep the indicator on ONE line.
const MAX_PHASE_DETAIL_CHARS = 64

/** Self-update: spinner + the operation's content line ("Updating
 * Cortex…") + the LATEST installer output line as the phase detail — a
 * multi-minute install must visibly progress, not sit on a static label. */
export function engineUpdateIndicatorLine(
  progress: ProgressRecord,
  content: string,
): string {
  const label = content.trim().length > 0 ? content.trim() : `Updating ${progress.repoID}…`
  const parts = [`${spinnerFrame()} ${label}`]
  const phase = progress.phase.trim()
  if (phase.length > 0) {
    parts.push(
      phase.length > MAX_PHASE_DETAIL_CHARS
        ? `${phase.slice(0, MAX_PHASE_DETAIL_CHARS - 1)}…`
        : phase,
    )
  }
  return parts.join(" · ")
}
