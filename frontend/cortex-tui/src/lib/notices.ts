// Classification of system.notice payloads for the store's transcription
// gate. While a slash command is in flight, the store DROPS notices — they
// mirror the command's own result message and would double it. Out-of-band
// notices (the worker's async update check, which emits exactly ONCE per
// session) are not mirrors: dropping one loses it forever, so they must be
// transcribed regardless of command state.

/** Sources whose notices are emitted asynchronously, not as command echoes. */
const OUT_OF_BAND_ORIGINS = new Set(["update-check"])

export function isOutOfBandNotice(payload: Record<string, unknown>): boolean {
  const origin = typeof payload.origin === "string" ? payload.origin : ""
  return OUT_OF_BAND_ORIGINS.has(origin)
}
