// Drives the REAL store event path (createSessionStore → applyEvent →
// applyOne) for the system.notice transcription gate. This is the direct
// regression pin for the commandInFlight clause in store.tsx: reverting
// `&& !isOutOfBandNotice(event.payload)` must fail the first test — the
// worker emits the update notice exactly ONCE per session, so dropping it
// while an unrelated command is in flight loses it forever.
import { describe, expect, test } from "bun:test"
import { createRoot } from "solid-js"
import { createSessionStore, type SessionStore } from "../src/context/session_store"
import type { EventEnvelope } from "../src/protocol"

function withStore(run: (store: SessionStore) => void): void {
  createRoot((dispose) => {
    try {
      run(createSessionStore())
    } finally {
      dispose()
    }
  })
}

function noticeEvent(seq: number, payload: Record<string, unknown>): EventEnvelope {
  return {
    session_id: "s1",
    seq,
    ts_ms: 1_000_000 + seq,
    event_type: "system.notice",
    payload,
  }
}

function transcribedContents(store: SessionStore): string[] {
  return store.state.orderedMessageIDs.map((id) => store.state.messages[id]?.content ?? "")
}

const UPDATE_NOTICE = "Lumen 0.4.0 available — update with /update lumen"

describe("system.notice transcription gate (store event path)", () => {
  test("out-of-band update notice transcribes even while a command is in flight", () => {
    withStore((store) => {
      store.armCommandInFlight() // a slash command is mid-flight
      store.applyEvent(noticeEvent(1, { message: UPDATE_NOTICE, origin: "update-check" }))
      store.flushPendingEvents()

      expect(transcribedContents(store)).toContain(UPDATE_NOTICE)
      const id = store.state.orderedMessageIDs[0]!
      expect(store.state.messages[id]!.role).toBe("system")
      expect(store.state.messages[id]!.final).toBe(true)
    })
  })

  test("plain notices are suppressed while a command is in flight (result mirrors)", () => {
    withStore((store) => {
      store.armCommandInFlight()
      store.applyEvent(noticeEvent(1, { message: "Saved conversation: /tmp/x.json" }))
      store.flushPendingEvents()

      expect(transcribedContents(store)).toEqual([])

      // …and the suppression ends with the command: a later notice lands.
      store.disarmCommandInFlight()
      store.applyEvent(noticeEvent(2, { message: "Conversation cleared." }))
      store.flushPendingEvents()
      expect(transcribedContents(store)).toEqual(["Conversation cleared."])
    })
  })

  test("plain notices transcribe normally when no command is in flight", () => {
    withStore((store) => {
      store.applyEvent(noticeEvent(1, { message: "Session ready" }))
      store.flushPendingEvents()
      expect(transcribedContents(store)).toEqual(["Session ready"])
    })
  })
})
