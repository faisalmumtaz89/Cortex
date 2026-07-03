import { RGBA } from "@opentui/core"
import { createEffect, createSignal, onCleanup, untrack, type Accessor } from "solid-js"

// Small, reusable presentation-layer animation primitives for the TUI.
//
// Terminals can't fade opacity, but the renderer repaints at 60fps, so smooth
// perception comes from two techniques:
//   - createFadeIn:   an eased 0→1 progress signal driven on mount, used to
//                     interpolate colors so new content eases in instead of
//                     popping.
//   - createMinDwell: a display-state mirror that guarantees each LIVE value
//                     stays visible for a minimum time. Real state changes can
//                     be faster than one frame (a read_file completes in ~5ms,
//                     so "running" was never visible); the underlying state is
//                     untouched — only its presentation is smoothed, the same
//                     minimum-spinner-time rule used across UI toolkits.
//   - mixRgba:        linear color interpolation between two RGBA values.
//
// All primitives are owner-scoped (timers cleared via onCleanup), so they can
// be used inside any component without leak risk.

export function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3)
}

/** Interpolate between two colors; `t` is clamped to [0, 1]. */
export function mixRgba(from: RGBA, to: RGBA, t: number): RGBA {
  const k = Math.max(0, Math.min(1, t))
  return RGBA.fromValues(
    from.r + (to.r - from.r) * k,
    from.g + (to.g - from.g) * k,
    from.b + (to.b - from.b) * k,
    from.a + (to.a - from.a) * k,
  )
}

const TICK_MS = 33 // ~30fps is ample for short color fades

/** Eased 0→1 progress starting when the owning component mounts. The interval
 * self-clears at completion and on owner disposal. */
export function createFadeIn(durationMs = 240, easing: (t: number) => number = easeOutCubic): Accessor<number> {
  const [progress, setProgress] = createSignal(durationMs <= 0 ? 1 : 0)
  if (durationMs > 0) {
    const startedAt = Date.now()
    const timer = setInterval(() => {
      const t = Math.min(1, (Date.now() - startedAt) / durationMs)
      setProgress(easing(t))
      if (t >= 1) {
        clearInterval(timer)
      }
    }, TICK_MS)
    onCleanup(() => clearInterval(timer))
  }
  return progress
}

/** Mirror `source`, but hold each displayed value for at least `minMs` before
 * showing the next one. By default the INITIAL value is adopted instantly
 * (content that mounts already-final — e.g. a restored transcript — must not
 * replay fake intermediate states); pass `options.initial` to start from an
 * earlier value instead (e.g. a row whose real state advanced within the same
 * tick as its mount, so the transition still reads as a progression). If the
 * source changes again while a transition is pending, the newest value
 * supersedes it and is shown at the same scheduled time. */
export function createMinDwell<T>(
  source: Accessor<T>,
  minMs: number,
  options?: { initial?: T },
): Accessor<T> {
  const [shown, setShown] = createSignal<T>(options?.initial ?? untrack(source))
  let shownSince = Date.now()
  let timer: ReturnType<typeof setTimeout> | undefined

  const apply = (value: T) => {
    setShown(() => value)
    shownSince = Date.now()
  }

  createEffect(() => {
    const next = source()
    if (next === untrack(shown)) {
      return
    }
    if (timer !== undefined) {
      clearTimeout(timer)
      timer = undefined
    }
    const remaining = minMs - (Date.now() - shownSince)
    if (remaining <= 0) {
      apply(next)
      return
    }
    timer = setTimeout(() => {
      timer = undefined
      apply(next)
    }, remaining)
  })
  onCleanup(() => {
    if (timer !== undefined) {
      clearTimeout(timer)
    }
  })
  return shown
}
