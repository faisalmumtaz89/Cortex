"""Conservative stream-delta normalization helpers."""

from __future__ import annotations

MIN_RETRANSMIT_CHARS = 24
MIN_OVERLAP_STITCH = 24
MAX_OVERLAP_SCAN = 2048


def _longest_suffix_prefix_overlap(assembled: str, chunk: str) -> int:
    """Return longest overlap where assembled suffix equals chunk prefix."""
    max_scan = min(len(assembled), len(chunk), MAX_OVERLAP_SCAN)
    for size in range(max_scan, MIN_OVERLAP_STITCH - 1, -1):
        if assembled[-size:] == chunk[:size]:
            return size
    return 0


def merge_stream_text(raw_chunk: str, assembled_text: str) -> tuple[str, str]:
    """Merge stream chunks without risking text loss.

    Strategy:
    - Treat incoming chunks as true deltas by default.
    - Deduplicate only high-confidence retransmits.
    - Support cumulative snapshots and large suffix/prefix overlap stitching.
    """
    chunk = str(raw_chunk or "")
    if not chunk:
        return "", assembled_text

    if not assembled_text:
        return chunk, chunk

    # Exact duplicate snapshot.
    if chunk == assembled_text:
        return "", assembled_text

    stripped_chunk = chunk.lstrip()

    # Whitespace-prefixed duplicate snapshot.
    if stripped_chunk == assembled_text:
        return "", assembled_text

    # Cumulative snapshot: provider may re-send full text-so-far.
    if chunk.startswith(assembled_text):
        delta = chunk[len(assembled_text) :]
        return delta, chunk

    # Whitespace-prefixed cumulative snapshot.
    if stripped_chunk.startswith(assembled_text):
        delta = stripped_chunk[len(assembled_text) :]
        if not delta:
            return "", assembled_text
        return delta, assembled_text + delta

    # High-confidence retransmit of previously emitted prefix/suffix.
    if len(chunk) >= MIN_RETRANSMIT_CHARS and (
        assembled_text.startswith(chunk) or assembled_text.endswith(chunk)
    ):
        return "", assembled_text

    if len(stripped_chunk) >= MIN_RETRANSMIT_CHARS and (
        assembled_text.startswith(stripped_chunk) or assembled_text.endswith(stripped_chunk)
    ):
        return "", assembled_text

    # Large overlap stitch for snapshot-like chunks.
    overlap = _longest_suffix_prefix_overlap(assembled_text, chunk)
    if overlap > 0:
        delta = chunk[overlap:]
        if not delta:
            return "", assembled_text
        return delta, assembled_text + delta

    # Default: preserve chunk as-is to avoid accidental truncation.
    return chunk, assembled_text + chunk
