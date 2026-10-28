#!/usr/bin/env python3
"""Print the notes CHANGELOG.md has for a version: its `## <version>` section.

The release workflow publishes these as the release's notes, and stops when a
version has none, so every release says what changed. Usage:
release_notes.py VERSION [CHANGELOG]"""

from __future__ import annotations

import sys
from pathlib import Path


def notes(changelog: str, version: str) -> str:
    """The text under `## <version>` up to the next `## ` heading, stripped;
    empty when the section is missing or has nothing in it."""
    lines: list[str] = []
    inside = False
    for line in changelog.splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip() == version
            continue
        if inside:
            lines.append(line)
    return "\n".join(lines).strip()


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        sys.stderr.write("usage: release_notes.py VERSION [CHANGELOG]\n")
        return 2
    version = argv[1]
    path = Path(argv[2]) if len(argv) == 3 else Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    text = notes(path.read_text(encoding="utf-8"), version)
    if not text:
        sys.stderr.write(f"{path.name} has no notes for {version}: add a '## {version}' section.\n")
        return 1
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
