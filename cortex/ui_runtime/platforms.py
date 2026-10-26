"""The platforms Cortex ships for: one wheel per platform, each bundling the
TUI sidecar compiled for it.

Standalone on purpose (no imports from the package): setup.py loads this file
to tag wheels, and install.sh mirrors the table (tests/test_packaging.py keeps
the two in step).
"""

from __future__ import annotations

import platform
from typing import NamedTuple, Optional


class Target(NamedTuple):
    system: str  # platform.system()
    machine: str  # normalized platform.machine()
    bun_target: str  # `bun build --compile` target for the sidecar
    wheel_tag: str  # wheel platform tag


# Intel builds use bun's "baseline" runtime, which also runs on CPUs without
# AVX2. manylinux_2_28: glibc 2.28 or newer (Debian 10, Ubuntu 20.04, RHEL 8).
TARGETS = (
    Target("Darwin", "arm64", "bun-darwin-arm64", "macosx_13_0_arm64"),
    Target("Darwin", "x86_64", "bun-darwin-x64-baseline", "macosx_13_0_x86_64"),
    Target("Linux", "x86_64", "bun-linux-x64-baseline", "manylinux_2_28_x86_64"),
    Target("Linux", "arm64", "bun-linux-arm64", "manylinux_2_28_aarch64"),
)

_MACHINE_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}


def current_target(system: Optional[str] = None, machine: Optional[str] = None) -> Optional[Target]:
    """The target for this machine (or the given one); None when Cortex has no build for it."""
    system = system or platform.system()
    raw = (machine or platform.machine()).lower()
    machine = _MACHINE_ALIASES.get(raw, raw)
    return next((t for t in TARGETS if t.system == system and t.machine == machine), None)


def wheel_asset_name(version: str, target: Target) -> str:
    """The release asset name of the wheel for `target`."""
    return f"cortex_llm-{version}-py3-none-{target.wheel_tag}.whl"
