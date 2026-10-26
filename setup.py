"""Tag the wheel for the platform its bundled TUI sidecar was built for.

The package is pure Python, but cortex/ui_runtime/bin/cortex-tui is a native
binary, so a py3-none-any wheel would install an unusable binary everywhere
else. The tag is this machine's (the sidecar is built natively), or the one in
CORTEX_WHEEL_PLATFORM, or `--plat-name`.
"""

import os
import runpy
from pathlib import Path

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

PLATFORMS = runpy.run_path(str(Path(__file__).parent / "cortex" / "ui_runtime" / "platforms.py"))


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        if self.plat_name is None:
            tag = os.environ.get("CORTEX_WHEEL_PLATFORM")
            if not tag:
                target = PLATFORMS["current_target"]()
                if target is None:
                    raise SystemExit(
                        "Cortex has no build for this platform; supported: "
                        + ", ".join(f"{t.system} {t.machine}" for t in PLATFORMS["TARGETS"])
                    )
                tag = target.wheel_tag
            self.plat_name = tag
        super().finalize_options()


setup(cmdclass={"bdist_wheel": PlatformWheel})
