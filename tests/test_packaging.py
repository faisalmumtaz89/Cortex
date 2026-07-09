"""Packaging/release invariants the update system depends on.

Source-invariant pins (same idiom as the TUI window-size test): they guard
declarations whose silent removal would only surface at release time —
long after the suite went green.
"""

from __future__ import annotations

import configparser
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_platform_tag_declared_macos_arm64() -> None:
    """The wheel bundles a darwin-arm64 Mach-O sidecar (cortex-tui): it must
    never be tagged `any`, or pip would install a 65MB unusable binary on
    every other platform. Declared in setup.cfg so LOCAL builds carry the tag
    too; the release workflow enforces it as a hard gate on the artifact."""
    parser = configparser.ConfigParser()
    assert parser.read(REPO_ROOT / "setup.cfg"), "setup.cfg missing"
    plat_name = parser.get("bdist_wheel", "plat_name")
    assert plat_name.startswith("macosx"), plat_name
    assert plat_name.endswith("arm64"), plat_name


def test_release_workflow_is_single_channel_github_assets() -> None:
    """GitHub Releases is the ONLY distribution channel: no PyPI publish step
    may exist, the wheel and its .sha256 sibling are attached as release
    assets, and the release — discovery AND install source in one artifact —
    is created strictly AFTER every gate, so a discoverable version is
    installable by construction. Also pins the idempotency/concurrency
    guards."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    # Single channel: nothing in the release pipeline may touch PyPI.
    assert "pypi" not in workflow.lower(), "PyPI must not appear in the release workflow"
    # Release creation is LAST: every gate (tag==pyproject, sidecar-in-wheel,
    # platform tag, sha256 generation) precedes it.
    release_index = workflow.index("action-gh-release")
    for gate in (
        "Verify tag matches the package version",
        "cortex/ui_runtime/bin/cortex-tui",  # sidecar-bundled hard gate
        "macosx*arm64*.whl",  # platform-tag hard gate
        "shasum -a 256",  # .sha256 sibling generation
    ):
        assert workflow.index(gate) < release_index, f"{gate!r} must precede release creation"
    # The install source: wheel + .sha256 sibling attached as release assets.
    assert "dist/*.whl" in workflow
    assert "dist/*.whl.sha256" in workflow
    # Idempotent re-runs: same-named assets are REPLACED on the existing
    # release (explicit — never rely on the floating @v2 default).
    assert "overwrite_files: true" in workflow
    assert "concurrency:" in workflow and "cancel-in-progress: false" in workflow


def test_release_workflow_pins_the_frontend_toolchain() -> None:
    """The shipped sidecar's toolchain must be reproducible: the workflow's
    bun pin must match the version the frontend declares."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    package = json.loads(
        (REPO_ROOT / "frontend" / "cortex-tui" / "package.json").read_text(encoding="utf-8")
    )
    declared = str(package["devDependencies"]["bun"]).lstrip("^~")
    assert f'bun-version: "{declared}"' in workflow, (
        f"release.yml must pin bun-version to {declared} (frontend package.json)"
    )


def test_wheel_asset_name_reconstruction_matches_packaging_config() -> None:
    """install.sh and /update cortex RECONSTRUCT the release wheel's filename
    from the version (GitHub asset URLs need the exact PEP 427 name, and pip
    refuses renamed wheels). The hardcoded platform tag and normalized project
    name in both installers must match setup.cfg's plat_name and pyproject's
    project name — drift would 404 every release install."""
    parser = configparser.ConfigParser()
    assert parser.read(REPO_ROOT / "setup.cfg"), "setup.cfg missing"
    plat_name = parser.get("bdist_wheel", "plat_name")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    normalized_name = str(pyproject["project"]["name"]).replace("-", "_")

    from cortex.app.update_service import CORTEX_WHEEL_ASSET_TEMPLATE

    assert CORTEX_WHEEL_ASSET_TEMPLATE == (
        f"{normalized_name}-{{version}}-py3-none-{plat_name}.whl"
    )

    install_sh = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    assert f'WHEEL_PLAT_TAG="{plat_name}"' in install_sh
    assert (
        f'wheel_name="{normalized_name}-${{version}}-py3-none-${{WHEEL_PLAT_TAG}}.whl"'
        in install_sh
    )


def test_install_sh_verifies_wheel_checksum_before_pip() -> None:
    """install.sh's release mode must die on a checksum mismatch or a missing
    .sha256 asset BEFORE pip ever sees the wheel, and must refuse non-GitHub
    asset origins without the explicit insecure override."""
    install_sh = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "shasum -a 256" in install_sh
    assert "installing nothing" in install_sh  # mismatch refusal
    assert "refusing to install an unverified wheel" in install_sh  # missing .sha256
    assert "https://github.com/*" in install_sh  # trusted-origin enforcement
    assert "CORTEX_ALLOW_INSECURE_BASE" in install_sh
    # The verification (die on failure) must precede the pip install.
    assert install_sh.index("Checksum mismatch") < install_sh.index('"${PIP}" install "${WHEEL_PATH}"')
