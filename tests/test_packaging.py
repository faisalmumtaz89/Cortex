"""Packaging/release invariants the installers and the update system depend on.

Source-invariant pins (same idiom as the TUI window-size test): they guard
declarations whose silent drift would only surface at release time — long
after the suite went green.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from cortex.ui_runtime.platforms import TARGETS, current_target, wheel_asset_name

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CI = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
INSTALL_SH = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
BUILD_TS = (REPO_ROOT / "frontend" / "cortex-tui" / "scripts" / "build.ts").read_text(encoding="utf-8")


def test_platform_table_resolves_each_machine() -> None:
    assert current_target("Darwin", "arm64").wheel_tag == "macosx_13_0_arm64"
    assert current_target("Darwin", "x86_64").wheel_tag == "macosx_13_0_x86_64"
    assert current_target("Linux", "x86_64").wheel_tag == "manylinux_2_28_x86_64"
    assert current_target("Linux", "amd64").wheel_tag == "manylinux_2_28_x86_64"
    assert current_target("Linux", "aarch64").wheel_tag == "manylinux_2_28_aarch64"
    assert current_target("Windows", "AMD64") is None
    assert current_target("Linux", "riscv64") is None


def test_every_place_that_lists_the_platforms_agrees() -> None:
    """The sidecar build, the installer, and the release and CI matrices must
    list the same platforms as cortex/ui_runtime/platforms.py — a platform
    missing from one would ship a wheel nobody can install, or 404 installs."""
    tags = {t.wheel_tag for t in TARGETS}
    assert set(re.findall(r"wheel_tag: (\S+)", RELEASE)) == tags
    assert set(re.findall(r'WHEEL_PLAT_TAG="([a-z0-9_]+)"', INSTALL_SH)) == tags
    host_targets = dict(re.findall(r'"(\w+-\w+)": "(bun-[\w-]+)"', BUILD_TS))
    assert set(host_targets.values()) == {t.bun_target for t in TARGETS}
    release_runners = re.findall(r"runner: (\S+)", RELEASE)
    ci_runners = re.search(r"runner: \[([^\]]+)\]", CI).group(1).replace(" ", "").split(",")
    assert sorted(release_runners) == sorted(ci_runners) and len(ci_runners) == len(TARGETS)


def test_wheel_asset_name_matches_the_built_wheels() -> None:
    """install.sh and /update cortex RECONSTRUCT the wheel's filename from the
    version (pip refuses renamed wheels): project name normalized, py3-none,
    then the platform tag setup.py stamps on the wheel."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    normalized = str(pyproject["project"]["name"]).replace("-", "_")
    target = TARGETS[0]
    assert wheel_asset_name("1.2.3", target) == f"{normalized}-1.2.3-py3-none-{target.wheel_tag}.whl"
    assert (
        f'wheel_name="{normalized}-${{version}}-py3-none-${{WHEEL_PLAT_TAG}}.whl"' in INSTALL_SH
    )
    setup_py = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "CORTEX_WHEEL_PLATFORM" in setup_py and "current_target" in setup_py
    assert not (REPO_ROOT / "setup.cfg").exists(), "a fixed plat_name would mis-tag other platforms"


def test_release_workflow_is_single_channel_and_publishes_last() -> None:
    """GitHub Releases is the ONLY distribution channel; every wheel and its
    .sha256 are release assets; the release is created strictly AFTER every
    gate on every platform, so a discoverable version is installable."""
    assert "pypi" not in RELEASE.lower(), "PyPI must not appear in the release workflow"
    release_index = RELEASE.index("action-gh-release")
    for gate in (
        "Verify tag matches the package version",
        "cortex/ui_runtime/bin/cortex-tui",  # sidecar-bundled hard gate
        "-py3-none-${{ matrix.wheel_tag }}.whl",  # platform-tag hard gate
        "hashlib.sha256",  # .sha256 sibling generation
        "scripts/smoke_install.sh dist",  # installed and run on its own platform
        "test \"$(ls dist/*.whl | wc -l)\" -eq 4",  # every platform present
    ):
        assert RELEASE.index(gate) < release_index, f"{gate!r} must precede release creation"
    assert "needs: [build, windows-installer]" in RELEASE
    # A run by hand publishes only when asked to, and only from main; the tag
    # it creates is the package version's, on the commit that was tested.
    assert (
        "if: startsWith(github.ref, 'refs/tags/v') || "
        "(inputs.publish && github.ref == 'refs/heads/main')"
    ) in RELEASE
    assert "default: false" in RELEASE
    assert "tag_name: ${{ steps.tag.outputs.tag }}" in RELEASE
    assert "target_commitish: ${{ github.sha }}" in RELEASE
    assert RELEASE.index("Resolve the release tag") < release_index
    assert "dist/*.whl" in RELEASE and "dist/*.whl.sha256" in RELEASE
    assert "overwrite_files: true" in RELEASE
    assert "concurrency:" in RELEASE and "cancel-in-progress: false" in RELEASE


def test_release_workflow_pins_the_frontend_toolchain() -> None:
    """The shipped sidecar's toolchain must be reproducible: the workflows'
    bun pin must match the version the frontend declares."""
    package = json.loads(
        (REPO_ROOT / "frontend" / "cortex-tui" / "package.json").read_text(encoding="utf-8")
    )
    declared = str(package["devDependencies"]["bun"]).lstrip("^~")
    for workflow in (RELEASE, CI):
        assert f'bun-version: "{declared}"' in workflow


def test_install_sh_writes_only_under_home() -> None:
    """The launcher goes in ~/.local/bin, never a system folder: a link there
    (root, or Homebrew's user-writable bin) points every user at one person's
    install and survives the documented uninstall."""
    assert "/usr/local/bin" not in INSTALL_SH and "/opt/homebrew/bin" not in INSTALL_SH


def test_install_sh_verifies_wheel_checksum_before_pip() -> None:
    """install.sh's release mode must die on a checksum mismatch or a missing
    .sha256 asset BEFORE pip ever sees the wheel, and must refuse non-GitHub
    asset origins without the explicit insecure override."""
    assert 'got="$(sha256_of "${wheel_path}")"' in INSTALL_SH
    assert "installing nothing" in INSTALL_SH  # mismatch refusal
    assert "refusing to install an unverified wheel" in INSTALL_SH  # missing .sha256
    assert "https://github.com/*" in INSTALL_SH  # trusted-origin enforcement
    assert "CORTEX_ALLOW_INSECURE_BASE" in INSTALL_SH
    assert INSTALL_SH.index("Checksum mismatch") < INSTALL_SH.index('"${WHEEL_PATH}"')
    # The private Python's uv is verified the same way before it runs.
    assert INSTALL_SH.index('[[ "${want}" == "${got}" ]]') < INSTALL_SH.index('"${uv_bin}" python install')


def test_the_documented_uninstall_removes_what_the_installer_added_to_profiles(
    tmp_path: Path,
) -> None:
    """The profile commands in docs/installation.md undo install.sh's PATH
    lines: a user's own lines stay, and a profile the installer created goes."""
    import subprocess

    # install.sh appends a blank line, its marker and the PATH line.
    assert "printf '\\n# Added by the Cortex installer\\n%s\\n'" in INSTALL_SH
    added = '\n# Added by the Cortex installer\ncase ":$PATH:" in *) ;; esac\n'
    (tmp_path / ".bashrc").write_text("alias ll=ls\n" + added + "export KEEP=1\n")
    (tmp_path / ".profile").write_text(added)

    docs = (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    block = re.search(r"```bash\n(for f in ~/\.zshrc[^`]*)```", docs)
    assert block, "uninstall profile commands not found in docs/installation.md"
    subprocess.run(["bash", "-c", block.group(1)], env={"HOME": str(tmp_path),
                   "PATH": "/usr/bin:/bin"}, check=True)

    assert (tmp_path / ".bashrc").read_text() == "alias ll=ls\nexport KEEP=1\n"
    assert not (tmp_path / ".profile").exists()


def _docs_block(pattern: str) -> str:
    docs = (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    block = re.search(r"```bash\n(" + pattern + r"[^`]*)```", docs)
    assert block, f"{pattern!r} not found in docs/installation.md"
    return block.group(1)


def test_the_documented_program_removal_succeeds_and_keeps_the_users_files(
    tmp_path: Path,
) -> None:
    """The uninstall block exits 0 (a script that stops on errors goes on),
    also when ~/.local/bin holds the user's own tools, which stay."""
    import subprocess

    (tmp_path / ".cortex" / "install" / "venv").mkdir(parents=True)
    (tmp_path / ".cortex" / "config.yaml").write_text("tools_max_iterations: 80\n")
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cortex").symlink_to(tmp_path / ".cortex" / "install" / "venv" / "cortex")
    (bin_dir / "mytool").write_text("#!/bin/sh\n")

    subprocess.run(["bash", "-ec", _docs_block("# Remove the isolated runtime")],
                   env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=True)
    assert not (tmp_path / ".cortex" / "install").exists()
    assert not (bin_dir / "cortex").is_symlink()
    assert (bin_dir / "mytool").exists()
    assert (tmp_path / ".cortex" / "config.yaml").exists()


def test_the_documented_data_removal_keeps_config_and_the_program(tmp_path: Path) -> None:
    import subprocess

    home = tmp_path / ".cortex"
    for name in ("conversations", "install/venv"):
        (home / name).mkdir(parents=True)
    for name in ("config.yaml", "state.yaml", "credentials.json", "cortex.log",
                 "tool_permissions.yaml", ".tool_permissions.yaml.lock", "update-check.json"):
        (home / name).write_text("x\n")

    subprocess.run(["bash", "-ec", _docs_block("find ~/.cortex")],
                   env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=True)
    assert sorted(path.name for path in home.iterdir()) == ["config.yaml", "install"]


def test_the_installer_variables_are_documented_and_set_where_the_installer_sees_them() -> None:
    """Every CORTEX_* variable a user can give an installer is in the
    Installer Variables table, and a documented `curl ... | bash` sets one
    on bash: set before curl, it would reach only curl."""
    docs = (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    install_ps1 = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")
    test_seams = {"CORTEX_RELEASE_BASE", "CORTEX_ALLOW_INSECURE_BASE"}
    read = set(re.findall(r"\$\{(CORTEX_[A-Z_]+):?-", INSTALL_SH)) - test_seams
    read |= set(re.findall(r"\$env:(CORTEX_[A-Z_]+)", install_ps1))
    table = docs.split("### Installer Variables", 1)[1].split("\n## ", 1)[0]
    assert set(re.findall(r"^\| `(?:\$env:)?(CORTEX_[A-Z_]+)", table, re.M)) == read

    for doc in sorted((REPO_ROOT / "docs").glob("*.md")) + [REPO_ROOT / "README.md"]:
        for line in doc.read_text(encoding="utf-8").splitlines():
            if "install.sh |" in line:
                assert "CORTEX_" not in line.split("|", 1)[0], f"{doc.name}: {line}"


def test_an_installer_choice_that_cannot_be_met_stops_before_anything_is_installed(
    tmp_path: Path,
) -> None:
    """A Python named with CORTEX_PYTHON is used or the install stops: it is
    never swapped for another one. So does a choice that is not 1 or 0, or
    two that contradict each other. HOME is left as it was."""
    import subprocess
    import sys

    cases = {
        "CORTEX_PYTHON=/no/such/python": "CORTEX_PYTHON=/no/such/python is not a command",
        "CORTEX_PYTHON=/bin/true": "CORTEX_PYTHON=/bin/true is not a Python that runs",
        "CORTEX_MANAGED_PYTHON=yes": "CORTEX_MANAGED_PYTHON must be 1 or 0 (got: yes)",
        "CORTEX_NO_MODIFY_PATH=true": "CORTEX_NO_MODIFY_PATH must be 1 or 0 (got: true)",
        f"CORTEX_MANAGED_PYTHON=1 CORTEX_PYTHON={sys.executable}": "set one of them",
    }
    for assignments, message in cases.items():
        home = tmp_path / "home"
        home.mkdir()
        env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
        env.update(pair.split("=", 1) for pair in assignments.split())
        run = subprocess.run(["bash", str(REPO_ROOT / "install.sh"), "2.0.0"], env=env,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 1, (assignments, run.stdout, run.stderr)
        assert message in run.stderr, (assignments, run.stderr)
        assert list(home.iterdir()) == [], assignments
        home.rmdir()
