from __future__ import annotations

import os
from pathlib import Path

from cortex.ui_runtime import launcher


def _make_executable(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_candidate_command_skips_script_wrapper_when_bun_missing(
    tmp_path: Path, monkeypatch
) -> None:
    bundled = _make_executable(tmp_path / "cortex-tui", b"#!/usr/bin/env bash\necho test\n")
    entrypoint = tmp_path / "index.tsx"
    entrypoint.write_text("console.log('ok')", encoding="utf-8")

    monkeypatch.setattr(launcher, "_bundled_binary", lambda: bundled)
    monkeypatch.setattr(launcher, "_dev_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(launcher, "_find_bun", lambda: None)

    assert launcher._candidate_command() is None  # type: ignore[attr-defined]


def test_candidate_command_prefers_bundled_binary_without_bun(tmp_path: Path, monkeypatch) -> None:
    bundled = _make_executable(tmp_path / "cortex-tui", b"\xcf\xfa\xed\xfe")
    entrypoint = tmp_path / "index.tsx"
    entrypoint.write_text("console.log('ok')", encoding="utf-8")

    monkeypatch.setattr(launcher, "_bundled_binary", lambda: bundled)
    monkeypatch.setattr(launcher, "_dev_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(launcher, "_find_bun", lambda: None)

    assert launcher._candidate_command() == [str(bundled)]  # type: ignore[attr-defined]


def test_candidate_command_uses_bun_dev_entry_when_available(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    frontend_dir = repo_root / "frontend" / "cortex-tui"
    entrypoint = repo_root / "frontend" / "cortex-tui" / "src" / "index.tsx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("console.log('ok')", encoding="utf-8")

    bundled = repo_root / "cortex" / "ui_runtime" / "bin" / "cortex-tui"
    bundled.parent.mkdir(parents=True)
    # Non-executable placeholder means launcher should use bun run path.
    bundled.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(launcher, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(launcher, "_bundled_binary", lambda: bundled)
    monkeypatch.setattr(launcher, "_dev_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(launcher, "_frontend_project_dir", lambda: frontend_dir)
    monkeypatch.setattr(launcher, "_find_bun", lambda: "/usr/local/bin/bun")

    assert launcher._candidate_command() == [  # type: ignore[attr-defined]
        "/usr/local/bin/bun",
        "run",
        "--cwd",
        str(frontend_dir),
        "--preload",
        "@opentui/solid/preload",
        "--conditions=browser",
        "src/index.tsx",
    ]


def test_candidate_command_prefers_dev_entry_in_repo_checkout_when_bun_available(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    frontend_dir = repo_root / "frontend" / "cortex-tui"
    entrypoint = frontend_dir / "src" / "index.tsx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("console.log('ok')", encoding="utf-8")
    (repo_root / ".git").mkdir(parents=True)

    bundled = repo_root / "cortex" / "ui_runtime" / "bin" / "cortex-tui"
    bundled.parent.mkdir(parents=True)
    _make_executable(bundled, b"\xcf\xfa\xed\xfe")

    monkeypatch.setattr(launcher, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(launcher, "_bundled_binary", lambda: bundled)
    monkeypatch.setattr(launcher, "_dev_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(launcher, "_frontend_project_dir", lambda: frontend_dir)
    monkeypatch.setattr(launcher, "_find_bun", lambda: "/usr/local/bin/bun")

    assert launcher._candidate_command() == [  # type: ignore[attr-defined]
        "/usr/local/bin/bun",
        "run",
        "--cwd",
        str(frontend_dir),
        "--preload",
        "@opentui/solid/preload",
        "--conditions=browser",
        "src/index.tsx",
    ]


def test_candidate_command_force_bundled_overrides_repo_dev_preference(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    frontend_dir = repo_root / "frontend" / "cortex-tui"
    entrypoint = frontend_dir / "src" / "index.tsx"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("console.log('ok')", encoding="utf-8")
    (repo_root / ".git").mkdir(parents=True)

    bundled = repo_root / "cortex" / "ui_runtime" / "bin" / "cortex-tui"
    bundled.parent.mkdir(parents=True)
    _make_executable(bundled, b"\xcf\xfa\xed\xfe")

    monkeypatch.setattr(launcher, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(launcher, "_bundled_binary", lambda: bundled)
    monkeypatch.setattr(launcher, "_dev_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(launcher, "_frontend_project_dir", lambda: frontend_dir)
    monkeypatch.setattr(launcher, "_find_bun", lambda: "/usr/local/bin/bun")
    monkeypatch.setenv("CORTEX_TUI_FORCE_BUNDLED", "1")

    assert launcher._candidate_command() == [str(bundled)]  # type: ignore[attr-defined]


def test_find_bun_uses_local_frontend_runtime(tmp_path: Path, monkeypatch) -> None:
    frontend_dir = tmp_path / "frontend" / "cortex-tui"
    local_bun = frontend_dir / "node_modules" / "bun" / "bin" / "bun.exe"
    local_bun.parent.mkdir(parents=True)
    _make_executable(local_bun, b"#!/usr/bin/env bash\necho bun\n")

    monkeypatch.setattr(launcher.shutil, "which", lambda _name: None)
    monkeypatch.setattr(launcher, "_frontend_project_dir", lambda: frontend_dir)

    assert launcher._find_bun() == str(local_bun)  # type: ignore[attr-defined]


def test_build_worker_env_strips_otui_environment_by_default(monkeypatch) -> None:
    monkeypatch.setenv("OTUI_NO_NATIVE_RENDER", "1")
    monkeypatch.setenv("OTUI_DEBUG", "1")
    monkeypatch.setenv("OTUI_OVERRIDE_STDOUT", "0")
    monkeypatch.delenv("CORTEX_PRESERVE_OTUI_ENV", raising=False)

    env = launcher._build_worker_env()  # type: ignore[attr-defined]

    assert "OTUI_NO_NATIVE_RENDER" not in env
    assert "OTUI_DEBUG" not in env
    assert "OTUI_OVERRIDE_STDOUT" not in env
    assert env["CORTEX_WORKER_ARGS"] == "-P -m cortex --worker-stdio"


def test_build_worker_env_can_preserve_otui_environment_for_debugging(monkeypatch) -> None:
    monkeypatch.setenv("OTUI_NO_NATIVE_RENDER", "1")
    monkeypatch.setenv("CORTEX_PRESERVE_OTUI_ENV", "1")

    env = launcher._build_worker_env()  # type: ignore[attr-defined]

    assert env["OTUI_NO_NATIVE_RENDER"] == "1"


def test_sidecar_temp_folders_of_dead_launchers_are_removed(tmp_path, monkeypatch) -> None:
    """A launcher killed with its terminal cannot remove its sidecar's temp
    folder; the next launch removes it, and leaves a running launcher's own."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    dead = tmp_path / "cortex-tui-999999999-abc"
    live = tmp_path / f"cortex-tui-{os.getppid()}-def"
    for folder in (dead, live):
        folder.mkdir()
        (folder / ".lib-00000000.so").write_bytes(b"x")

    created = launcher._sidecar_tmpdir()

    assert not dead.exists()
    assert live.exists()
    assert created.parent == tmp_path and created.name.startswith(f"cortex-tui-{os.getpid()}-")


def _sidecar(tmp_path: Path, *, code: int) -> Path:
    """A stand-in sidecar that writes to stderr and exits with `code`."""
    return _make_executable(
        tmp_path / "sidecar.sh",
        f"#!/bin/sh\necho run >> {tmp_path / 'runs'}\necho 'sidecar says why' >&2\nexit {code}\n".encode(),
    )


def _launch(tmp_path, monkeypatch, sidecar: Path) -> int:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(launcher, "_candidate_command", lambda: [str(sidecar)])
    return launcher.launch_tui()


def test_a_sidecar_that_fails_shows_its_error_and_runs_once(tmp_path, monkeypatch, capsys) -> None:
    assert _launch(tmp_path, monkeypatch, _sidecar(tmp_path, code=1)) == 1
    assert (tmp_path / "runs").read_text().splitlines() == ["run"]
    assert "sidecar says why" in capsys.readouterr().err


def test_what_a_sidecar_writes_to_stderr_stays_off_screen_when_it_succeeds(
    tmp_path, monkeypatch, capsys
) -> None:
    """The screen belongs to the app: stderr is shown only for a failure."""
    assert _launch(tmp_path, monkeypatch, _sidecar(tmp_path, code=0)) == 0
    assert "sidecar says why" not in capsys.readouterr().err
