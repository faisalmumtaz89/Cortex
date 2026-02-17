"""Version check helpers for Cortex package installs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Optional

from packaging.version import InvalidVersion, Version

UPDATE_CACHE_TTL = timedelta(hours=24)


@dataclass(frozen=True)
class UpdateStatus:
    current_version: str
    latest_version: str


def _coerce_optional_str(value: object) -> Optional[str]:
    """Normalize arbitrary values into optional non-empty strings."""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def get_update_status(*, config) -> Optional[UpdateStatus]:
    """Return update status when a newer package version is available."""
    if not _should_check_updates(config):
        return None

    installed_version = _get_installed_version()
    if not installed_version:
        return None

    latest_version = _fetch_latest_version()
    if latest_version:
        _cache_latest_version(config, latest_version)
    else:
        latest_version = _get_latest_version_cached(config)
        if not latest_version:
            return None

    try:
        if Version(latest_version) <= Version(installed_version):
            return None
    except InvalidVersion:
        return None

    return UpdateStatus(current_version=installed_version, latest_version=latest_version)


def _should_check_updates(config) -> bool:
    """Determine if update checks should run."""
    if config.is_setting_explicit("auto_update_check"):
        return bool(config.system.auto_update_check)
    return True


def _get_installed_version() -> Optional[str]:
    """Read the currently installed package version."""
    for package_name in ("cortex-llm", "cortex_llm"):
        try:
            installed = package_version(package_name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue

        normalized = _coerce_optional_str(installed)
        if normalized:
            return normalized

    return _get_pipx_installed_version()


def _get_pipx_installed_version() -> Optional[str]:
    """Read the installed version from pipx, if available (legacy fallback)."""
    if not shutil.which("pipx"):
        return None

    try:
        result = subprocess.run(
            ["pipx", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _parse_pipx_list_text(result.stdout)

    venvs = payload.get("venvs", {})
    info = venvs.get("cortex-llm")
    if not info:
        return None
    metadata = info.get("metadata") or {}
    main_package = metadata.get("main_package") or {}
    return (
        _coerce_optional_str(main_package.get("package_version"))
        or _coerce_optional_str(main_package.get("version"))
        or _coerce_optional_str(info.get("package_version"))
        or _coerce_optional_str(info.get("version"))
    )


def _parse_pipx_list_text(output: str) -> Optional[str]:
    """Fallback parser for pipx list output."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("package ") and "cortex-llm" in line:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "cortex-llm":
                return parts[2].rstrip(",")
    return None


def _get_latest_version_cached(config) -> Optional[str]:
    """Use cached update info when available."""
    cached_version = _coerce_optional_str(config.get_state_value("latest_version"))
    cached_at = _coerce_optional_str(config.get_state_value("latest_version_checked_at"))
    if not cached_version or not cached_at:
        return None
    try:
        checked_at = datetime.fromisoformat(cached_at)
    except ValueError:
        return None
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - checked_at > UPDATE_CACHE_TTL:
        return None
    return cached_version


def _cache_latest_version(config, latest_version: str) -> None:
    """Persist the latest version and check timestamp."""
    config.set_state_value("latest_version", latest_version)
    config.set_state_value("latest_version_checked_at", datetime.now(timezone.utc).isoformat())


def _fetch_latest_version() -> Optional[str]:
    """Fetch the latest version from pip in the current environment."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "cortex-llm"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        result = None

    if result and result.returncode == 0:
        combined = f"{result.stdout}\n{result.stderr}".strip()
        parsed = _parse_latest_version(combined)
        if parsed:
            return parsed

    installed_version = _get_installed_version()
    if installed_version:
        return installed_version

    return None


def _parse_latest_version(output: str) -> Optional[str]:
    """Parse the latest version from pip index output."""
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("available versions:"):
            _, versions = line.split(":", 1)
            parts = [v.strip() for v in versions.split(",") if v.strip()]
            return parts[0] if parts else None
    return None
