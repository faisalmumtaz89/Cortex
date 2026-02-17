from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

from cortex.ui import version_check


class _ConfigStub:
    def __init__(self) -> None:
        self.system = SimpleNamespace(auto_update_check=True)
        self._state: dict[str, str] = {}

    def is_setting_explicit(self, _key: str) -> bool:
        return False

    def get_state_value(self, key: str):
        return self._state.get(key)

    def set_state_value(self, key: str, value: str) -> None:
        self._state[key] = value


def test_get_installed_version_prefers_package_metadata(monkeypatch) -> None:
    monkeypatch.setattr(version_check, "package_version", lambda name: "1.2.3")
    monkeypatch.setattr(version_check, "_get_pipx_installed_version", lambda: "9.9.9")

    assert version_check._get_installed_version() == "1.2.3"


def test_get_installed_version_falls_back_to_pipx_when_metadata_missing(monkeypatch) -> None:
    def _missing(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(version_check, "package_version", _missing)
    monkeypatch.setattr(version_check, "_get_pipx_installed_version", lambda: "1.0.5")

    assert version_check._get_installed_version() == "1.0.5"


def test_get_update_status_uses_installed_version(monkeypatch) -> None:
    config = _ConfigStub()
    monkeypatch.setattr(version_check, "_get_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(version_check, "_fetch_latest_version", lambda: "1.1.0")

    status = version_check.get_update_status(config=config)

    assert status is not None
    assert status.current_version == "1.0.0"
    assert status.latest_version == "1.1.0"
