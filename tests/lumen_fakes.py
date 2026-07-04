"""Shared Lumen fakes for worker/service tests."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from cortex.lumen_runtime import LumenModel


def catalog(*entries: tuple[str, str, bool]) -> List[LumenModel]:
    """(name, quant, cached) triples → LumenModel list."""
    return [
        LumenModel(name=name, quant=quant, cached=cached, display_name=f"{name} {quant}")
        for name, quant, cached in entries
    ]


DEFAULT_CATALOG = catalog(
    ("qwen3-5-9b", "Q4_0", True),
    ("qwen3-5-9b", "Q8_0", False),
    ("qwen3-6-27b", "Q4_0", False),
)


class FakeLumenRuntime:
    """In-memory stand-in for LumenRuntime with call recording."""

    def __init__(
        self,
        *,
        models: Optional[List[LumenModel]] = None,
        ensure_ok: bool = True,
        ensure_message: str = "",
        pull_ok: bool = True,
        available_ok: bool = True,
    ) -> None:
        self.models = list(DEFAULT_CATALOG if models is None else models)
        self.ensure_ok = ensure_ok
        self.ensure_message = ensure_message
        self.pull_ok = pull_ok
        self.available_ok = available_ok
        self.ensure_calls: List[str] = []
        self.pull_calls: List[str] = []
        self.stopped = 0
        self._active: Optional[str] = None
        self._starting: Optional[str] = None  # tests set to simulate a boot in flight
        self._port = 8399

    def available(self) -> Tuple[bool, str]:
        if self.available_ok:
            return True, ""
        return False, "Lumen binary not found: lumen. Install Lumen: https://servelumen.com"

    def list_models(self) -> List[LumenModel]:
        return [] if not self.available_ok else list(self.models)

    def ensure_server(self, selector: str) -> Tuple[bool, str]:
        self.ensure_calls.append(selector)
        if not self.ensure_ok:
            return False, self.ensure_message or "lumen-server failed to become ready"
        self._active = selector
        return True, self.ensure_message or f"Lumen serving {selector} on port {self._port}."

    def active_selector(self) -> Optional[str]:
        return self._active or self._starting

    def serving_selector(self) -> Optional[str]:
        return self._active

    def starting_selector(self) -> Optional[str]:
        return self._starting

    def base_url(self) -> Optional[str]:
        return f"http://127.0.0.1:{self._port}/v1" if self._active else None

    def status(self) -> dict:
        if self._starting and not self._active:
            return {"running": True, "ready": False, "selector": self._starting, "port": self._port}
        if not self._active:
            return {"running": False}
        return {
            "running": True,
            "ready": True,
            "selector": self._active,
            "port": self._port,
            "uptime_seconds": 1.0,
        }

    def pull(
        self,
        selector: str,
        *,
        on_line: Optional[Callable[[str], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        self.pull_calls.append(selector)
        if on_line is not None:
            on_line("downloading 50%")
            on_line("converted to LBC")
        if cancel_requested is not None and cancel_requested():
            return False, "Download cancelled."
        if not self.pull_ok:
            return False, "network unreachable"
        name, _, quant = selector.partition(":")
        self.models = [
            LumenModel(
                name=m.name,
                quant=m.quant,
                cached=True if (m.name == name and m.quant.lower() == quant.lower()) else m.cached,
                display_name=m.display_name,
                size=m.size,
            )
            for m in self.models
        ]
        return True, f"Pulled {selector}."

    def stop(self) -> None:
        self.stopped += 1
        self._active = None
