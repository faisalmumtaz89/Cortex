"""Lazy inference-engine proxy for worker startup optimization."""

from __future__ import annotations

import threading
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class LazyProxy(Generic[T]):
    """Thread-safe lazy proxy that instantiates a target object on first use."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._instance: Optional[T] = None
        self._lock = threading.Lock()

    def get_if_initialized(self) -> Optional[T]:
        return self._instance

    def get(self) -> T:
        if self._instance is not None:
            return self._instance
        with self._lock:
            if self._instance is None:
                self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get(), name)
