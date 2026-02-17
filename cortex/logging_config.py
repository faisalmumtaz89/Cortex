"""Shared logging configuration helpers."""

from __future__ import annotations

import logging
import sys
from logging import Handler
from pathlib import Path

from cortex.config import Config


def configure_logging(config: Config) -> None:
    """Configure application logging outputs from config."""
    log_level_name = str(getattr(config.logging, "log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_file = Path(getattr(config.logging, "log_file", Path.home() / ".cortex" / "cortex.log")).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler: Handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    handlers: list[Handler] = [file_handler]
    if getattr(config.developer, "debug_mode", False):
        stderr_handler: Handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(log_level)
        stderr_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handlers.append(stderr_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    logging.getLogger(__name__).info("Logging initialized level=%s file=%s", log_level_name, log_file)

