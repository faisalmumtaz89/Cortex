"""Shared logging configuration helpers."""

from __future__ import annotations

import logging
import re
import sys
from logging import Handler
from pathlib import Path

from cortex.cloud.router import SECRET_PARAMETER
from cortex.config import Config, keep_private

_USERINFO = re.compile(r"(://)[^/?#\s]+@")
# The name cannot hold a ?, or a long run of them is matched from every one.
_PARAMETER = re.compile(r"([?&;])([^=&;?\s]+)=([^&;\s\"'<>]+)")


class _MaskingFormatter(logging.Formatter):
    """Formats a record with the credentials of any URL in it masked: a
    user:password@ before the host, and a query parameter such as api_key=.
    The provider SDKs and the HTTP library under them log request URLs, under
    logger names that change between their versions, so the line is masked
    where it is written, whichever logger it came from."""

    def format(self, record: logging.LogRecord) -> str:
        text = _USERINFO.sub(r"\1***@", super().format(record))
        return _PARAMETER.sub(
            lambda found: f"{found[1]}{found[2]}=***" if SECRET_PARAMETER.search(found[2])
            else found[0],
            text,
        )


def configure_logging(config: Config) -> None:
    """Configure application logging outputs from config."""
    log_level_name = str(getattr(config.logging, "log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_file = Path(
        getattr(config.logging, "log_file", Path.home() / ".cortex" / "cortex.log")
    ).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    keep_private(log_file)

    file_handler: Handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(_MaskingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    handlers: list[Handler] = [file_handler]
    if getattr(config.developer, "debug_mode", False):
        stderr_handler: Handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(log_level)
        stderr_handler.setFormatter(
            _MaskingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handlers.append(stderr_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    # A line for every request is noise in the log (httpx2 and httpcore2 are
    # the names in newer SDK versions).
    for http_logger in ("httpx", "httpcore", "httpx2", "httpcore2"):
        logging.getLogger(http_logger).setLevel(max(log_level, logging.WARNING))
    logging.getLogger(__name__).info(
        "Logging initialized level=%s file=%s", log_level_name, log_file
    )
