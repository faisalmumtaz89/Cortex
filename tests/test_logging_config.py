from __future__ import annotations

import logging
import sys
import time
from types import SimpleNamespace

import pytest

from cortex.logging_config import _MaskingFormatter, configure_logging


def _formatted(message: str, *, logger: str = "openai._base_client", exc_info=None) -> str:
    record = logging.LogRecord(logger, logging.DEBUG, __file__, 1, message, None, exc_info)
    return _MaskingFormatter("%(name)s %(message)s").format(record)


@pytest.mark.parametrize(
    ("message", "written"),
    [
        # As an SDK logs a request, under whatever logger name it has.
        ("Sending HTTP Request: POST http://u:sekret@h:9/v1/chat/completions",
         "Sending HTTP Request: POST http://***@h:9/v1/chat/completions"),
        ('HTTP Request: POST https://h/v1?api_key=sekret&region=eu "HTTP/1.1 200 OK"',
         'HTTP Request: POST https://h/v1?api_key=***&region=eu "HTTP/1.1 200 OK"'),
        ("two: http://a:sekret@h/x and wss://sekret@i/y?x-token=sekret;sig=sekret",
         "two: http://***@h/x and wss://***@i/y?x-token=***;sig=***"),
        ("a password with an @: http://u:sek@ret@h/v1", "a password with an @: http://***@h/v1"),
        ('url="https://h/v1?API_KEY=sekret" pwd: https://h?password=sekret&pwd=sekret&code=sekret',
         'url="https://h/v1?API_KEY=***" pwd: https://h?password=***&pwd=***&code=***'),
        ("https://h/v1?auth=sekret&credential=sekret", "https://h/v1?auth=***&credential=***"),
        # Not a credential.
        ("mail user@example.com about https://h/users/a@b?page=2&q=words",
         "mail user@example.com about https://h/users/a@b?page=2&q=words"),
        ("Logging initialized level=INFO file=/home/u/.cortex/cortex.log",
         "Logging initialized level=INFO file=/home/u/.cortex/cortex.log"),
    ],
)
def test_a_log_line_has_the_credentials_of_its_urls_masked(message, written):
    assert _formatted(message) == f"openai._base_client {written}"


def test_a_traceback_in_the_log_is_masked_too():
    try:
        raise ConnectionError("could not reach http://u:sekret@h/v1?token=sekret")
    except ConnectionError:
        written = _formatted("probe failed", logger="cortex.update_check", exc_info=sys.exc_info())

    assert "sekret" not in written
    assert "http://***@h/v1?token=***" in written and "Traceback" in written


def test_masking_a_long_line_takes_no_longer_than_reading_it():
    """A server's error text is logged as it came: masking it must not hold
    up the turn, whatever it is made of."""
    started = time.monotonic()
    for unit in ("?", "?a", "&", "://", "@", "="):
        _formatted(unit * 200_000)
    assert time.monotonic() - started < 5


def test_the_log_file_and_debug_output_mask_what_any_logger_writes(tmp_path, capsys):
    """The masking is on the handlers Cortex sets up: the file, and stderr
    with debug_mode."""
    root = logging.getLogger()
    before = (root.level, root.handlers[:])
    config = SimpleNamespace(
        logging=SimpleNamespace(log_level="INFO", log_file=tmp_path / "cortex.log"),
        developer=SimpleNamespace(debug_mode=True),
    )
    try:
        configure_logging(config)
        logging.getLogger("some.sdk").info("POST https://u:sekret@h/v1?api_key=sekret")
        logging.getLogger("httpx2").info("HTTP Request: POST https://h/v1")
        logging.getLogger("httpx2").warning("retrying https://u:sekret@h/v1")
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = before[1]
        root.setLevel(before[0])

    for written in ((tmp_path / "cortex.log").read_text(encoding="utf-8"), capsys.readouterr().err):
        assert "sekret" not in written
        assert "INFO some.sdk POST https://***@h/v1?api_key=***" in written
        assert "WARNING httpx2 retrying https://***@h/v1" in written
        assert "HTTP Request" not in written  # a line for every request is noise
