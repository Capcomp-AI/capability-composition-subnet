"""Logging setup shared by every entry point."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: str | os.PathLike[str] | None = None,
    max_bytes: int = 64 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure root logging once per process.

    Repeated calls are ignored so importing a module that sets up logging cannot
    duplicate handlers and double every line.
    """
    global _configured
    if _configured:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)

    handlers: list[logging.Handler] = [stream]

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        rotating.setFormatter(formatter)
        handlers.append(rotating)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    # These are chatty at DEBUG and drown out anything useful.
    for noisy in ("httpx", "httpcore", "urllib3", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
