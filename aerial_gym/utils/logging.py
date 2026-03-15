from __future__ import annotations

import logging
import os
from logging import Logger
from typing import Optional


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[37m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"
    fmt_str = (
        "[%(relativeCreated)d ms][%(name)s] - %(levelname)s : %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: cyan + fmt_str + reset,
        logging.INFO: white + fmt_str + reset,
        logging.WARNING: yellow + fmt_str + reset,
        logging.ERROR: red + fmt_str + reset,
        logging.CRITICAL: bold_red + fmt_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """Formatter without ANSI color codes, for file output."""

    fmt_str = (
        "[%(relativeCreated)d ms][%(name)s] - %(levelname)s : %(message)s (%(filename)s:%(lineno)d)"
    )

    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(self.fmt_str)
        return formatter.format(record)


class CustomLogger(Logger):
    def __init__(self, logger_name: str) -> None:
        super().__init__(logger_name)
        # Resolve log level from env var (set by run.py from config)
        env_level = os.environ.get("SF_LOG_LEVEL", "INFO")
        level = getattr(logging, env_level, logging.INFO)
        self.setLevel(level)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(level)
        self.ch.setFormatter(CustomFormatter())
        self.addHandler(self.ch)
        self._file_handler: Optional[logging.FileHandler] = None

    def add_file_handler(self, log_path: str) -> None:
        """Add a file handler that writes plain-text logs (no ANSI colors)."""
        if self._file_handler is not None:
            self.removeHandler(self._file_handler)
            self._file_handler.close()
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(self.level)
        fh.setFormatter(PlainFormatter())
        self.addHandler(fh)
        self._file_handler = fh

    def close_file_handler(self) -> None:
        """Close and remove the file handler if present."""
        if self._file_handler is not None:
            self.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def setLoggerLevel(self, level: int) -> None:
        self.setLevel(level)
        self.ch.setLevel(level)
        if self._file_handler is not None:
            self._file_handler.setLevel(level)
