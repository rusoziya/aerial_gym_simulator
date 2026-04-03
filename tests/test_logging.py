"""Tests for aerial_gym.utils.logging — CustomLogger, no Isaac Gym required.

Uses importlib to load the logging module directly, bypassing aerial_gym.__init__
which imports isaacgym.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the logging module via importlib to avoid isaacgym import
# ---------------------------------------------------------------------------
_log_spec = importlib.util.spec_from_file_location(
    "aerial_gym.utils.logging", "aerial_gym/utils/logging.py"
)
_log_mod = importlib.util.module_from_spec(_log_spec)
sys.modules[_log_spec.name] = _log_mod
_log_spec.loader.exec_module(_log_mod)

CustomLogger = _log_mod.CustomLogger
CustomFormatter = _log_mod.CustomFormatter
PlainFormatter = _log_mod.PlainFormatter

# Counter to ensure unique logger names (Python caches loggers by name)
_counter = 0


def _unique_name(prefix: str = "test") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestCustomLoggerDefaults:
    """CustomLogger should default to INFO level."""

    def test_default_level_is_info(self) -> None:
        saved = os.environ.pop("SF_LOG_LEVEL", None)
        try:
            logger = CustomLogger(_unique_name())
            assert logger.level == logging.INFO
        finally:
            if saved is not None:
                os.environ["SF_LOG_LEVEL"] = saved

    def test_is_python_logger(self) -> None:
        logger = CustomLogger(_unique_name())
        assert isinstance(logger, logging.Logger)

    def test_has_stream_handler(self) -> None:
        logger = CustomLogger(_unique_name())
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_different_names_different_loggers(self) -> None:
        a = CustomLogger(_unique_name("a"))
        b = CustomLogger(_unique_name("b"))
        assert a.name != b.name


class TestSFLogLevelEnvVar:
    """SF_LOG_LEVEL env var should control the initial log level."""

    def test_debug_level_from_env(self) -> None:
        saved = os.environ.get("SF_LOG_LEVEL")
        try:
            os.environ["SF_LOG_LEVEL"] = "DEBUG"
            logger = CustomLogger(_unique_name())
            assert logger.level == logging.DEBUG
        finally:
            if saved is not None:
                os.environ["SF_LOG_LEVEL"] = saved
            else:
                os.environ.pop("SF_LOG_LEVEL", None)

    def test_warning_level_from_env(self) -> None:
        saved = os.environ.get("SF_LOG_LEVEL")
        try:
            os.environ["SF_LOG_LEVEL"] = "WARNING"
            logger = CustomLogger(_unique_name())
            assert logger.level == logging.WARNING
        finally:
            if saved is not None:
                os.environ["SF_LOG_LEVEL"] = saved
            else:
                os.environ.pop("SF_LOG_LEVEL", None)

    def test_invalid_env_var_falls_back_to_info(self) -> None:
        saved = os.environ.get("SF_LOG_LEVEL")
        try:
            os.environ["SF_LOG_LEVEL"] = "BOGUS_LEVEL"
            logger = CustomLogger(_unique_name())
            assert logger.level == logging.INFO
        finally:
            if saved is not None:
                os.environ["SF_LOG_LEVEL"] = saved
            else:
                os.environ.pop("SF_LOG_LEVEL", None)


class TestFileHandler:
    """add_file_handler / close_file_handler lifecycle."""

    def test_add_file_handler_creates_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = CustomLogger(_unique_name())
        logger.add_file_handler(str(log_file))
        logger.info("hello file handler")
        # Flush to disk
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()
        contents = log_file.read_text()
        assert "hello file handler" in contents

    def test_file_handler_writes_plain_text(self, tmp_path: Path) -> None:
        log_file = tmp_path / "plain.log"
        logger = CustomLogger(_unique_name())
        logger.add_file_handler(str(log_file))
        logger.warning("check plain output")
        for h in logger.handlers:
            h.flush()
        contents = log_file.read_text()
        # PlainFormatter should not contain ANSI escape sequences
        assert "\x1b[" not in contents
        assert "check plain output" in contents

    def test_close_file_handler_removes_handler(self, tmp_path: Path) -> None:
        log_file = tmp_path / "close_test.log"
        logger = CustomLogger(_unique_name())
        logger.add_file_handler(str(log_file))
        file_handlers_before = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers_before) == 1

        logger.close_file_handler()
        file_handlers_after = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers_after) == 0

    def test_close_file_handler_noop_when_none(self) -> None:
        logger = CustomLogger(_unique_name())
        # Should not raise even when no file handler was ever added
        logger.close_file_handler()

    def test_add_replaces_existing_file_handler(self, tmp_path: Path) -> None:
        log_a = tmp_path / "a.log"
        log_b = tmp_path / "b.log"
        logger = CustomLogger(_unique_name())
        logger.add_file_handler(str(log_a))
        logger.add_file_handler(str(log_b))
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        logger.info("only in b")
        for h in logger.handlers:
            h.flush()
        assert "only in b" in log_b.read_text()


class TestPlainFormatter:
    """PlainFormatter should produce output without ANSI escape codes."""

    def test_no_ansi_codes_in_output(self) -> None:
        formatter = PlainFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\x1b[" not in output
        assert "test message" in output

    def test_includes_logger_name(self) -> None:
        formatter = PlainFormatter()
        record = logging.LogRecord(
            name="my_module",
            level=logging.WARNING,
            pathname="my_module.py",
            lineno=42,
            msg="warn msg",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        assert "my_module" in output


class TestCustomFormatter:
    """CustomFormatter should include ANSI codes for console output."""

    def test_info_output_contains_ansi(self) -> None:
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="info msg",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\x1b[" in output
        assert "info msg" in output

    def test_warning_output_uses_yellow(self) -> None:
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="warn",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        # Yellow ANSI code
        assert "\x1b[33;20m" in output


class TestSetLoggerLevel:
    """setLoggerLevel should update logger, stream handler, and file handler."""

    def test_changes_logger_level(self) -> None:
        logger = CustomLogger(_unique_name())
        logger.setLoggerLevel(logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_changes_stream_handler_level(self) -> None:
        logger = CustomLogger(_unique_name())
        logger.setLoggerLevel(logging.ERROR)
        assert logger.ch.level == logging.ERROR

    def test_changes_file_handler_level(self, tmp_path: Path) -> None:
        log_file = tmp_path / "level.log"
        logger = CustomLogger(_unique_name())
        logger.add_file_handler(str(log_file))
        logger.setLoggerLevel(logging.CRITICAL)
        assert logger._file_handler.level == logging.CRITICAL
