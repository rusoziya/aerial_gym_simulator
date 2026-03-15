"""Tests for aerial_gym.utils.logging — CustomLogger."""
import isaacgym  # noqa: F401
import pytest
import logging


class TestCustomLogger:
    def setup_method(self):
        from aerial_gym.utils.logging import CustomLogger
        self.Logger = CustomLogger

    def test_creates_logger(self):
        logger = self.Logger("test_module")
        assert logger is not None

    def test_is_python_logger(self):
        logger = self.Logger("test_module2")
        assert isinstance(logger, logging.Logger)

    def test_has_handlers(self):
        logger = self.Logger("test_module3")
        assert len(logger.handlers) > 0

    def test_different_names_different_loggers(self):
        a = self.Logger("module_a")
        b = self.Logger("module_b")
        assert a.name != b.name
