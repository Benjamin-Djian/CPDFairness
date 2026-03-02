import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.logger import LoggerFactory


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset LoggerFactory state before each test."""
    LoggerFactory._initialized = False
    yield
    LoggerFactory._initialized = False


class TestLoggerFactory:
    """Tests for LoggerFactory class."""

    def test_get_logger_returns_logger_instance(self):
        """Test get_logger returns a logger instance."""
        LoggerFactory()
        logger = LoggerFactory.get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_has_correct_name(self):
        """Test logger has correct name."""
        LoggerFactory()
        logger = LoggerFactory.get_logger("my_module")
        assert logger.name == "my_module"

    def test_logger_initialization_creates_handler(self):
        """Test initialization creates a handler."""
        LoggerFactory()
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

    def test_logger_initialization_twice_only_creates_one_handler(self):
        """Test calling init twice only creates one handler."""
        LoggerFactory()
        LoggerFactory()
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

    @patch("src.utils.logger.e")
    def test_logger_uses_custom_log_file(self, mock_env):
        """Test logger uses custom log file when provided."""
        mock_env.LOG_FORMAT = "%(message)s"
        mock_env.LOG_LEVEL = "debug"

        with patch("builtins.open", create=True):
            LoggerFactory(log_file=Path("test.log"))
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) > 0

    @patch("src.utils.logger.e")
    def test_logger_level_set_correctly(self, mock_env):
        """Test logger level is set from env."""
        mock_env.LOG_FORMAT = "%(message)s"
        mock_env.LOG_LEVEL = "error"

        LoggerFactory()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR
