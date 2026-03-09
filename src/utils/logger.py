import logging
import sys
from pathlib import Path

import src.utils.env as e


class LoggerFactory:
    _initialized = False

    def __init__(self, log_file: Path | None = None, log_format=e.LOG_FORMAT):
        if LoggerFactory._initialized:
            return
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(logging.Formatter(log_format))

        root_logger.addHandler(handler)

        logging_levels = {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'warn': logging.WARNING,
            'info': logging.INFO,
            'test': logging.DEBUG,
            'debug': logging.DEBUG
        }
        root_logger.setLevel(logging_levels[e.LOG_LEVEL])

        LoggerFactory._initialized = True

    @staticmethod
    def reset() -> None:
        LoggerFactory._initialized = False
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)
