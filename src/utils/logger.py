import logging
import sys

import src.utils.env as e


class LoggerFactory:
    def __init__(self, log_format=e.LOG_FORMAT):
        root_logger = logging.getLogger()
        hand = logging.StreamHandler(sys.stdout)
        hand.setFormatter(logging.Formatter(log_format))

        root_logger.addHandler(hand)

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

    @staticmethod
    def get_logger(name: str):
        return logging.getLogger(name)
