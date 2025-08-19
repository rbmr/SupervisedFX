"""
Code adapted from tqdm.contrib.logging
"""

import logging.config
import time

from tqdm import tqdm

import logging

class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def setup_logging():
    """Sets up the logging configuration for the entire application."""
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'tqdm': {
                'class': f'{__name__}.{TqdmLoggingHandler.__name__}',
                'formatter': 'default',
                'level': logging.INFO,
            },
        },
        'loggers': {
            'src.trade.dp': {
                'level': logging.WARNING,
            },
            'src.models.sl': {
                'level': logging.INFO,
            },
        },
        'root': {
            'level': logging.INFO,
            'handlers': ['tqdm'],
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)


if __name__ == "__main__":
    setup_logging()

    for i in tqdm(range(10), desc="Processing items"):
        if i == 3:
            logging.info(f"Processed item {i}. This message respects the progress bar.")
        if i == 7:
            logging.info(f"A warning occurred at item {i}, but the bar is fine.")
        time.sleep(0.2)

