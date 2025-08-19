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

def _is_console_logging_handler(handler):
    return (isinstance(handler, logging.StreamHandler) and handler.stream in {sys.stdout, sys.stderr})

def _get_first_console_handler(handlers):
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler
    return None

def setup_tqdm_logging():
    """Configures the root logger to log using tqdm."""
    log = logging.getLogger()
    console_handler = _get_first_console_handler(log.handlers)
    if console_handler is None:
        return
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(console_handler.formatter)
    tqdm_handler.setLevel(console_handler.level)
    log.removeHandler(console_handler)
    log.addHandler(tqdm_handler)

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

