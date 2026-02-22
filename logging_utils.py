#!/usr/bin/env python3
"""
Shared logging configuration utility for YouTube summarizer scripts.

Provides consistent logging with:
- Dual output (file + console)
- Source information (filename:lineno)
- Immediate flush for critical logs
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent / ".logs"
LOG_DIR.mkdir(exist_ok=True)

FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


class ImmediateFlushHandler(logging.FileHandler):
    """File handler that flushes immediately after each log entry."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(
    name: str = None,
    log_file: str = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up a logger with dual output (file + console).

    Args:
        name: Logger name (typically __name__)
        log_file: Log file name (defaults to <module_name>.log)
        console_level: Minimum level for console output
        file_level: Minimum level for file output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    if log_file is None:
        module_name = name or "app"
        if "." in module_name:
            module_name = module_name.rsplit(".", 1)[-1]
        log_file = f"{module_name}.log"

    log_path = LOG_DIR / log_file

    file_handler = ImmediateFlushHandler(log_path, mode='a', encoding='utf-8', delay=False)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(FORMATTER)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(FORMATTER)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def flush_logger(logger: logging.Logger):
    """Flush all handlers for the given logger."""
    for handler in logger.handlers:
        handler.flush()
