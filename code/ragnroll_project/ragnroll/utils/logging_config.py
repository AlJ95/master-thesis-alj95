"""
Logging configuration for RAGnRoll framework.
Provides structured logging with configurable verbosity and output formats.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        # Add color if stream supports it
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
            if record.levelname == 'WARNING':
                record.msg = f"{self.COLORS['WARNING']}{record.msg}{self.COLORS['RESET']}"
            elif record.levelname == 'ERROR':
                record.msg = f"{self.COLORS['ERROR']}{record.msg}{self.COLORS['RESET']}"

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add any extra attributes from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value

        return json.dumps(log_entry)


class DeduplicationFilter(logging.Filter):
    """Filter to prevent duplicate log messages."""

    def __init__(self):
        super().__init__()
        self.seen_messages = set()
        self.max_cache_size = 1000

    def filter(self, record):
        # Create a unique key for this message
        key = (record.levelno, record.getMessage(), record.module)

        if key in self.seen_messages:
            return False

        # Add to seen messages
        self.seen_messages.add(key)

        # Prevent cache from growing too large
        if len(self.seen_messages) > self.max_cache_size:
            self.seen_messages.clear()

        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    enable_deduplication: bool = True,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the RAGnRoll framework.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        json_format: Whether to use JSON format for logs
        enable_deduplication: Whether to prevent duplicate messages
        enable_colors: Whether to use colored output for console

    Returns:
        Root logger instance
    """

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger('ragnroll')
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    elif enable_colors:
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)-35s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)-35s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    if enable_deduplication:
        console_handler.addFilter(DeduplicationFilter())

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)

        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)-35s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f'ragnroll.{name}')


def configure_from_environment() -> logging.Logger:
    """
    Configure logging from environment variables.

    Environment variables:
        RAGNROLL_LOG_LEVEL: Logging level (default: INFO)
        RAGNROLL_LOG_FILE: Log file path (optional)
        RAGNROLL_LOG_JSON: Use JSON format (default: false)
        RAGNROLL_LOG_DEDUP: Enable deduplication (default: true)
        RAGNROLL_LOG_COLORS: Enable colors (default: true)

    Returns:
        Configured root logger
    """

    level = os.getenv('RAGNROLL_LOG_LEVEL', 'INFO')
    log_file = os.getenv('RAGNROLL_LOG_FILE')
    json_format = os.getenv('RAGNROLL_LOG_JSON', 'false').lower() == 'true'
    deduplication = os.getenv('RAGNROLL_LOG_DEDUP', 'true').lower() == 'true'
    colors = os.getenv('RAGNROLL_LOG_COLORS', 'true').lower() == 'true'

    return setup_logging(
        level=level,
        log_file=log_file,
        json_format=json_format,
        enable_deduplication=deduplication,
        enable_colors=colors
    )


# Global logger instance
logger = None


def init_logging():
    """Initialize logging system. Call this at application startup."""
    global logger
    if logger is None:
        logger = configure_from_environment()
    return logger
