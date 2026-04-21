"""
Central logging setup for the backend service.

The logging system serves two goals:
1. Keep rich backend output visible in the terminal during development.
2. Persist structured-enough local log files for long-running job diagnosis.
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "logs"
APP_LOG_PATH = LOG_DIRECTORY / "app.log"
TASK_LOG_PATH = LOG_DIRECTORY / "tasks.log"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    Timed rotating handler that degrades gracefully on rollover contention.

    Local development often has multiple backend-related processes touching the
    same log files, especially when reload mode or parallel validation scripts
    are involved. On Windows, that can make `os.rename` fail during rollover.
    We treat that as a recoverable condition and keep writing to the current
    file instead of surfacing noisy logging tracebacks.
    """

    def doRollover(self) -> None:
        """Rotate the log file unless another process currently blocks the rename."""

        try:
            super().doRollover()
        except PermissionError:
            if self.stream is None:
                self.stream = self._open()


def ensure_log_directory() -> None:
    """Create the backend log directory if it does not already exist."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _build_file_handler(log_path: Path) -> TimedRotatingFileHandler:
    """
    Create a daily rotating file handler with UTF-8 output.

    Daily rotation matches long-running local jobs better than size-only
    rotation and keeps logs grouped by run date for easier diagnosis.
    """

    handler = SafeTimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=5,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.suffix = "%Y-%m-%d"
    return handler


def _build_console_handler() -> logging.StreamHandler:
    """Create a console handler for interactive local development output."""

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def configure_logging() -> None:
    """
    Configure process-wide logging once.

    The setup is idempotent so importing modules can safely call it without
    stacking duplicate handlers.
    """

    ensure_log_directory()

    app_logger = logging.getLogger("river")
    if app_logger.handlers:
        return

    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    app_logger.addHandler(_build_console_handler())
    app_logger.addHandler(_build_file_handler(APP_LOG_PATH))

    task_logger = logging.getLogger("river.tasks")
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    task_logger.addHandler(_build_console_handler())
    task_logger.addHandler(_build_file_handler(TASK_LOG_PATH))


def get_logger(name: str) -> logging.Logger:
    """Return an application logger after ensuring logging is configured."""

    configure_logging()
    return logging.getLogger(name)
