# Third-party imports
from typing import Literal
import os
import sys
import time
import pathlib
import logging
import traceback
from logging.handlers import WatchedFileHandler
from datetime import datetime, timedelta, timezone
from filelock import FileLock
from pydantic import BaseModel, ByteSize
from flask import has_request_context, request, g

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import LOGS_DIR, API_USAGE_LOG_PATH
from data import DataRequest

os.makedirs(LOGS_DIR, exist_ok=True)

api_usage_log_lock = FileLock(f"{API_USAGE_LOG_PATH}.lock")
"""Lock for the API usage log file."""

IntervalStr = Literal["month", "day", "hour", "minute", "second"]
"""Type hint for interval string."""

INTERVAL_FORMAT_STR: dict[IntervalStr, str] = {
    "month": "%Y-%m",
    "day": "%Y-%m-%d",
    "hour": "%Y-%m-%d-%H",
    "minute": "%Y-%m-%d-%H-%M",
    "second": "%Y-%m-%d-%H-%M-%S",
}
"""Dictionary mapping interval strings to date format strings."""


class APIUsageLog(BaseModel):
    """Log corresponding to an API request."""

    level: Literal["INFO", "ERROR"]
    """Level of the log (e.g. `"INFO"` for success or `"ERROR"` for failure)."""

    created: datetime
    """When the log was created."""

    path: str
    """API path of the request (e.g. ``"/data"``)."""

    github_username: str | None
    """GitHub username of the user, or ``None`` if unauthenticated."""

    github_id: int | None
    """GitHub ID number of the user, or ``None`` if unauthenticated."""

    duration: timedelta | None
    """
    Duration of the request, or ``None`` if an error occurred before duration started
    being tracked.
    """

    num_bytes: ByteSize | None
    """
    Number of bytes of data returned, or ``None`` if an error occurred before the number
    of bytes started being tracked.
    """

    data_request: DataRequest | None
    """Data requested, or ``None`` if the request was invalid."""

    exception: list[str] | None
    """Exception that occurred, if any."""

    traceback: list[str] | None
    """Traceback for the exception, if any."""


class CustomFileHandler(WatchedFileHandler):
    """
    Custom file handler with locking and timed log rotation, for use across multiple
    worker processes.
    """

    def __init__(self, filename: str | pathlib.Path, interval: IntervalStr) -> None:
        self.interval: IntervalStr = interval
        # Setting delay to True avoids opening the file until a log is about to be
        # created, which avoids creating and potentially rotating an empty log file.
        super().__init__(filename, delay=True)

    def _interval_start(self, timestamp: float) -> datetime:
        """Start time of the interval ``timestamp`` is in.

        :param timestamp: UTC timestamp.
        """
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if self.interval == "month":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if self.interval == "day":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.interval == "hour":
            return dt.replace(minute=0, second=0, microsecond=0)
        if self.interval == "minute":
            return dt.replace(second=0, microsecond=0)
        return dt.replace(microsecond=0)

    def _rotate(self, record_created: float) -> None:
        """
        Rotate the current log file if needed via renaming. Note that a new log file
        will not be created by this function.

        :param record_created: Time that the current record being logged was created.
        """
        if not os.path.isfile(self.baseFilename):
            return
        # If the record was created in a new interval from when the log file was
        # created, start a new log file.
        record_interval_start = self._interval_start(record_created)
        file_interval_start = self._interval_start(os.path.getctime(self.baseFilename))
        if record_interval_start > file_interval_start:
            date_str = file_interval_start.strftime(INTERVAL_FORMAT_STR[self.interval])
            os.rename(self.baseFilename, f"{self.baseFilename}.{date_str}")

    def emit(self, record: logging.LogRecord) -> None:
        """
        Write the given record to the log file. Since this class inherits from
        ``WatchedFileHandler``, the log file will be reopened (and created if necessary)
        if the current log file was renamed by rotation (either by this worker or
        another worker).

        :param record: Current record being logged.
        """
        with api_usage_log_lock:
            self._rotate(record.created)
            super().emit(record)


class ApiUsageFilter(logging.Filter):
    """Filter that adds API usage information to the current record being logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add information to the current record being logged in a custom property called
        ``json_str``. If the Flask request context is not active, this function will
        return False and the record will not be logged.

        :param record: Current record being logged.
        """
        if not has_request_context():
            return False
        request_started = g.get("request_started")
        duration_seconds = (
            time.perf_counter() - request_started
            if isinstance(request_started, float)
            else None
        )
        _, exc, tb = sys.exc_info()
        api_usage_log = APIUsageLog(
            level=record.levelname,
            created=record.created,
            path=request.path,
            github_username=g.get("github_username"),
            github_id=g.get("github_id"),
            duration=duration_seconds,
            num_bytes=g.get("num_bytes"),
            data_request=g.get("data_request"),
            exception=None if exc is None else traceback.format_exception_only(exc),
            traceback=None if tb is None else traceback.format_tb(tb),
        )
        record.json_str = api_usage_log.model_dump_json()
        return True


api_usage_file_handler = CustomFileHandler(API_USAGE_LOG_PATH, interval="month")
api_usage_file_handler.setFormatter(logging.Formatter("%(json_str)s"))
api_usage_file_handler.addFilter(ApiUsageFilter())

api_usage_logger = logging.getLogger("dataset_api_app.api_usage")
"""
Logger for logging API usage. Call ``error`` in a Flask request if an error occurred,
and call ``info`` to log a successful request.
"""

api_usage_logger.setLevel(logging.DEBUG)
api_usage_logger.addHandler(api_usage_file_handler)
