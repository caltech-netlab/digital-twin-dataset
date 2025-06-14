# Third-party imports
from typing import Literal
import os
import sys
import time
import pathlib
import logging
import json
import traceback
from logging.handlers import WatchedFileHandler
from datetime import datetime, timezone
from filelock import FileLock
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

    def _format_interval_start(self, interval_start: datetime) -> datetime:
        """
        Format the given interval start time as a string for use in the rotated file
        name.

        :param interval_start: Start time of an interval.
        """
        if self.interval == "month":
            return interval_start.strftime("%Y-%m")
        if self.interval == "day":
            return interval_start.strftime("%Y-%m-%d")
        if self.interval == "hour":
            return interval_start.strftime("%Y-%m-%d-%H")
        if self.interval == "minute":
            return interval_start.strftime("%Y-%m-%d-%H-%M")
        return interval_start.strftime("%Y-%m-%d-%H-%M-%S")

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
            os.rename(
                self.baseFilename,
                f"{self.baseFilename}.{self._format_interval_start(file_interval_start)}",
            )

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
        created = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        request_started = g.get("request_started")
        duration = (
            round(time.perf_counter() - request_started, 2)
            if isinstance(request_started, float)
            else None
        )
        data_request = g.get("data_request")
        data_request_dict = (
            data_request.model_dump(mode="json")
            if isinstance(data_request, DataRequest)
            else None
        )
        _, exc, tb = sys.exc_info()
        log_dict = {
            "level": record.levelname,
            "created": created,
            "path": request.path,
            "github_username": g.get("github_username"),
            "github_id": g.get("github_id"),
            "duration": duration,
            "num_bytes": g.get("num_bytes"),
            "data_request": data_request_dict,
            "exception": None if exc is None else traceback.format_exception_only(exc),
            "traceback": None if tb is None else traceback.format_tb(tb),
        }
        record.json_str = json.dumps(log_dict)
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
