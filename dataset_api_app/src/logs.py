# Third-party imports
import os
import sys
import time
import pathlib
import logging
import json
import traceback
from logging.handlers import WatchedFileHandler
from datetime import datetime, timezone, timedelta
from filelock import FileLock
from flask import has_request_context, request, g

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import LOGS_DIR
from data import DataRequest

API_USAGE_LOG_PATH = LOGS_DIR / "api_usage.log"
os.makedirs(LOGS_DIR, exist_ok=True)
api_usage_lock = FileLock(f"{API_USAGE_LOG_PATH}.lock")


class CustomWatchedFileHandler(WatchedFileHandler):
    """Watched file handler with locking and monthly log rotation."""
    def emit(self, record: logging.LogRecord) -> None:
        with api_usage_lock:
            # If the record was created in a new month from when the log file was
            # created, start a new log file.
            record_created = datetime.fromtimestamp(record.created)
            file_created = datetime.fromtimestamp(os.path.getctime(self.baseFilename))
            record_created_tuple = (record_created.year, record_created.month)
            file_created_tuple = (file_created.year, file_created.month)
            if record_created_tuple > file_created_tuple:
                os.rename(
                    self.baseFilename,
                    f"{self.baseFilename}.{file_created.strftime('%Y-%m')}",
                )
            super().emit(record)


class AddRequestInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not has_request_context():
            return False
        created = (
            datetime.fromtimestamp(round(record.created), tz=timezone.utc)
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


api_usage_formatter = logging.Formatter("%(json_str)s")

api_usage_file_handler = CustomWatchedFileHandler(API_USAGE_LOG_PATH)
api_usage_file_handler.setLevel(logging.DEBUG)
api_usage_file_handler.setFormatter(api_usage_formatter)
api_usage_file_handler.addFilter(AddRequestInfoFilter())

api_usage_logger = logging.getLogger("dataset_api_app.api_usage")
api_usage_logger.setLevel(logging.DEBUG)
api_usage_logger.addHandler(api_usage_file_handler)
