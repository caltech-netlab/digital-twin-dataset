# Third-party imports
import os
import sys
import time
import pathlib
import logging
from logging.handlers import WatchedFileHandler
from filelock import FileLock
from flask import has_request_context, request, g

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

API_USAGE_LOG_PATH = LOGS_DIR / "api_usage.log"

api_usage_lock = FileLock(f"{API_USAGE_LOG_PATH}.lock")


class WatchedFileHandlerLocking(WatchedFileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        with api_usage_lock:
            super().emit(record)


class AddRequestInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not has_request_context():
            return False
        record.request_path = request.path
        record.github_id = g.get("github_id")
        record.github_username = g.get("github_username")
        request_started = g.get("request_started")
        num_bytes = g.get("num_bytes")
        record.request_duration = (
            f"{time.perf_counter() - request_started:.2f} s"
            if isinstance(request_started, float)
            else None
        )
        record.num_bytes = f"{num_bytes} B" if isinstance(num_bytes, int) else None
        return True


api_usage_formatter = logging.Formatter(
    "{levelname} - {asctime} - {request_path} - {github_username} ({github_id}) - {request_duration} - {num_bytes} - {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
)

api_usage_file_handler = WatchedFileHandlerLocking(API_USAGE_LOG_PATH)
api_usage_file_handler.setLevel(logging.DEBUG)
api_usage_file_handler.setFormatter(api_usage_formatter)
api_usage_file_handler.addFilter(AddRequestInfoFilter())

api_usage_logger = logging.getLogger("dataset_api_app.api_usage")
api_usage_logger.setLevel(logging.DEBUG)
api_usage_logger.addHandler(api_usage_file_handler)
