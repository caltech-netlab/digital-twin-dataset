# Third-party imports
from typing import Any, Literal
from collections.abc import Iterator
import sys
import pathlib
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pydantic

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import API_USAGE_LOG_PATH
from logs import IntervalStr, format_datetime
from data import DataRequest

NP_UNITS: dict[IntervalStr, str] = {
    "month": "M",
    "day": "D",
    "hour": "h",
    "minute": "m",
    "second": "s",
}


def get_log_paths(
    base: Path, start: datetime, stop: datetime, interval: IntervalStr
) -> Iterator[Path]:
    unit = NP_UNITS[interval]
    start_date = np.datetime64(start, unit)
    end_date = np.datetime64(stop, unit)
    current_date = start_date
    latest_existing_date = None
    while current_date <= end_date:
        dt_str = format_datetime(current_date.astype(datetime), interval)
        log_path = Path(f"{base}.{dt_str}")
        if log_path.exists():
            latest_existing_date = current_date
            yield log_path
        current_date += 1
    if (
        latest_existing_date is None or latest_existing_date < end_date
    ) and base.exists():
        yield base


def print_log_warning(log_path: Path, line_number: int, message: str) -> None:
    print(f"{log_path}:{line_number}: {message}", file=sys.stderr)


def get_logs(
    base: Path, start: datetime, stop: datetime, interval: IntervalStr
) -> Iterator[dict[str, Any]]:
    start = (
        start.replace(tzinfo=timezone.utc)
        if start.tzinfo is None
        else start.astimezone(timezone.utc)
    )
    stop = (
        stop.replace(tzinfo=timezone.utc)
        if stop.tzinfo is None
        else stop.astimezone(timezone.utc)
    )
    start_naive_utc = start.replace(tzinfo=None)
    stop_naive_utc = stop.replace(tzinfo=None)
    for log_path in get_log_paths(base, start_naive_utc, stop_naive_utc, interval):
        with open(log_path) as log_file:
            for line_number, line in enumerate(log_file, start=1):
                log_str = line.strip()
                try:
                    log = json.loads(log_str)
                except json.JSONDecodeError:
                    print_log_warning(
                        log_path, line_number, f"Failed to parse log: {log_str!r}"
                    )
                    continue
                if not isinstance(log, dict):
                    print_log_warning(
                        log_path, line_number, f"Log is not a dictionary: {log!r}"
                    )
                    continue
                log_created = log.get("created")
                try:
                    log_created_dt = datetime.fromisoformat(log_created)
                except (TypeError, ValueError):
                    print_log_warning(
                        log_path,
                        line_number,
                        f"Log created time is not valid: {log_created!r}",
                    )
                    continue
                log_data_request = log.get("data_request")
                if log_data_request is not None:
                    try:
                        data_request_obj = DataRequest.model_validate(
                            log["data_request"]
                        )
                    except pydantic.ValidationError:
                        print_log_warning(
                            log_path,
                            line_number,
                            f"Log data request could not be parsed: {log_data_request!r}",
                        )
                        continue
                log["created"] = log_created_dt
                log["data_request"] = data_request_obj
                if start <= log_created_dt <= stop:
                    yield log


def get_usage(
    start: datetime,
    stop: datetime,
    interval: IntervalStr = "month",
) -> dict[str, Any]:
    logs = get_logs(API_USAGE_LOG_PATH, start, stop, interval)
    usage = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "duration": 0,
        "num_bytes": 0,
    }
    for log in logs:
        usage["total_requests"] += 1
        usage["successful_requests"] += log.get("level") == "INFO"
        usage["failed_requests"] += log.get("level") == "ERROR"
        usage["duration"] += log.get("duration") or 0
        usage["num_bytes"] += log.get("num_bytes") or 0
    return usage


def get_usage_by_user(
    start: datetime,
    stop: datetime,
    interval: IntervalStr = "month",
    num_top_users: int | None = None,
    sort_by: Literal["total_requests", "duration", "num_bytes"] = "total_requests",
) -> dict[str, Any]:
    logs = get_logs(API_USAGE_LOG_PATH, start, stop, interval)
    usage_by_user: dict[int, Any] = {}
    for log in logs:
        github_id = log.get("github_id")
        if github_id not in usage_by_user:
            usage_by_user[github_id] = {
                "github_username": log.get("github_username"),
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "duration": 0,
                "num_bytes": 0,
            }
        usage = usage_by_user[github_id]
        usage["total_requests"] += 1
        usage["successful_requests"] += log.get("level") == "INFO"
        usage["failed_requests"] += log.get("level") == "ERROR"
        usage["duration"] += log.get("duration") or 0
        usage["num_bytes"] += log.get("num_bytes") or 0
    top_users = sorted(
        usage_by_user.keys(),
        key=lambda user: usage_by_user[user][sort_by],
        reverse=True,
    )
    return {user: usage_by_user[user] for user in top_users[:num_top_users]}
