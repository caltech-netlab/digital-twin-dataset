# Third-party imports
from typing import Any, Literal
from collections.abc import Iterator
from collections import Counter
import sys
import pathlib
from pathlib import Path
import traceback
from datetime import datetime, timedelta, timezone
import numpy as np
from pydantic import ValidationError

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
from paths import API_USAGE_LOG_PATH
from logs import IntervalStr, INTERVAL_FORMAT_STR, APIUsageLog

LogT = APIUsageLog  # If more log types are added, use a `TypeVar` here.

NP_UNIT: dict[IntervalStr, str] = {
    "month": "M",
    "day": "D",
    "hour": "h",
    "minute": "m",
    "second": "s",
}
"""
Dictionary mapping interval strings to NumPy datetime units (see
https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units).
"""


def get_log_paths(
    base: Path, start: datetime, end: datetime, interval: IntervalStr
) -> Iterator[Path]:
    """
    Return paths with date extensions appended for all dates between ``start`` and
    ``end`` (inclusive) at the given rotation ``interval``. Any paths that do not exist
    will be skipped. If there is no existing path for the end date, the base path (i.e.
    the live log) will be returned as well, if it exists.

    :param base: Path to the base log file.
    :param start: Start date for log paths.
    :param end: End date for log paths.
    :param interval: Interval logs were rotated at.
    """
    unit = NP_UNIT[interval]
    start_date = np.datetime64(start, unit)
    end_date = np.datetime64(end, unit)
    current_date = start_date
    latest_existing_date = None
    while current_date <= end_date:
        current_dt: datetime = current_date.astype(datetime)
        date_str = current_dt.strftime(INTERVAL_FORMAT_STR[interval])
        log_path = Path(f"{base}.{date_str}")
        if log_path.exists():
            latest_existing_date = current_date
            yield log_path
        current_date += 1
    if (
        latest_existing_date is None or latest_existing_date < end_date
    ) and base.exists():
        yield base


def get_logs(
    base: Path,
    log_type: type[LogT],
    start: datetime,
    end: datetime,
    interval: IntervalStr = "month",
) -> Iterator[LogT]:
    """
    Return logs for the given base log file and loge type between ``start`` and ``end``
    (inclusive) at the given rotation ``interval``. For invalid logs, the validation
    error will be printed and the log will be skipped.

    :param base: Path to the base log file.
    :param start: Start date to return logs from.
    :param end: End date to return logs to.
    :param interval: Interval logs were rotated at. Defaults to ``"month"``.
    """
    start = (
        start.replace(tzinfo=timezone.utc)
        if start.tzinfo is None
        else start.astimezone(timezone.utc)
    )
    end = (
        end.replace(tzinfo=timezone.utc)
        if end.tzinfo is None
        else end.astimezone(timezone.utc)
    )
    start_naive_utc = start.replace(tzinfo=None)
    end_naive_utc = end.replace(tzinfo=None)
    for log_path in get_log_paths(base, start_naive_utc, end_naive_utc, interval):
        with open(log_path) as log_file:
            for line_number, log_str in enumerate(log_file, start=1):
                try:
                    log = log_type.model_validate_json(log_str)
                except ValidationError as exc:
                    print(f"{log_path}:{line_number}: Skipping log", file=sys.stderr)
                    for exc_line in traceback.format_exception_only(exc):
                        print(f"{exc_line}", file=sys.stderr)
                    continue
                if start <= log.created <= end:
                    yield log


def get_api_usage_logs(
    start: datetime, end: datetime, interval: IntervalStr = "month"
) -> Iterator[APIUsageLog]:
    """
    Return API usage logs between ``start`` and ``end`` (inclusive) at the given
    rotation ``interval``. For invalid logs, the validation error will be printed and
    the log will be skipped.

    :param start: Start date to return logs from.
    :param end: End date to return logs to.
    :param interval: Interval logs were rotated at. Defaults to ``"month"``.
    """
    yield from get_logs(
        base=API_USAGE_LOG_PATH,
        log_type=APIUsageLog,
        start=start,
        end=end,
        interval=interval,
    )


def get_usage(
    start: datetime,
    end: datetime,
    interval: IntervalStr = "month",
) -> dict[str, Any]:
    """
    Return total usage information between ``start`` and ``end`` (inclusive) at the
    given rotation ``interval``. The returned dictionary includes:

    - `total_requests`: Total number of requests.
    - `successful_requests`: Number of requests that were completed successfully.
    - `failed_requests`: Number of requests that failed for a variety of reasons,
      include authentication errors, request format errors, or server bugs. Note that
      some data may have been streamed before the error occurred.
    - `duration`: Total duration across all requests.
    - `num_bytes`: Total number of bytes of data sent across all requests.

    :param start: Start date to return logs from.
    :param end: End date to return logs to.
    :param interval: Interval logs were rotated at. Defaults to ``"month"``.
    """
    logs = get_api_usage_logs(start, end, interval)
    usage = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "duration": timedelta(0),
        "num_bytes": 0,
    }
    for log in logs:
        usage["total_requests"] += 1
        usage["successful_requests"] += log.level == "INFO"
        usage["failed_requests"] += log.level == "ERROR"
        usage["duration"] += log.duration or timedelta(0)
        usage["num_bytes"] += log.num_bytes or 0
    return usage


def get_usage_by_user(
    start: datetime,
    end: datetime,
    interval: IntervalStr = "month",
    num_top_users: int | None = None,
    sort_by: Literal["total_requests", "duration", "num_bytes"] = "total_requests",
) -> dict[int | None, dict[str, Any]]:
    """
    Return usage information by user between ``start`` and ``end`` (inclusive) at the
    given rotation ``interval``. For each user's GitHub ID number (or ``None`` if
    unauthenticated), the returned dictionary includes:

    - ``total_requests``: Total number of requests.
    - ``successful_requests``: Number of requests that were completed successfully.
    - ``failed_requests``: Number of requests that failed for a variety of reasons,
      include authentication errors, request format errors, or server bugs. Note that
      some data may have been streamed before the error occurred.
    - ``duration``: Total duration across all requests.
    - ``num_bytes``: Total number of bytes of data sent across all requests.

    Users in the dictionary will be sorted according to ``sort_by``, descending.

    :param start: Start date to return logs from.
    :param end: End date to return logs to.
    :param interval: Interval logs were rotated at. Defaults to ``"month"``.
    :param num_top_users: If given, only this may top users will be included.
    :param sort_by: Field to sort users by, defaulting to ``"total_requests"``.
    """
    logs = get_api_usage_logs(start, end, interval)
    usage_by_user: dict[int, Any] = {}
    for log in logs:
        if log.github_id not in usage_by_user:
            usage_by_user[log.github_id] = {
                "github_username": log.github_username,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "duration": timedelta(0),
                "num_bytes": 0,
            }
        usage = usage_by_user[log.github_id]
        usage["total_requests"] += 1
        usage["successful_requests"] += log.level == "INFO"
        usage["failed_requests"] += log.level == "ERROR"
        usage["duration"] += log.duration or timedelta(0)
        usage["num_bytes"] += log.num_bytes or 0
    top_users = sorted(
        usage_by_user.keys(),
        key=lambda user: usage_by_user[user][sort_by],
        reverse=True,
    )
    return {user: usage_by_user[user] for user in top_users[:num_top_users]}


def get_top_meters(
    start: datetime,
    end: datetime,
    interval: IntervalStr = "month",
    num_top_elements: int | None = None,
) -> dict[str, dict[str, int]]:
    """
    Return how many times each meter was requested for each type of data. The returned
    dictionary includes:

    - ``top_magnitudes``: Dictionary from meter names to number of requests, sorted
      descending by count.
    - ``top_waveforms``: Dictionary from meter names to number of requests, sorted
      descending by count.
    - ``top_phasors``: Dictionary from meter names to number of requests, sorted
      descending by count.

    :param start: Start date to return logs from.
    :param end: End date to return logs to.
    :param interval: Interval logs were rotated at. Defaults to ``"month"``.
    :param num_top_elements: If given, only this may top elements will be included for
        each type of data.
    """
    logs = get_api_usage_logs(start, end, interval)
    magnitudes_counter = Counter()
    waveforms_counter = Counter()
    phasors_counter = Counter()
    for log in logs:
        if log.data_request is not None:
            magnitudes_counter.update(log.data_request.magnitudes_for)
            waveforms_counter.update(log.data_request.waveforms_for)
            phasors_counter.update(log.data_request.phasors_for)
    return {
        "top_magnitudes": {
            k: v for k, v in magnitudes_counter.most_common(num_top_elements)
        },
        "top_waveforms": {
            k: v for k, v in waveforms_counter.most_common(num_top_elements)
        },
        "top_phasors": {k: v for k, v in phasors_counter.most_common(num_top_elements)},
    }
