# Third-party imports
from typing import Annotated
from collections.abc import Iterable, Iterator
import os
import sys
import pathlib
import stat
import itertools
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyarrow.csv as csv
import pyarrow.parquet as pq
from pydantic import BaseModel, AfterValidator, Field
from stream_zip import MemberFile, ZIP_64
from flask import g

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[2] / "utils"))
from paths import (
    MAGNITUDES_DIR,
    PHASORS_DIR,
    WAVEFORMS_DIR,
    WAVEFORMS_2024_10_DIR,
    WAVEFORMS_2024_10_CUTOFF,
)
from anonymize import deanonymize_elements
import utils
import phasor_utils

FILE_MODE = stat.S_IFREG | stat.S_IRUSR | stat.S_IWUSR
"""
File mode to use when generating ZIP files.

Corresponds to a regular file (``S_IFREG``) with owner read (``S_IRUSR``) and write
(``S_IWUSR``) permissions.
"""

WAVEFORM_DATE_FORMAT = "%Y-%m-%dT%H-%M-%S.%f"
"""Expected format of the date in waveform file names."""


def validate_resolution(resolution: timedelta) -> timedelta:
    """
    Pydantic validator for the resolution parameter of ``DataRequest``.

    :param resolution: Input resolution.
    :returns: Validated resolution.
    """
    resolution_seconds = resolution / timedelta(seconds=1)
    if resolution_seconds < 1:
        raise ValueError("resolution cannot be less than 1 second")
    if not resolution_seconds.is_integer():
        raise ValueError("resolution must be an integer number of seconds")
    return resolution


def validate_time(time: datetime) -> datetime:
    """
    Pydantic validator for the start or end time given in ``DataRequest``.

    :param time: Input time.
    :returns: Validated time.
    """
    requested_month = np.datetime64(time, "M")
    current_month = np.datetime64("now", "M")
    if requested_month >= current_month:
        raise ValueError(
            f"can only request data through the previous month ({current_month - 1})"
        )
    return time


class DataRequest(BaseModel):
    """Request body for a ``POST`` request to ``/data``."""

    magnitudes_for: list[str] = Field(default_factory=list[str])
    """Network elements to download magnitude data for."""

    phasors_for: list[str] = Field(default_factory=list[str])
    """Network elements to download phasor data for."""

    waveforms_for: list[str] = Field(default_factory=list[str])
    """Network elements to download waveform data for."""

    time_range: tuple[
        Annotated[datetime, AfterValidator(validate_time)],
        Annotated[datetime, AfterValidator(validate_time)],
    ]
    """Time range to retrieve data for."""

    resolution: Annotated[timedelta, AfterValidator(validate_resolution)] | None = None
    """Interval to sample data by."""


def count_bytes_of_contents(contents: Iterable[bytes]) -> Iterator[bytes]:
    """
    Keep a running count of the number of bytes of the given file contents in Flask `g`
    for use in logs.

    :param contents: Contents as an iterable of bytes.
    """
    for chunk in contents:
        g.num_bytes = g.get("num_bytes", 0) + len(chunk)  # Used in logs
        yield chunk


def make_member_file(path: str, contents: Iterable[bytes]) -> MemberFile:
    """
    Create a ``MemberFile`` tuple to pass to ``stream_zip()``.

    :param path: File path within the ZIP.
    :param contents: Contents of the file as an iterable of bytes.
    """
    return (path, datetime.now(), FILE_MODE, ZIP_64, count_bytes_of_contents(contents))


def get_date_paths(
    time_range: tuple[datetime, datetime],
    unit: str,
    directory: Path = Path(),
    extensions: list[str] | None = None,
    alternative_dirs: list[tuple[Path, datetime]] | None = None,
) -> Iterator[Path]:
    """
    Return paths of the form ``<directory>/<date><extension>``, where dates are of the
    specified ``unit`` between ``start_time`` and ``end_time``, inclusive. Any paths
    that do not exist will be skipped.

    :param time_range: A tuple of the start and end times.
    :param unit: The NumPy datetime unit to increment by (see
        https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units).
    :param directory: Base directory for output file paths.
    :param extensions: List of extension for output file paths. For each date, the
        extensions will be tried in order and the first existing path will be used.
    :param alternative_dirs: Optional, a list of alternative directory and cutoff time
        pairs. If a a given time is before the cutoff time, the alternative directory
        will be used instead. Cutoff times later in the list take priority.
    :return: The list of date file paths.
    """
    if extensions is None:
        extensions = [""]
    if alternative_dirs:
        alternative_dirs = [
            (alternative_dir, np.datetime64(cutoff, unit))
            for alternative_dir, cutoff in alternative_dirs
        ]
    start_date = np.datetime64(time_range[0], unit)
    end_date = np.datetime64(time_range[1], unit)
    current_date = start_date
    while current_date <= end_date:
        dir_to_use = directory
        if alternative_dirs:
            for alternative_dir, cutoff in alternative_dirs:
                if current_date < cutoff:
                    dir_to_use = alternative_dir
        for extension in extensions:
            date_path = dir_to_use / f"{current_date}{extension}"
            if date_path.exists():
                yield date_path
                break
        current_date += 1


def read_file(path: Path, chunk_size: int = 65536) -> Iterable[bytes]:
    """
    Read the file at the given path as chunks of bytes.

    :param path: Path to the file to read.
    :returns: An iterable of bytes.
    """
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def read_dataframes(files: Iterable[Path]) -> Iterator[pd.DataFrame]:
    """
    Read the given files into DataFrames. Note that the files are read in batches, so
    there may be multiple DataFrames returned per file. CSV and Parquet files are
    supported.

    :param files: An iterable of paths to CSV and/or Parquet files.
    :returns: An iterable of DataFrames.
    """
    for file in files:
        if file.suffix == ".csv":
            with csv.open_csv(
                file,
                convert_options=csv.ConvertOptions(null_values=["NaT"]),
            ) as reader:
                for batch in reader:
                    df = batch.to_pandas()
                    yield df
        elif file.suffix == ".parquet":
            parquet_file = pq.ParquetFile(file)
            for record_batch in parquet_file.iter_batches():
                df = record_batch.to_pandas()
                yield df


def dataframes_to_csv(dataframes: Iterable[pd.DataFrame]) -> Iterator[bytes]:
    """
    Convert the given DataFrames into a CSV file. The DataFrames will be concatenated in
    order, with the header row taken from the first.

    :param dataframes: An iterable of DataFrames.
    :returns: An iterable of bytes of a CSV file.
    """
    first = True
    for dataframe in dataframes:
        yield dataframe.to_csv(index=False, header=first).encode()
        if first:
            first = False


def select_dataframes(
    dataframes: Iterable[pd.DataFrame], time_range: tuple[datetime, datetime]
) -> Iterator[pd.DataFrame]:
    """
    Select data within the given time range.

    :param dataframes: An iterable of DataFrames with a column ``t`` of times increasing
        across all DataFrames.
    :param time_range: The time range to select data within.
    :returns: An iterable of DataFrames within ``time_range``.
    """
    start_time, end_time = time_range
    for df in dataframes:
        t_series = df["t"]
        if t_series.iloc[-1] < start_time:
            continue
        if t_series.iloc[0] > end_time:
            break
        yield df.iloc[
            t_series.searchsorted(start_time) : t_series.searchsorted(end_time)
        ]


def dataframes_to_dict(
    dataframes: Iterable[pd.DataFrame],
) -> dict[str, np.typing.NDArray] | None:
    """
    Load the given DataFrames into memory and turn them into a dictionary.

    :param dataframes: An iterable of DataFrames.
    :returns: A dictionary made from concatenating the DataFrames, or ``None`` if the
        ``dataframes`` is empty.
    """
    try:
        df_all = pd.concat(dataframes)
    except ValueError:
        # If pd.concat() fails, there is no data.
        return None
    return {column: df_all[column].to_numpy() for column in df_all.columns}


def rechunk_dataframe(
    df: pd.DataFrame, chunk_size: int = 65536
) -> Iterator[pd.DataFrame]:
    """
    Turn DataFrame back into chunks for further stream processing.

    :param df: A DataFrame.
    :param chunk_size: Optional, output DataFrames will contain at most this many rows.
    :returns: An iterable of DataFrames.
    """
    i = 0
    while i < len(df):
        yield df.iloc[i : i + chunk_size]
        i += chunk_size


def generate_magnitudes_files(
    zip_root_dir: str,
    real_elements: list[str],
    anon_elements: list[str],
    time_range: tuple[datetime, datetime],
    resolution: timedelta | None = None,
) -> Iterator[MemberFile]:
    """
    Return files (one per month) of magnitude data.

    :param zip_root_dir: Name of the root directory of the ZIP.
    :param real_elements: List of real element names. These will be used to read data.
    :param anon_elements: List of anonymized element names corresponding to each of
        ``real_elements`` in order. These will be used in the output file names.
    :param time_range: Time range to retrieve data for.
    :param resolution: Interval to sample data by.
    :returns: An iterable of ``MemberFiles`` to return in a streamed ZIP.
    """
    if not real_elements:
        return []
    resolution_np: np.timedelta64 | None = None
    if resolution is not None:
        resolution_np = np.timedelta64(resolution // timedelta(seconds=1), "s")
    for real_element, anon_element in zip(real_elements, anon_elements):
        file_paths = get_date_paths(
            time_range,
            unit="M",
            directory=MAGNITUDES_DIR / real_element,
            extensions=[".parquet"],
        )
        dataframes = read_dataframes(file_paths)
        dataframes = select_dataframes(dataframes, time_range)
        if resolution_np is not None:
            # Adapted from data.copy_subset_data()
            # This step loads all the data into memory at once, which gets rid of some
            # advantages of streaming. In the future, resampling could be improved to
            # work on an iterable of DataFrame chunks to improve memory usage.
            df_dict = dataframes_to_dict(dataframes)
            if df_dict is not None:
                df_resampled = pd.DataFrame(utils.downsample(df_dict, resolution_np))
                dataframes = rechunk_dataframe(df_resampled)
        yield make_member_file(
            f"{zip_root_dir}/magnitudes/{anon_element}.csv",
            dataframes_to_csv(dataframes),
        )


def generate_phasors_files(
    zip_root_dir: str,
    real_elements: list[str],
    anon_elements: list[str],
    time_range: tuple[datetime, datetime],
    resolution: timedelta | None,
) -> Iterator[MemberFile]:
    """
    Return files (one per month) of phasor data.

    :param zip_root_dir: Name of the root directory of the ZIP.
    :param real_elements: List of real element names. These will be used to read data.
    :param anon_elements: List of anonymized element names corresponding to each of
        ``real_elements`` in order. These will be used in the output file names.
    :param time_range: Time range to retrieve data for.
    :param resolution: Interval to sample data by.
    :returns: An iterable of ``MemberFiles`` to return in a streamed ZIP.
    """
    if not real_elements:
        return []
    delta_t_threshold: float | None = None
    time_column: np.typing.NDArray[np.datetime64] | None = None
    resampled_time_column: np.typing.NDArray[np.datetime64] | None = None
    if resolution is not None:
        resolution_seconds = resolution // timedelta(seconds=1)
        delta_t_threshold = resolution_seconds / 2
        time_column = pd.date_range(
            *time_range,
            freq=timedelta(seconds=resolution_seconds),
            unit="s",
            inclusive="left",
        ).to_numpy()
    else:
        # If there is no resolution, also select and return time column data. This
        # assumes a directory "t" exists on the server in the phasor data directory.
        # This will return a t.csv file of times.
        real_elements.append("t")
        anon_elements.append("t")
    for real_element, anon_element in zip(real_elements, anon_elements):
        file_paths = get_date_paths(
            time_range,
            unit="M",
            directory=PHASORS_DIR / real_element,
            extensions=[".csv", ".parquet"],
        )
        dataframes = read_dataframes(file_paths)
        dataframes = select_dataframes(dataframes, time_range)
        if time_column is not None and delta_t_threshold is not None:
            # This step loads all the data into memory at once, which gets rid of some
            # advantages of streaming. In the future, resampling could be improved to
            # work on an iterable of DataFrame chunks to improve memory usage.
            df_dict = dataframes_to_dict(dataframes)
            if df_dict is not None:
                resampled_time_column, df_resampled_dict = phasor_utils.align_phasors(
                    {"phasors": df_dict},
                    time_column_file=time_column,
                    delta_t_threshold=delta_t_threshold,
                )
                df_resampled_dict = df_resampled_dict["phasors"]
                del df_resampled_dict["t"]
                df_resampled_dict = {"t": resampled_time_column, **df_resampled_dict}
                df_resampled = pd.DataFrame(df_resampled_dict)
                dataframes = rechunk_dataframe(df_resampled)
        yield make_member_file(
            f"{zip_root_dir}/phasors/{anon_element}.csv",
            dataframes_to_csv(dataframes),
        )
    if isinstance(resampled_time_column, np.ndarray):
        # If a resolution was given, return the generated time column in a t.csv file.
        time_dataframes = rechunk_dataframe(pd.DataFrame({"t": resampled_time_column}))
        yield make_member_file(
            f"{zip_root_dir}/phasors/t.csv",
            dataframes_to_csv(time_dataframes),
        )


def generate_waveforms_files(
    zip_root_dir: str,
    real_elements: list[str],
    anon_elements: list[str],
    time_range: tuple[datetime, datetime],
    resolution: timedelta | None,
) -> Iterator[MemberFile]:
    """
    Return directories (one per day) of waveform data.

    :param zip_root_dir: Name of the root directory of the ZIP.
    :param real_elements: List of real element names. These will be used to read data.
    :param anon_elements: List of anonymized element names corresponding to each of
        ``real_elements`` in order. These will be used in the output directory names.
    :param time_range: Time range to retrieve data for.
    :param resolution: Interval to sample data by.
    :returns: An iterable of ``MemberFiles`` to return in a streamed ZIP.
    """
    if not real_elements:
        return []
    desired_timestamps: pd.DatetimeIndex | None = None
    delta_t_threshold: float | None = None
    if resolution is not None:
        resolution_seconds = resolution // timedelta(seconds=1)
        desired_timestamps = pd.date_range(
            *time_range,
            freq=timedelta(seconds=resolution_seconds),
            unit="s",
            inclusive="left",
        )
        delta_t_threshold = resolution_seconds / 2
    for real_element, anon_element in zip(real_elements, anon_elements):
        real_day_dirs = get_date_paths(
            time_range,
            unit="D",
            directory=WAVEFORMS_DIR / real_element,
            alternative_dirs=[
                (WAVEFORMS_2024_10_DIR / real_element, WAVEFORMS_2024_10_CUTOFF)
            ],
        )
        timestamp_dataframes: Iterator[pd.DataFrame] = (
            pd.to_datetime(
                os.listdir(real_day_dir), format=f"{WAVEFORM_DATE_FORMAT}.parquet"
            )
            .sort_values()
            .to_frame(index=False, name="t")
            for real_day_dir in real_day_dirs
        )
        timestamp_dataframes = select_dataframes(timestamp_dataframes, time_range)
        if desired_timestamps is not None and delta_t_threshold is not None:
            # This step loads all timestamps into memory at once, which gets rid of some
            # advantages of streaming. In the future, resampling could be improved to
            # work on an iterable of DataFrame chunks to improve memory usage.
            #
            # (Part of the reason this was done is for the integrity of finding the
            # closest timestamp at chunk boundaries, for example between days, so future
            # solutions should be careful.)
            try:
                timestamps: pd.Series[pd.Timestamp] = pd.concat(timestamp_dataframes)[
                    "t"
                ]
            except ValueError:
                # If pd.concat() fails, there is no data so we can continue to the next
                # element.
                continue
            nearest_timestamps: list[pd.Timestamp] = []
            for desired_timestamp in desired_timestamps:
                nearest_timestamp_index = timestamps.searchsorted(desired_timestamp)
                if nearest_timestamp_index < len(timestamps):
                    nearest_timestamp = timestamps.iloc[nearest_timestamp_index]
                    if (
                        abs(nearest_timestamp - desired_timestamp).seconds
                        < delta_t_threshold
                    ):
                        nearest_timestamps.append(nearest_timestamp)
            timestamp_dataframes = rechunk_dataframe(
                pd.to_datetime(nearest_timestamps).to_frame(index=False, name="t")
            )
        for timestamp_dataframe in timestamp_dataframes:
            timestamps: pd.Series[pd.Timestamp] = timestamp_dataframe["t"]
            for timestamp in timestamps:
                waveforms_dir = WAVEFORMS_DIR
                if timestamp < WAVEFORMS_2024_10_CUTOFF:
                    waveforms_dir = WAVEFORMS_2024_10_DIR
                day_dir_name = timestamp.strftime("%Y-%m-%d")
                timestamp_str = timestamp.strftime(WAVEFORM_DATE_FORMAT)[:-3]
                waveform_bytes = read_file(
                    waveforms_dir
                    / real_element
                    / day_dir_name
                    / f"{timestamp_str}.parquet"
                )
                yield make_member_file(
                    f"{zip_root_dir}/waveforms/{anon_element}/{day_dir_name}/{timestamp_str}.parquet",
                    waveform_bytes,
                )


def generate_files(
    zip_root_dir: str, data_request: DataRequest
) -> Iterator[MemberFile]:
    """
    Generate files for the given ``DataRequest``.

    :param zip_root_dir: Name of the root directory of the ZIP.
    :param data_request: The request for data.
    :returns: An iterable of ``MemberFiles`` to return in a streamed ZIP.
    """
    # We use itertools.chain() rather than yields so the body of this function is
    # evaluated when called, properly raising an HTTPException if deanonymize_elements()
    # raises one. Otherwise the error will be raised after data is already streaming to
    # the client.
    return itertools.chain(
        (
            make_member_file(
                f"{zip_root_dir}/request.json",
                (data_request.model_dump_json(indent=2).encode(), b"\n"),
            ),
        ),
        generate_magnitudes_files(
            zip_root_dir=zip_root_dir,
            real_elements=deanonymize_elements(data_request.magnitudes_for),
            anon_elements=data_request.magnitudes_for,
            time_range=data_request.time_range,
            resolution=data_request.resolution,
        ),
        generate_phasors_files(
            zip_root_dir=zip_root_dir,
            real_elements=deanonymize_elements(data_request.phasors_for),
            anon_elements=data_request.phasors_for,
            time_range=data_request.time_range,
            resolution=data_request.resolution,
        ),
        generate_waveforms_files(
            zip_root_dir=zip_root_dir,
            real_elements=deanonymize_elements(data_request.waveforms_for),
            anon_elements=data_request.waveforms_for,
            time_range=data_request.time_range,
            resolution=data_request.resolution,
        ),
    )
