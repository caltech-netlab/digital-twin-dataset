# Third-party imports
from typing import Annotated
from collections.abc import Iterable, Iterator
import os
import io
import sys
import pathlib
import stat
import itertools
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from pydantic import BaseModel, AfterValidator, Field
from stream_zip import MemberFile, ZIP_64

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[2] / "utils"))
from paths import MAGNITUDES_DIR, PHASORS_DIR, WAVEFORMS_DIR, WAVEFORMS_2024_10_DIR
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


def make_member_file(path: str, contents: Iterable[bytes]) -> MemberFile:
    """
    Create a ``MemberFile`` tuple to pass to ``stream_zip()``.

    :param path: File path within the ZIP.
    :param contents: Contents of the file as an iterable of bytes.
    """
    return (path, datetime.now(), FILE_MODE, ZIP_64, contents)


def get_date_paths(
    time_range: tuple[datetime, datetime],
    unit: str,
    directory: Path = Path(),
    extension: str = "",
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
    :param extension: Extension for output file paths.
    :param alternative_dirs: Optional, a list of alternative directory and cutoff time
        pairs. If a a given time is before the cutoff time, the alternative directory
        will be used instead. Cutoff times later in the list take priority.
    :return: The list of date file paths.
    """
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
        date_path = dir_to_use / f"{current_date}{extension}"
        if date_path.exists():
            yield date_path
        current_date += 1


def read_csv_dataframes(files: Iterable[Path]) -> Iterator[pd.DataFrame]:
    """
    Read the given CSV files into DataFrames. There may be multiple DataFrames returned
    per file.

    :param files: An iterable of paths to CSV files.
    :returns: An iterable of DataFrames.
    """
    for file in files:
        with csv.open_csv(file) as reader:
            for batch in reader:
                df = batch.to_pandas()
                yield df


def read_parquet_dataframes(files: Iterable[Path]) -> Iterator[pd.DataFrame]:
    """
    Read the given Parquet files into DataFrames. Note that there may be multiple
    DataFrames returned per file.

    :param files: An iterable of paths to Parquet files.
    :returns: An iterable of DataFrames.
    """
    for file in files:
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
        f = io.BytesIO()
        csv.write_csv(
            data=pa.Table.from_pandas(dataframe),
            output_file=f,
            write_options=csv.WriteOptions(include_header=first),
        )
        if first:
            first = False
        yield f.getvalue()


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
    resolution_np: np.timedelta64 | None = None
    if resolution is not None:
        # Adapted from data.copy_subset_data()
        resolution_np = np.timedelta64(resolution // timedelta(seconds=1), "s")
    for real_element, anon_element in zip(real_elements, anon_elements):
        file_paths = get_date_paths(
            time_range,
            unit="M",
            directory=MAGNITUDES_DIR / real_element,
            extension=".parquet",
        )
        dataframes = read_parquet_dataframes(file_paths)
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
    delta_t_threshold: float | None = None
    time_column: np.typing.NDArray[np.datetime64] | None = None
    if resolution is not None:
        # Adapted from data.copy_subset_align_phasors()
        resolution_seconds = resolution // timedelta(seconds=1)
        delta_t_threshold = resolution_seconds / 2
        time_column = pd.date_range(
            *time_range,
            freq=timedelta(seconds=resolution_seconds),
            unit="s",
            inclusive="left",
        ).to_numpy()
    for real_element, anon_element in zip(real_elements, anon_elements):
        file_paths = get_date_paths(
            time_range,
            unit="M",
            directory=PHASORS_DIR / real_element,
            extension=".csv",
        )
        dataframes = read_csv_dataframes(file_paths)
        dataframes = select_dataframes(dataframes, time_range)
        if time_column is not None and delta_t_threshold is not None:
            # Adapted from data.copy_subset_align_phasors()
            # This step loads all the data into memory at once, which gets rid of some
            # advantages of streaming. In the future, resampling could be improved to
            # work on an iterable of DataFrame chunks to improve memory usage.
            df_dict = dataframes_to_dict(dataframes)
            if df_dict is not None:
                time_column, df_resampled_dict = phasor_utils.align_phasors(
                    {"phasors": df_dict},
                    time_column_file=time_column,
                    delta_t_threshold=delta_t_threshold,
                )
                df_resampled_dict = df_resampled_dict["phasors"]
                df_resampled_dict["t"] = time_column
                df_resampled = pd.DataFrame(df_resampled_dict)
                dataframes = rechunk_dataframe(df_resampled)
        yield make_member_file(
            f"{zip_root_dir}/phasors/{anon_element}.csv",
            dataframes_to_csv(dataframes),
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
    desired_timestamps: pd.DatetimeIndex | None = None
    delta_t_threshold: float | None = None
    if resolution is not None:
        # Adapted from data.list_waveform()
        resolution_seconds = resolution // timedelta(seconds=1)
        delta_t_threshold = resolution_seconds / 2
        desired_timestamps = pd.date_range(
            *time_range,
            freq=timedelta(seconds=resolution_seconds),
            unit="s",
            inclusive="left",
        )
    for real_element, anon_element in zip(real_elements, anon_elements):
        for real_day_directory in get_date_paths(
            time_range,
            unit="D",
            directory=WAVEFORMS_DIR / real_element,
            alternative_dirs=[
                (WAVEFORMS_2024_10_DIR / real_element, datetime(2024, 11, 1))
            ],
        ):
            day_directory_name = real_day_directory.parts[-1]
            timestamp_paths = os.listdir(real_day_directory)
            timestamps: pd.DatetimeIndex = pd.to_datetime(
                timestamp_paths, format=f"{WAVEFORM_DATE_FORMAT}.parquet"
            ).sort_values()
            timestamps = timestamps[
                timestamps.searchsorted(time_range[0]) : timestamps.searchsorted(
                    time_range[1]
                )
            ]
            if desired_timestamps is not None and delta_t_threshold is not None:
                # Adapted from data.list_waveform()
                nearest_timestamps: list[pd.Timestamp] = []
                for desired_timestamp in desired_timestamps:
                    nearest_timestamp_index = timestamps.searchsorted(desired_timestamp)
                    if nearest_timestamp_index < len(timestamps):
                        nearest_timestamp = timestamps[nearest_timestamp_index]
                    if (
                        abs(nearest_timestamp - desired_timestamp).seconds
                        < delta_t_threshold
                    ):
                        nearest_timestamps.append(nearest_timestamp)
                timestamps = pd.DatetimeIndex(nearest_timestamps)
            for timestamp in timestamps:
                timestamp_str = timestamp.strftime(WAVEFORM_DATE_FORMAT)[:-3]
                table = pq.read_table(real_day_directory / f"{timestamp_str}.parquet")
                f = io.BytesIO()
                csv.write_csv(data=table, output_file=f)
                yield make_member_file(
                    f"{zip_root_dir}/waveforms/{anon_element}/{day_directory_name}/{timestamp_str}.csv",
                    (f.getvalue(),),
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
