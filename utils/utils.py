# Third-party imports
import os
import shutil
import traceback
import sys
import copy
import datetime
from pathlib import Path
from dateutil import parser
from ntplib import NTPClient
import pytz
import math
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import json
from io import StringIO
import re
import operator
import subprocess

# First-party imports
from global_param import *


"""CSV and Parquet data read/write helper functions"""


def read_ts(
    filepath,
    datetimespan=(None, None),
    timedeltaspan=(None, None),
    mode="nearest",
    timedeltaspan_ref="front",
    usecols=None,
    dtypes=dict(),
    default_dtype="float",
    output_type="np",
):
    """
    API-level entry point for reading timeseries in either csv or parquet formats.
    Accepts both single files and files broken down by monthly chunks.
    A note on output time format: the output time column is in utc time.
    Assumptions
        - Time increases monotonically with rows
    :param filepath: string or path
    :param datetimespan: tuple of two str, specifying the start and/or end time.
    :param timedeltaspan: tuple of two dictionaries, specifying the start and/or end.
        e.g. [{'hours': 4, 'minutes': 2}, None] returns data after 4 hours 2 minutes from the first row
        e.g. [{'minutes': 2}, {'hours': 4}] returns data between the 2nd minute and the 4th hour.
        e.g. [None, {'hours': 4}] returns data from the beginning of file to the 4th hour.
        Note: the reference timestamp becomes the last row if timedeltaspan_ref == 'back',
            and the first entry in tuple is always the smaller one (closer to reference).
    :param mode: str, one of {'nearest', 'strict'}, for datetimespan or timdeltaspan bounds,
        - 'nearest': include the nearest data point
        - 'strict': the first datapoint starts strictly after the bound specified,
            the last datapoint is strictly before the bound specified.
            The convention is that the start of the time interval is inclusive and the end
            of the time interval is exclusive, i.e. [t0, t1).
    :param timedeltaspan_ref: str, one of {'front', 'back'}, whether to take timedelta from
        the first or last row
    :param usecols: tuple of ints or str, columns index or name to read, see np.loadtxt documentation
    :param dtypes: dict, k:v = column_name:numpy_dtype, any entries supplied here will
        override the default dtypes in global_param.STD_DTYPE
    :param output_type: one of 'np' or 'pd', whether to output a dictionary of numpy arrays
        or a pandas.DataFrame
    :return: a tuple of
        - dictionary of numpy arrays, i.e. custom dataframe
        - err: 0 - no error, 1 - empty csv, 2 - empty csv with only header
            3 - has data, but not enough to fulfill timedeltaspan specified
        - byte location where reading ends
    """

    def read_ts_singlefile(
        filepath,
        datetimespan=(None, None),
        timedeltaspan=(None, None),
        mode="nearest",
        timedeltaspan_ref="front",
        usecols=None,
        dtypes=dict(),
        default_dtype="float",
        output_type="np",
    ):
        """Dynamically select read function based on file extension"""
        filepath = os.path.splitext(filepath)[0]
        if os.path.exists(filepath + ".csv"):
            filepath += ".csv"
            return read_csv_ts_singlefile(**locals())
        elif os.path.exists(filepath + ".parquet"):
            filepath += ".parquet"
            return read_parquet_ts_singlefile(**locals())
        else:
            raise ValueError(f"Input file {filepath} is neither csv nor parquet")
    filepath = str(Path(os.path.splitext(filepath)[0]).expanduser())
    # Special case: singlefile
    if os.path.exists(filepath + ".parquet"):
        filepath += ".parquet"
        return read_ts_singlefile(
            **{k: v for k, v in locals().items() if k != "read_ts_singlefile"}
        )
    elif os.path.exists(filepath + ".csv"):
        filepath += ".csv"
        return read_ts_singlefile(
            **{k: v for k, v in locals().items() if k != "read_ts_singlefile"}
        )
    try:
        # Special case: empty
        if (not os.path.exists(filepath)) or (len(os.listdir(filepath)) == 0):
            return ({} if output_type == "np" else pd.DataFrame({})), 1

        # Scan all existing monthly files
        files = os.listdir(filepath)
        ext = os.path.splitext(files[0])[-1]
        all_months = np.array(
            list(
                set(
                    [
                        os.path.splitext(f)[0]
                        for f in files
                        if (f[0] != ".") and (f[:5] != "empty")
                    ]
                )
            ),
            "datetime64[M]",
        )
        all_months = np.sort(all_months)
        m0, m1 = 0, len(all_months) - 1
        # Format start/end time if specified
        if any(datetimespan) or any(timedeltaspan):
            if any(datetimespan):
                if datetimespan[0]:
                    m0 = (
                        strptime_np(datetimespan[0], unit="M")
                        if type(datetimespan[0]) is str
                        else np.datetime64(datetimespan[0]).astype("datetime64[M]")
                    )
                    m0 = np_searchsorted(
                        all_months, m0, mode="nearest_after", inclusive=True
                    )
                if datetimespan[1]:
                    m1 = (
                        strptime_np(datetimespan[1], unit="M")
                        if type(datetimespan[1]) is str
                        else np.datetime64(datetimespan[1]).astype("datetime64[M]")
                    )
                    m1 = np_searchsorted(
                        all_months, m1, mode="nearest_before", inclusive=True
                    )
            elif any(timedeltaspan):
                t0, t1 = None, None
                if timedeltaspan_ref == "front":
                    t_ref = read_ts_singlefile(
                        os.path.join(filepath, str(all_months[0]) + ext), usecols=["t"]
                    )[0]["t"][0]
                    if timedeltaspan[0]:
                        t0 = t_ref + timedelta_(timedeltaspan[0], mode="np")
                    if timedeltaspan[1]:
                        t1 = t_ref + timedelta_(timedeltaspan[1], mode="np")
                else:
                    t_ref = read_ts_singlefile(
                        os.path.join(filepath, str(all_months[-1]) + ext), usecols=["t"]
                    )[0]["t"][-1]
                    if timedeltaspan[1]:
                        t0 = t_ref - timedelta_(timedeltaspan[1], mode="np")
                    if timedeltaspan[0]:
                        t1 = t_ref - timedelta_(timedeltaspan[0], mode="np")
                m0 = (
                    np_searchsorted(
                        all_months,
                        t0.astype("datetime64[M]"),
                        mode="nearest_after",
                        inclusive=True,
                    )
                    if t0
                    else m0
                )
                m1 = (
                    np_searchsorted(
                        all_months,
                        t1.astype("datetime64[M]"),
                        mode="nearest_before",
                        inclusive=True,
                    )
                    if t1
                    else m1
                )

        # Read monthly files
        df_list = []
        for i in range(m0, m1 + 1):
            dtspan = [None, None]
            if (i == m0) and datetimespan[0]:
                dtspan[0] = datetimespan[0]
            elif (i == m0) and (
                (timedeltaspan[0] and timedeltaspan_ref == "front")
                or (timedeltaspan[1] and timedeltaspan_ref == "back")
            ):
                dtspan[0] = t0
            if (i == m1) and datetimespan[1]:
                dtspan[1] = datetimespan[1]
            elif (i == m1) and (
                (timedeltaspan[0] and timedeltaspan_ref == "back")
                or (timedeltaspan[1] and timedeltaspan_ref == "front")
            ):
                dtspan[1] = t1
            df, err = read_ts_singlefile(
                os.path.join(filepath, str(all_months[i])),
                datetimespan=dtspan,
                dtypes=dtypes,
                default_dtype=default_dtype,
                output_type=output_type,
                usecols=usecols,
                mode=mode,
            )
            if err != 1:
                df_list.append(df)
        # Concatenate dataframes
        if not len(df_list):
            if os.path.exists(os.path.join(filepath, "empty.parquet")) or os.path.exists(
                os.path.join(filepath, "empty.csv")
            ):
                return read_ts_singlefile(os.path.join(filepath, "empty"))
            else:
                return ({} if output_type == "np" else pd.DataFrame({})), 1
        else:
            df = concatenate_df(df_list, output_type=output_type)
            return df, 0 if len(df) else 2
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Reading file failed: {filepath}")
        raise e


def read_csv_ts_singlefile(
    filepath,
    bytespan=(None, None),
    datetimespan=(None, None),
    timedeltaspan=(None, None),
    mode="nearest",
    timedeltaspan_ref="front",
    usecols=None,
    dtypes=None,
    default_dtype="float",
    return_end_byte=False,
    output_type="np",
):
    """
    High-performance reading csv file into numpy arrays.
    A note on output time format: the output time column is in utc time.
    Assumptions
        - First row is header and header is complete
        - Time increases monotonically with rows
        - String time format (in csv file, in datetimespan) conforms to TIME_FMT
        - At most one of {bytespan, rowidxspan, datetimespan, timedeltaspan}
            is active, the rest is ignored, with priorities in the order above
    :param filepath: string or path
    :param bytespan: tuple of two ints, start and end byte locations specifying the
        range of data to read
    :param datetimespan: tuple of two str, specifying the start and/or end time.
    :param timedeltaspan: tuple of two dictionaries, specifying the start and/or end.
        e.g. [{'hours': 4, 'minutes': 2}, None] returns data after 4 hours 2 minutes from the first row
        e.g. [{'minutes': 2}, {'hours': 4}] returns data between the 2nd minute and the 4th hour.
        e.g. [None, {'hours': 4}] returns data from the beginning of file to the 4th hour.
        Note: the reference timestamp becomes the last row if timedeltaspan_ref == 'back',
            and the first entry in tuple is always the smaller one (closer to reference).
    :param mode: str, see read_ts
    :param timedeltaspan_ref: str, one of {'front', 'back'}, whether to take timedelta from
        the first or last row
    :param usecols: tuple of ints or str, columns index or name to read, see np.loadtxt documentation
    :param dtypes: dict, k:v = column_name:numpy_dtype, any entries supplied here will
        override the default dtypes in global_param.STD_DTYPE
    :param return_end_byte: bool, whether to return the last byte location read.
        This shouldn't be used
    :return: a tuple of
        - dictionary of numpy arrays, i.e. custom dataframe
        - err: 0 - no error, 1 - empty csv, 2 - empty csv with only header
            3 - has data, but not enough to fulfill timedeltaspan specified
        - byte location where reading ends
    """
    err = 0
    _dtypes = copy.deepcopy(STD_DTYPE)
    _dtypes.update(dtypes) if dtypes else None
    # Read column headers, check for empty file
    try:
        with open(filepath, "r") as f:
            header = f.readline().strip().split(",")
            if header == [""]:
                return ({}, 1, None) if return_end_byte else ({}, 1)
            first_byte = f.tell()
            firstrow = f.readline().strip().split(",")
            if firstrow == [""]:
                return (
                    ({h: np.zeros(0, dtype=_dtypes[h]) for h in header}, 2, first_byte)
                    if return_end_byte
                    else ({h: np.zeros(0, dtype=_dtypes[h]) for h in header}, 2)
                )
            last_byte = os.path.getsize(filepath)
    except:
        traceback.print_exc()
        return ({}, 1, None) if return_end_byte else ({}, 1)

    # Configure start/end of read/write (partial read/partial update)
    seek_byte, end_byte, skiprows, maxrows, start_time, end_time = (
        0,
        None,
        1,
        None,
        None,
        None,
    )
    if tuple(bytespan) != (None, None):
        seek_byte, skiprows = (0, 1) if bytespan[0] is None else (bytespan[0], 0)
        end_byte = bytespan[1]
    elif any(datetimespan):
        if datetimespan[0]:
            if mode == "strict":
                start_time = parser.parse(datetimespan[0])
            seek_byte = binary_search_timestamp(
                filepath, str(datetimespan[0]), find_closest=True
            )
            skiprows = 0
        if datetimespan[1]:
            if mode == "strict":
                end_time = parser.parse(datetimespan[1])
            end_byte = binary_search_timestamp(
                filepath, str(datetimespan[1]), find_closest=True
            )
    elif any(timedeltaspan):
        if timedeltaspan_ref == "front":
            dt_ref = parser.parse(firstrow[header.index("t")])
            if timedeltaspan[0]:
                start_time = dt_ref + timedelta_(timedeltaspan[0], mode="py")
                seek_byte = binary_search_timestamp(
                    filepath, strftime_(start_time), find_closest=True
                )
                skiprows = 0
            if timedeltaspan[1]:
                end_time = dt_ref + timedelta_(timedeltaspan[1], mode="py")
                end_byte = binary_search_timestamp(
                    filepath, strftime_(end_time), find_closest=True
                )
        else:
            dt_ref = parser.parse(
                read_csv_last_row_text(filepath).split(",")[header.index("t")]
            )
            if timedeltaspan[1]:
                start_time = dt_ref - timedelta_(timedeltaspan[1], mode="py")
                seek_byte = binary_search_timestamp(
                    filepath, strftime_(start_time), find_closest=True
                )
                skiprows = 0
            if timedeltaspan[0]:
                end_time = dt_ref - timedelta_(timedeltaspan[0], mode="py")
                end_byte = binary_search_timestamp(
                    filepath, strftime_(end_time), find_closest=True
                )
    # If using end_byte, get the byte location of the end of this row
    end_byte = (
        get_next_datapoint_info_safe(end_byte, filepath)[0] if end_byte else end_byte
    )
    if (maxrows == 0) or (not (end_byte is None) and (seek_byte >= end_byte)):
        df_empty = {
            h: np.zeros(0, dtype=_dtypes[h] if h in _dtypes else float) for h in header
        }
        return (df_empty, 2, seek_byte) if return_end_byte else (df_empty, 2)
    assert (bool(skiprows) != bool(seek_byte)) and (not (maxrows and end_byte)), (
        skiprows,
        seek_byte,
        maxrows,
        end_byte,
    )

    # If we are specifying the end byte location
    try:
        temp_file = None
        if (not (end_byte is None)) and (end_byte < last_byte):
            mkdir(os.path.expanduser("~/tmp"))
            temp_file = os.path.expanduser("~/tmp/") + os.path.basename(filepath)
            shutil.copyfile(filepath, temp_file)
            with open(temp_file, "a") as f:
                f.truncate(end_byte)
            filepath = temp_file

        # Read csv
        if usecols and (type(usecols[0]) is str):
            usecols = sorted(
                [header.index(element) for element in usecols if element in header]
            )
            header = [header[c] for c in usecols]
        with open(filepath, "r") as f:
            f.seek(seek_byte)
            # Performance improvement: np.loadtxt is 3x faster, but cannot handle empty fields
            structured_arr = None
            __dtypes = copy.deepcopy(_dtypes)
            __dtypes.update({"t": "<U35"})
            if (not ("" in firstrow)) and (not usecols):
                try:
                    structured_arr = np.loadtxt(
                        f,
                        delimiter=",",
                        dtype=format_dtypes(
                            header, override=__dtypes, default_dtype=default_dtype
                        ),
                        skiprows=skiprows,
                        max_rows=maxrows,
                        usecols=usecols,
                    )
                    header_lookup = {h: h for h in header}
                except:
                    # traceback.print_exc()     # debug
                    pass
            if structured_arr is None:
                f.seek(seek_byte)
                structured_arr = np.genfromtxt(
                    f,
                    delimiter=",",
                    dtype=format_dtypes(
                        header, override=__dtypes, default_dtype=default_dtype
                    ),
                    skip_header=skiprows,
                    max_rows=maxrows,
                    usecols=usecols,
                    filling_values=format_fill_values(header, override={"t": "NaT"}),
                )
                # np.genfromtxt replaced space ' ' in column headers with underscores '_'
                header_lookup = {h.replace(" ", "_"): h for h in header}
            if not structured_arr.size:
                return (
                    ({h: np.zeros(0, dtype=_dtypes[h]) for h in header}, 2, seek_byte)
                    if return_end_byte
                    else ({h: np.zeros(0, dtype=_dtypes[h]) for h in header}, 2)
                )

        # Convert numpy structured array to dictionary
        df = {
            header_lookup[k]: structured_arr[k].reshape(-1)
            for k in structured_arr.dtype.names
        }
        # Convert column 't' from bytes array to string array, then
        # convert timezone-aware time to utc time, and a separate timezone column
        if "t" in df:
            df["t"] = df["t"].astype(str)
            res = datetime_str2utc(df["t"])
            if type(res) is tuple:
                df["t"], df["timezone"] = res
            else:
                df["t"] = res

        # Clean up temp file
        if temp_file:
            os.remove(temp_file)

        # Check for incomplete data error
        if maxrows and (maxrows > len(df)):
            err = 3

        if (mode == "strict") and len(df["t"]):
            lo_ind = (
                np_searchsorted(
                    df["t"], 
                    np.datetime64(start_time), 
                    mode="nearest_after",
                    inclusive=True,
                )
                if start_time
                else 0
            )
            hi_ind = (
                np_searchsorted(
                    df["t"],
                    np.datetime64(end_time),
                    mode="nearest_before",
                    inclusive=False,
                    safe_clip=False,
                ) + 1
                if end_time
                else len(df["t"])
            )
            df = slice_df(df, lo_ind, hi_ind)
            if lo_ind == hi_ind:
                err = 3

        end_byte = os.path.getsize(filepath) if end_byte is None else end_byte

        if output_type == "pd":
            df = pd.DataFrame.from_dict(df)

        return (df, err, end_byte) if return_end_byte else (df, err)
    except Exception as e:
        print("File loading failed for:", filepath)
        if temp_file:
            os.remove(temp_file)
        raise RuntimeError(str(e)[:1000] + '...')


def read_parquet_ts_singlefile(
    filepath,
    datetimespan=(None, None),
    timedeltaspan=(None, None),
    mode="nearest",
    timedeltaspan_ref="front",
    usecols=None,
    dtypes=dict(),
    default_dtype="float",
    output_type="np",
):
    """
    :param filepath: string or path
    :param datetimespan: tuple of two str or numpy.datetime64, start and/or end time.
        Both start and end times are inclusive, i.e. the nearest row is included in returned df.
    :param timedeltaspan: tuple of two dictionaries, specifying the start and/or end.
        Both start and end times are inclusive, i.e. the nearest row is included in returned df.
        e.g. [{'hours': 4, 'minutes': 2}, None] returns data after 4 hours 2 minutes from the first row
        e.g. [{'minutes': 2}, {'hours': 4}] returns data between the 2nd minute and the 4th hour.
        e.g. [None, {'hours': 4}] returns data from the beginning of file to the 4th hour.
        Or tuple of two datetime.
        Note: the reference timestamp becomes the last row if timedeltaspan_ref == 'back',
            and the first entry in tuple is always the smaller one (closer to reference).
    :param timedeltaspan_ref: str, one of {'front', 'back'}, whether to take timedelta from
        the first or last row
    :param usecols: tuple of ints, columns to read, see np.loadtxt documentation
    :param dtypes: dict, k:v = column_name:numpy_dtype, any entries supplied here will
        override the default dtypes in global_param.STD_DTYPE
    :param output_type: str, one of {'np', 'pd'}
        - 'np': dictionary of numpy.ndarray
        - 'pd': pandas.DataFrame
    :param mode: str, see read_ts
    :return: a tuple of
        - dictionary of numpy arrays, i.e. custom dataframe
        - err: 0 - no error, 1 - empty/no file, 2 - empty file with only header
            3 - has data, but not enough to fulfill timedeltaspan specified
    """
    if not os.path.exists(filepath):
        return {} if output_type == "np" else pd.DataFrame({}), 1

    schema = pq.read_schema(filepath, memory_map=True)
    usecols = [col for col in usecols if col in schema.names] if usecols else usecols
    df = pd.read_parquet(filepath, columns=usecols)

    if not len(df.columns):
        return {} if output_type == "np" else pd.DataFrame({}), 1
    L = len(df)
    # Convert dataframe type
    if output_type == "np":
        df = pd_df2np_df(df)
    # Convert data type for each column
    df = df_astype(df, dtypes)
    if not L:
        return df, 2

    # Return entire data
    if not (any(datetimespan) or any(timedeltaspan)):
        return df, 0
    # Select subset of data
    lo, hi, lo_t, hi_t = 0, None, None, None
    if any(datetimespan):
        if datetimespan[0]:
            lo_t = (
                strptime_np(datetimespan[0])
                if type(datetimespan[0]) is str
                else np.datetime64(datetimespan[0]).astype(df["t"].dtype)
            )
            lo = np_searchsorted(
                df["t"],
                lo_t,
                mode="nearest" if mode == "nearest" else "nearest_after",
                inclusive=True,
            )
        if datetimespan[1]:
            hi_t = (
                strptime_np(datetimespan[1])
                if type(datetimespan[1]) is str
                else np.datetime64(datetimespan[1]).astype(df["t"].dtype)
            )
            hi = (
                np_searchsorted(
                    df["t"],
                    hi_t,
                    mode="nearest" if mode == "nearest" else "nearest_before",
                    inclusive=False,
                    safe_clip=False,
                )
                + 1
            )
    elif any(timedeltaspan):
        if timedeltaspan_ref == "front":
            if timedeltaspan[0]:
                lo_t = df["t"][0] + timedelta_(timedeltaspan[0], mode="np")
                lo = np_searchsorted(
                    df["t"],
                    lo_t,
                    mode="nearest" if mode == "nearest" else "nearest_after",
                    inclusive=True,
                )
            if timedeltaspan[1]:
                hi_t = df["t"][0] + timedelta_(timedeltaspan[1], mode="np")
                hi = (
                    np_searchsorted(
                        df["t"],
                        hi_t,
                        mode="nearest" if mode == "nearest" else "nearest_before",
                        inclusive=False,
                        safe_clip=False,
                    )
                    + 1
                )
        elif timedeltaspan_ref == "back":
            if timedeltaspan[1]:
                lo_t = df["t"][-1] - timedelta_(timedeltaspan[1], mode="np")
                lo = np_searchsorted(
                    df["t"],
                    lo_t,
                    mode="nearest" if mode == "nearest" else "nearest_after",
                    inclusive=False,
                )
            if timedeltaspan[0]:
                hi_t = df["t"][-1] - timedelta_(timedeltaspan[0], mode="np")
                hi = (
                    np_searchsorted(
                        df["t"],
                        hi_t,
                        mode="nearest" if mode == "nearest" else "nearest_before",
                        inclusive=True,
                        safe_clip=False,
                    )
                    + 1
                )
        else:
            raise ValueError(
                f"Invalid argument {timedeltaspan_ref} for timedeltaspan_ref"
            )
    err = (
        3
        if (lo_t and lo == 0 and df["t"][lo] > lo_t)
        or (hi_t and hi == L and df["t"][hi - 1] < hi_t)
        else 0
    )
    return slice_df(df, lo, hi, use_pandas=(output_type == "pd")), err


def read_ts_last_row(filepath, usecols=None):
    filepath = os.path.splitext(filepath)[0]
    # Special case: singlefile
    if os.path.exists(filepath + ".parquet"):
        return read_parquet_last_row(filepath, usecols=usecols)
    elif os.path.exists(filepath + ".csv"):
        return read_csv_last_row(filepath, usecols=usecols)

    # Special case: empty
    if os.path.exists(os.path.join(filepath, "empty.parquet")) or os.path.exists(
        os.path.join(filepath, "empty.csv")
    ):
        return None
    if (not os.path.exists(filepath)) or (len(os.listdir(filepath)) == 0):
        return None

    # Scan all existing monthly files
    all_files = sorted(list(set([f for f in os.listdir(filepath) if f[0] != "."])))
    all_months = np.array([os.path.splitext(f)[0] for f in all_files], "datetime64[M]")
    ext = os.path.splitext(all_files[-1])[-1]
    if ext == ".csv":
        return read_csv_last_row(os.path.join(filepath, all_files[-1]), usecols=usecols)
    elif ext == ".parquet":
        return read_parquet_last_row(
            os.path.join(filepath, all_files[-1]), usecols=usecols
        )
    else:
        raise RuntimeError(
            f"Unknown extension for file {os.path.join(filepath, all_files[-1])}"
        )


def read_csv_last_row_text(filepath):
    byte_loc = os.path.getsize(filepath)
    with open(filepath, "r") as f:
        f.seek(byte_loc)
        content = f.read().rstrip()
        while (not content) or (content[0] != "\n"):
            byte_loc -= 1
            f.seek(byte_loc)
            content = f.read().rstrip()
    return content.strip()


def read_csv_last_row(filepath, usecols=None):
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")
    last_row = read_csv_last_row_text(filepath)
    dtypes = format_dtypes(header, override={"t": "<U35", "t_zero_crossing": "<U35"})
    if usecols is not None:
        dtypes_ = {"names": [], "formats": []}
        for i, name in enumerate(dtypes["names"]):
            if name in usecols:
                dtypes_["names"].append(dtypes["names"][i])
                dtypes_["formats"].append(dtypes["formats"][i])
        usecols = [header.index(c) for c in usecols]
    else:
        dtypes_ = dtypes
    structured_arr = np.loadtxt(
        StringIO(last_row), delimiter=",", dtype=dtypes_, usecols=usecols
    )
    df = {k: structured_arr[k][()] for k in structured_arr.dtype.names}
    # Format t column
    res = datetime_str2utc(str(df["t"]))
    if type(res) is tuple:
        df["t"], df["timezone"] = res
    else:
        df["t"] = res
    if "t_zero_crossing" in df:
        df["t_zero_crossing"] = datetime_str2utc(str(df["t_zero_crossing"]))

    return df


def read_parquet_last_row(filepath, usecols=None):
    """Can handle monthly chunks or single file."""
    filepath = os.path.splitext(filepath)[0]
    # Single file without monthly chunks
    if os.path.exists(filepath + ".parquet"):
        df, err = read_ts(filepath + ".parquet", usecols=usecols)
        if err in (1, 2):
            return None
        return {k: arr[-1] for k, arr in df.items()}
    # Empty file
    if os.path.exists(os.path.join(filepath, "empty.parquet")):
        return None
    if len(os.listdir(filepath)) == 0:
        return None
    # Monthly chunks
    all_months = np.array(
        [os.path.splitext(f)[0] for f in os.listdir(filepath) if f[0] != "."],
        "datetime64[M]",
    )
    df, err = read_ts(
        os.path.join(filepath, str(np.sort(all_months)[-1]) + ".parquet"),
        usecols=usecols,
    )
    if err in (1, 2):
        return None
    return {k: arr[-1] for k, arr in df.items()}


def read_ts_dict(
    data_dict,
    datetimespan=(None, None),
    resample_max_interval=4,
    period_len=None,
    mode="union",
    incldue_empty=False,
    round_to=None,
):
    """
    Read timeseries data from a collection of csv or parquet files. 
    Align data to the same time column.
    :param data_dict: dict, keys: name, values: path to file or custom dataframe
    :param datetimespan: see read_ts
    :param resample_max_interval: int or dict, see resample function
    :param period_len: dict, unit --> amount, period length between data samples
        (i.e. inverse of sampling frequency) e.g. {'minutes': 15}
    :param mode: str, one of {'union', 'intersection'}.
        In 'union', the union of all time intervals are taken, and the final
            output timeseries is from the earliest known time to the latest seen time.
        In 'intersection', the intersection of all time intervals are taken,
            and the final output timeseries is the smallest interval of all.
    :param incldue_empty: bool, if True, includes empty input csv files
    :return: a tuple of
        - time_column, numpy array
        - dict, name -> custom dataframe (see align_timeseries)
    """
    if not data_dict:
        return None, {}
    # Read files one by one
    df_dict = {}
    empty_df = []
    headers = set()
    start, end = None, None
    for k in data_dict:
        # If data is already in memory, no need to load from disk
        if type(data_dict[k]) is dict:
            df, err = data_dict[k], 0
        # Read from data file
        else:
            try:
                df, err = read_ts(data_dict[k], datetimespan=datetimespan)
            except:
                traceback.print_exc()
                raise RuntimeError(f"read_ts failed on: {data_dict[k]}")
        if not (err in (1, 2)):
            df_dict[k] = df
            headers = headers.union(set(df.keys()))
            # Find overlapping timestamps
            if mode == "intersection":
                if (start is None) or (df["t"][0] > start):
                    start = df["t"][0]
                if (end is None) or (df["t"][-1] < end):
                    end = df["t"][-1]
            elif mode == "union":
                if (start is None) or (df["t"][0] < start):
                    start = df["t"][0]
                if (end is None) or (df["t"][-1] > end):
                    end = df["t"][-1]
        else:
            empty_df.append(k)
    if (start is None) or (end is None) or (start >= end):
        print(f"No data found in the specified time range for rule {mode}. {start} to {end}")
        return None, {}
    # Use datetimespan (truncated to file's time range) as start/end if specified.
    if datetimespan[0]:
        datetimespan0 = np.datetime64(datetimespan[0])
        start = datetimespan0 if datetimespan0 > start else start
    if datetimespan[1]:
        datetimespan1 = np.datetime64(datetimespan[1])
        end = datetimespan1 if datetimespan1 < end else end

    # Resample to the same timestamps
    period_len = timedelta_(period_len, mode="np") if period_len else None
    time_column, df_dict = align_timeseries(
        df_dict,
        start,
        end,
        period_len=period_len,
        round_to=round_to,
        resample_max_interval=resample_max_interval,
    )

    # Include the faulty meters (create empty array)
    if incldue_empty:
        headers.add("err")
        for k in empty_df:
            df_dict[k] = {
                h: np.full(time_column.shape, STD_VALUE[h], dtype=STD_DTYPE[h])
                for h in headers
            }
    return time_column, df_dict


def write_ts(df, folderpath, mode="new"):
    """Write monthly chunks"""
    if os.path.splitext(folderpath)[-1] == ".csv":
        return write_csv_ts(**locals())
    else:
        return write_parquet_ts(**locals())


def write_csv_ts(df, folderpath, mode="new"):
    """Write data to csv files in monthly chunks"""
    folderpath = os.path.splitext(folderpath)[0]
    mkdir(folderpath)
    if df and df_len(df):
        i0 = 0
        while True:
            t0 = df["t"][i0].astype("datetime64[M]")
            t1 = t0 + np.timedelta64(1, "M")
            i1 = np.searchsorted(df["t"][i0:], t1) + i0
            write_csv_ts_singlefile(
                slice_df(df, i0, i1),
                os.path.join(folderpath, str(t0) + ".csv"),
                mode=mode,
            )
            i0 = i1
            if i1 == len(df["t"]):
                break


def write_csv_ts_singlefile(df, filepath, mode="new", truncate_byte=None):
    """
    Saves our custom dataframe to csv file.
    There are three modes of operation
        - Save to a new file (overwriting existing file)
        - Append rows to an existing file
        - Replace rows in an existing file (where timestamp overlaps) (not finished_
    Assumptions:
        - Timestamp is sorted and in chronological order.
    :param df: pd or custom dataframe
    :param filepath: str, path to output csv
    :param mode: str, one of {'append', 'new'}
    :param truncate_byte: int, byte location. Active in mode='append'. If specified, the output file will be
        truncated to this byte location, otherwise the output file will be
        truncated to the first timestamp in df.
    :return: None
    """
    def find_byte(filepath, t_str, rule='exact'):
        """
        Find the byte corresponding to the row closest to timestamp t_str.
        :param rule: str, one of {'exact', 'nearest', 'before', 'after'}
        """
        # Exact search
        if rule == 'exact':
            b = binary_search_timestamp(filepath, t_str, find_closest=False)
        # Nearest search
        if (rule in ('nearest', operator.lt, operator.le, operator.gt, operator.ge)) or (b is None):
            b, t_nearest = binary_search_timestamp(filepath, t_str, find_closest=True, return_time=True)
            # We want the returned row to be before t_str
            if (rule in (operator.lt, operator.le)) and compare_timestamps(t_str, rule, t_nearest):
                b = get_previous_datapoint_info(b, filepath, suppress_warning=True, include_timestamp=False)
            # We want the returned row to be after t_str
            if (rule in (operator.gt, operator.ge)) and compare_timestamps(t_str, rule, t_nearest):
                b, eof = get_next_datapoint_info_safe(b, filepath)
        return b
    header = []
    if ((isinstance(df, pd.DataFrame) and not df.empty) or df) and len(df.keys()):
        header = sorted(
            df.keys(),
            key=lambda x: COLUMNS_ORDER.index(x) if x in COLUMNS_ORDER else 10000,
        )
    mkdir(os.path.dirname(filepath))
    # Handle empty df
    if df_empty(df):
        if not os.path.exists(filepath):
            with open(filepath, "w+") as f:
                if header:
                        f.write(",".join(header) + "\n")
        return
    # Convert dictionary to numpy structured array
    dtype = {
        "names": header,
        "formats": [str(df[h].dtype) for h in header],
    }
    structured_array = np.empty(df_len(df), dtype=dtype)
    for k in df:
        structured_array[k] = df[k]
    # Save to a new file
    if (mode == "new") or (not os.path.exists(filepath)):
        with open(filepath, "w+") as f:
            f.write(",".join(header) + "\n")
            np.savetxt(f, structured_array, fmt="%s", delimiter=",")
    elif mode == "append":
        # Truncate output file to appropriate timestamp and save
        if not truncate_byte:
            truncate_byte = find_byte(filepath, str(df["t"][0]), rule=operator.ge)
        with open(filepath, "a") as f:
            f.truncate(truncate_byte)
            np.savetxt(f, structured_array, fmt="%s", delimiter=",")
    elif mode == "append_blind":
        with open(filepath, "a") as f:
            np.savetxt(f, structured_array, fmt="%s", delimiter=",")
    elif mode == "insert":
        b0 = find_byte(filepath, str(df["t"][0]), rule=operator.gt)
        b1 = find_byte(filepath, str(df["t"][-1]), rule=operator.ge)
        with open(filepath, "r") as f:
            f.seek(b1)
            content_after = f.read()
        with open(filepath, 'a') as f:
            f.truncate(b0)
            np.savetxt(f, structured_array, fmt="%s", delimiter=",")
            f.write(content_after)
    else:
        raise ValueError(f"Unknown argument 'mode': {mode}")


def write_parquet_ts_singlefile(df, filepath, mode="new", compression="snappy"):
    """
    Write timeseries
    :param df: pandas.DataFrame or dict of numpy arrays (custom dataframe)
    :param filepath: str, path to output csv
    :param mode: str, one of {'new', 'append', 'append_blind', 'insert'}
    :param compression: str, see parquet documentation for avaiable codecs
    :return: None
    """
    if not (mode in ("new", "append", "append_blind", "insert")):
        raise ValueError(f"mode {mode} is invalid")
    if type(df) is pd.DataFrame:
        if mode == "new" or (not os.path.exists(filepath)):
            df.to_parquet(filepath, compression=compression)
        else:
            raise ValueError(f"mode {mode} for pandas.DataFrame is not implemented")
    elif (df is None) or (df == {}):
        pd.DataFrame({}).to_parquet(filepath, compression=compression)
    elif type(df) is dict:
        if mode != "new" and os.path.exists(filepath):
            df_existing, err = read_parquet_ts_singlefile(filepath)
            if err != 1:
                assert (
                    df.keys() == df_existing.keys()
                ), f"Mismatching keys in existing and incoming df: {df_existing.keys()} vs. {df.keys()}"
                if mode == "append_blind":
                    assert (
                        df_existing["t"][-1] < df["t"][0]
                    ), f"Time is not chronological\n{df['t']}\n{df_existing['t']}"
                    df = {k: np.append(arr, df[k]) for k, arr in df_existing.items()}
                elif mode == "insert":
                    df = insert_ts(df_existing, df, remove_duplicate=True)
                elif mode == "append":
                    lo = np_searchsorted(
                        df_existing["t"],
                        df["t"][0],
                        mode="nearest_after",
                        inclusive=True,
                    )
                    hi = np_searchsorted(
                        df_existing["t"],
                        df["t"][-1],
                        mode="nearest_after",
                        inclusive=False,
                    )
                    df_existing = {
                        k: np.delete(arr, np.s_[lo:hi])
                        for k, arr in df_existing.items()
                    }
                    df = {
                        k: np.insert(arr, lo, df[k]) for k, arr in df_existing.items()
                    }
        pq.write_table(pa.table(df), filepath, compression=compression)
    else:
        raise TypeError(f"Input df has unsupported type: {type(df)}")


def write_parquet_ts(df, folderpath, mode="new", compression="snappy"):
    """
    In the new monthly file format:
        - Empty file (err == 1): empty folder, but folder exists
        - Empty file with header only (err == 2): folder with one file empty.parquet, which contains header
        - Has data (err == 0 or 3): folder with one or more parquet files, <year>_<month>.parquet
    :param df: pandas.DataFrame or dict of numpy arrays (custom dataframe)
    :param folderpath: str, path to output csv
    :param mode: str, one of {'new', 'append', 'append_blind', 'insert'}
    :param compression: str, see parquet documentation for avaiable codecs
    :return: None
    """
    folderpath = os.path.splitext(folderpath)[0]
    mkdir(folderpath)
    if not df:
        pass
    elif df_len(df) == 0:
        write_parquet_ts_singlefile(df, os.path.join(folderpath, "empty.parquet"))
    else:
        i0 = 0
        while True:
            t0 = df["t"][i0].astype("datetime64[M]")
            t1 = t0 + np.timedelta64(1, "M")
            i1 = np.searchsorted(df["t"][i0:], t1) + i0
            write_parquet_ts_singlefile(
                slice_df(df, i0, i1),
                os.path.join(folderpath, str(t0) + ".parquet"),
                mode=mode,
                compression=compression,
            )
            i0 = i1
            if i1 == len(df["t"]):
                break


def csv2parquet(
    indir,
    outdir=None,
    recursive=True,
    pop_timezone=True,
    delete_input=False,
    copy_noncsv=True,
    ignore=[],
):
    """
    :param indir: str, path to input directory
    :param outdir: str, path to output directionary. If None, then outputs to indir.
    :param recursive: bool, whether to search subdirectories as well.
    :param pop_timezone: bool, whether to keep utc time without timezone info
    :param ignore: list of str, list of file name regex patterns to ignore
    :return: None
    """

    def convert(in_file, out_file):
        try:
            df, err = read_csv_ts_singlefile(in_file)
        except Exception as e:
            print(e)
            print(f"Skipping: {in_file}")
            return 1
        if ("timezone" in df) and pop_timezone:
            df.pop("timezone")
        write_parquet_ts_singlefile(df, out_file)
        return 0

    assert (
        type(ignore) is list
    ), f"Input argument ignore must be a list of regex patterns: {ignore}"
    if recursive:
        for root, _, files in os.walk(indir):
            outdir_ = root.replace(indir, outdir) if outdir else root
            mkdir(outdir_)
            for f in files:
                if any([re.search(p, f) for p in ignore]):
                    continue
                if f[-4:] == ".csv":
                    err = convert(
                        os.path.join(root, f),
                        os.path.join(outdir_, f[:-4] + ".parquet"),
                    )
                    if delete_input and (not err):
                        os.remove(os.path.join(root, f))
                elif copy_noncsv and (outdir != indir):
                    shutil.copyfile(os.path.join(root, f), os.path.join(outdir_, f))
    else:
        outdir = outdir or indir
        for f in os.listdir(indir):
            if any([re.search(p, f) for p in ignore]):
                continue
            if f[-4:] == ".csv":
                err = convert(
                    os.path.join(indir, f), os.path.join(outdir, f[:-4] + ".parquet")
                )
                if delete_input and (not err):
                    os.remove(os.path.join(indir, f))
            elif copy_noncsv and (outdir != indir):
                shutil.copyfile(os.path.join(indir, f), os.path.join(outdir, f))


def parquet2csv(indir, outdir=None, recursive=True):
    """Exact mirror of csv2parquet"""

    def convert(in_file, out_file):
        try:
            df, err = read_ts(in_file)
        except Exception as e:
            print(e)
            print(f"Skipping: {in_file}")
            return 1
        write_csv_ts_singlefile(df, out_file)
        return 0

    if recursive:
        for root, subdirs, files in os.walk(indir):
            print("Now processing:", root)
            outdir_ = root.replace(indir, outdir) if outdir else root
            mkdir(outdir_)
            for f in files:
                if f[-8:] == ".parquet":
                    convert(
                        os.path.join(root, f), os.path.join(outdir_, f[:-8] + ".csv")
                    )
    else:
        outdir = outdir or indir
        for f in os.listdir(indir):
            if f[-8:] == ".parquet":
                convert(os.path.join(indir, f), os.path.join(outdir, f[:-8] + ".csv"))


def format_dtypes(header, override=dict(), default_dtype="float"):
    dtypes = {"names": [], "formats": []}
    for k in header:
        dtypes["names"].append(k)
        # Default to float if header does not exist in STD_DTYPE
        dtypes["formats"].append(STD_DTYPE[k] if k in STD_DTYPE else default_dtype)
    # Handle overrides
    for k in override:
        if k in header:
            dtypes["formats"][dtypes["names"].index(k)] = override[k]
    return dtypes


def format_fill_values(header, override=dict()):
    fill_values = [STD_VALUE[k] if k in STD_VALUE else np.nan for k in header]
    for k in override:
        if k in header:
            fill_values[header.index(k)] = override[k]
    return fill_values


def align_timeseries(
    df_dict,
    start,
    end,
    period_len=None,
    round_to="m",
    resample_max_interval=4,
):
    """
    Converts a dictionary of scalar-valued timeseries to a vector-valued timeseries
    with aligned timestamps.
    :param df_dict: dict of dict of numpy arrays,
        each sub-dictionary contains two numpy arrays under keys 't' and 'v'
        {
            '<timeseries1>': {'t': array, 'v': array, ...},
            '<timeseries2>': {'t': array, 'v': array, ...},
            ...
        }
    :param start: numpy datetime64, start timestamp used for resampling
    :param end: numpy datetime64, end timestamp used for resampling
    :param period_len: numpy timedelta64, period length between data samples
    :param round_to: str, unit in numpy datetime format ('m', 's', etc.)
    :param resample_max_interval: int or dict, see resample function
    :return: a tuple of
        - time_column: numpy array, now the same time array shared across all timeseries.
        - dict of dict of numpy arrays same as input df_dict, with time column removed
    """
    # Determine interval size
    if period_len is None:
        time_column_appended = np.concatenate([df_dict[m]["t"] for m in df_dict])
        period_len, _, err = determine_interval_size(time_column_appended)
        if err:
            return (None, None), None
    # Round to the nearest minute, if specified
    period_len = (
        round_time_np(period_len, round_to)
        if not (type(round_to) is dict)
        else period_len
    )
    start = round_time_np(start, round_to) if round_to else start
    end = round_time_np(end, round_to) if round_to else end
    if not period_len:
        raise ValueError(f"Check round_to argument: {round_to} and {period_len}")

    # Resample each dataframe
    has_timezone, time_column = False, None
    for m in df_dict:
        if "timezone" in df_dict[m]:
            df_dict[m].pop("timezone")
            has_timezone = True
        df_dict[m], _ = resample(
            df_dict[m],
            interval_size=period_len,
            max_interval=resample_max_interval,
            start=start,
            end=end,
            time_new=time_column,
            out_of_bound=0,
        )
        time_column = df_dict[m].pop("t")

    return time_column, df_dict


def remove_err_df(df, use_pandas, reset_indices=True):
    """
    Simply remove any rows with r flag >= 3.

    :param df: input dataframe
    :param reset_indices: boolean, wheter to reset indices on the output dataframe.
    :return: error-free pd dataframe.
    """
    if df_empty(df):
        return df
    new_df = pd.DataFrame() if use_pandas else {}
    valid_mask = ~(df["r"] >= 3)

    valid_mask &= (~(df["v"].isna())) if use_pandas else ~np.isnan(df["v"])
    for key in df:
        new_df[key] = df[key][valid_mask]

    if reset_indices and use_pandas:
        return new_df.reset_index(drop=True)
    else:
        return new_df


def read_npy_ts(folder, keys=None):
    """
    Loads custom dataframe stored as one npy file per column.
    :param folder: str, path to folder to load numpy arrays
    :param keys: list of str, if specified, only load these df keys
    :return df: dict, custom dataframe
    """

    df = {}
    for fname in os.listdir(folder):
        if keys and (fname[:-4] not in keys):
            continue
        df[fname[:-4]] = np.load(os.path.join(folder, fname))
    return df


def write_npy_ts(folder, df, keys=None):
    """
    Saves custom dataframe.
    :param folder: str, path to folder to save numpy arrays
    :param df: dict, custom dataframe
    :param keys: list of str, if specified, only save these df keys
    :return: None
    """
    mkdir(folder)
    for k in df:
        if keys and (k not in keys):
            continue
        np.save(os.path.join(folder, k + ".npy"), df[k])


"""DataFrame (custom or pandas) related functions"""


def pd_df2np_df(pd_df):
    """Converts pandas.DataFrame to custom dataframe"""
    return {k: s.to_numpy() for k, s in pd_df.items()}


def infer_df_type(df):
    if type(df) is pd.DataFrame:
        return "pandas"
    if df == {}:
        return "numpy"
    if type(df[list(df.keys())[0]]) is np.ndarray:
        return "numpy"
    if type(df[list(df.keys())[0]]) is list:
        return "list"
    print(type(df[list(df.keys())[0]]))
    raise ValueError("Unknown dataframe type: ", df)


def iter_df(df):
    """Turn our custom dataframe into a list of dictionaries"""
    return [get_row(df, i) for i in range(df_len(df))]


def get_row(df, i):
    """Get a row in our custom dataframe as a dictionary"""
    return {k: df[k][i] for k in df}


def df_len(df):
    if type(df) is pd.DataFrame:
        return len(df.index)
    else:
        return 0 if not df else len(df[list(df.keys())[0]])


def slice_df(df, lo=0, hi=None, use_pandas=False):
    if (lo == 0) and (hi is None):
        return df
    if use_pandas:
        hi = df_len(df) if hi is None else hi
        return df[lo:hi].reset_index(drop=True)
    else:
        new_slice = {}
        for k in df:
            if lo and hi:
                new_slice[k] = df[k][lo:hi]
            elif lo:
                new_slice[k] = df[k][lo:]
            elif hi:
                new_slice[k] = df[k][:hi]
        return new_slice


def index_select_df(df, inds):
    return {k: np.take(arr, inds) for k, arr in df.items()}


def concatenate_df(df_list, axis=0, output_type=None):
    if (output_type == "np") or (type(df_list[0]) is dict):
        return {
            k: np.concatenate([df[k] for df in df_list], axis=axis) for k in df_list[0]
        }
    else:
        return pd.concat(df_list, ignore_index=True)


def mask_df(df, mask, use_pandas):
    if use_pandas:
        return df[mask].reset_index(drop=True)
    else:
        new_df = {}
        for k in df:
            new_df[k] = df[k][mask]
        return new_df


def drop(df, lo, hi, use_pandas):
    """Mimics pandas.DataFrame.drop method"""
    if use_pandas:
        return df.drop(list(range(lo, hi))).reset_index(drop=True)
    else:
        for k in df:
            # df[k] = np.concatenate((df[k][:lo], df[k][hi:]))
            df[k] = np.delete(df[k], list(range(lo, hi)))
        return df


def isempty(df, use_pandas):
    if df is None:
        return True
    if use_pandas and df.empty:
        return True
    if df == {}:
        return True
    for k in df.keys():
        if len(df[k]) > 0:
            return False
    return True


def df_empty(df):
    return not ((isinstance(df, pd.DataFrame) and not df.empty) or df) or (
        not len(df[list(df.keys())[0]])
    )


def df_equals(dfa, dfb):
    none = [(dfa is None), (dfb is None)]
    if all(none):
        return True
    elif any(none):
        return False
    if dfa.keys() != dfb.keys():
        return False
    for k in dfa:
        if "U" in str(dfa[k].dtype):
            if not np.array_equal(dfa[k], dfb[k], equal_nan=False):
                return False
        else:
            if not np.array_equal(dfa[k], dfb[k], equal_nan=True):
                return False
    return True


def df_astype(df, dtypes=dict()):
    dtypes_ = {}
    for k in df:
        if k in dtypes:
            dtypes_[k] = dtypes[k]
        else:
            dtypes_[k] = STD_DTYPE[k] if k in STD_DTYPE else float
    if type(df) is dict:
        return {k: arr.astype(dtypes_[k]) for k, arr in df.items()}
    else:
        return df.astype(dtypes_)


def rename(df, name_lookup):
    """
    Mimic pd.DataFrame.rename to support custom dataframe
    :param df: pandas or custom dataframe
    :param name_lookup: dictionary, key - old name, value - new name
    :return: updated df
    """
    if type(df) is pd.DataFrame:
        df.rename(columns=name_lookup, inplace=True)
        return df
    elif type(df) is dict:
        return update_dict_keys(df, name_lookup)
    else:
        raise ValueError("Unknown in put type: ", df)


def check_df_time_aligns(df_a, df_b):
    assert (df_a["t"].shape == df_b["t"].shape) and (df_a["t"] == df_b["t"]).all(), (
        df_a["t"][0],
        df_a["t"][-1],
        df_b["t"][0],
        df_b["t"][-1],
        df_a["t"].shape,
        df_b["t"].shape,
    )


def np_searchsorted(t_column, t, mode="nearest", inclusive=False, safe_clip=True):
    """
    Search a particular timestamp in df.
    :param t_column: pandas.Series or dict of numpy.ndarray
    :param t: numpy.datetime64 or an iterable of numpy.datetime64
    :param mode: str, one of
        - "nearest": find nearest element
        - "nearest_before": find the closest time before t. This is the same
            as the index to insert in order to maintain order.
        - "nearest_after": find the closest time after t. This is the same
            as the index to insert in order to maintain order, minus 1.
    :param inclusive: bool,
        - True: return the index of an exact match. If the match is a repeated entry,
            nearest_before returns the index of the first repeated entry and nearest_after
            returns the index of the last repeated entry.
        - False: strict inequality when comparing.
            nearest_before returns the index of the element strictly before t.
            nearest_after returns the index of the elmenet strictly after t.
    :return: array of int, indices into df
    """
    if (t_column is None) or (not len(t_column)):
        raise ValueError("Cannot search timestamp in empty column")
    is_iter = hasattr(t, "__iter__")
    t = t if hasattr(t, "__iter__") else [t]
    L = len(t_column)
    if mode == "nearest":
        inds = t_column.searchsorted(t, side="left")
        for i in range(len(inds)):
            if inds[i] > 0 and (
                inds[i] == len(t_column)
                or abs(t[i] - t_column[inds[i] - 1]) < abs(t[i] - t_column[inds[i]])
            ):
                inds[i] -= 1
    elif mode == "nearest_before":
        inds = t_column.searchsorted(t, side="left")
        for i in range(len(inds)):
            if (inds[i] == L) or (t_column[inds[i]] != t[i]) or not inclusive:
                inds[i] = max(0, inds[i] - 1) if safe_clip else inds[i] - 1
    elif mode == "nearest_after":
        inds = t_column.searchsorted(t, side="right")
        for i in range(len(inds)):
            if (inds[i] < L) and (t_column[inds[i]] == t[i]) and not inclusive:
                inds[i] += 1
            elif (inds[i] > 0) and (t_column[inds[i] - 1] == t[i]) and inclusive:
                inds[i] -= 1
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return inds if is_iter else inds[0]


def insert_ts(df_a, df_b, remove_duplicate=True):
    """
    Insert new rows of data into df
    :param df_a: dict of numpy arrays (custom dataframe)
    :param df_b: dict of numpy arrays (custom dataframe)
    :param remove_duplicate: bool, if True, when the same timestamp exists in both dataframes,
        df_b values will overwrite df_a values; otherwise rows will have the same timestamp.
    :return: df, updated
    """
    assert df_a.keys() == df_b.keys(), "df_a and df_b have mismatching key(s)"
    if df_len(df_a) == 0:
        for k in df_a:
            df_a[k] = df_b[k]
        return df_a
    if df_len(df_b) == 0:
        return df_a
    inds = np.searchsorted(df_a["t"], df_b["t"], side="left")
    if remove_duplicate:
        equal_mask = df_a["t"][np.clip(inds, 0, len(df_a["t"]) - 1)] == df_b["t"]
        for k in df_a:
            df_a[k][inds[equal_mask]] = df_b[k][equal_mask]
            df_a[k] = np.insert(df_a[k], inds[~equal_mask], df_b[k][~equal_mask])
        return df_a
    else:
        for k in df_a:
            df_a[k] = np.insert(df_a[k], inds, df_b[k])
        return df_a


def determine_data_column(header):
    for c in ("v", "sum", "mean", "rms"):
        if c in header:
            return c
    raise RuntimeError(f"Cannot determine data column from: {header}")


def serialize_dict(df):
    if type(df) is dict:
        return {k: serialize_dict(df[k]) for k in df}
    elif type(df) in (list, tuple):
        return [serialize_dict(l) for l in df]
    else:
        return df.astype(str).tolist()


"""Time-related functions"""


def now(tz="utc", include_tz=True, mode="py"):
    if tz == "utc":
        dt = (
            datetime.datetime.now(datetime.timezone.utc)
            if include_tz
            else datetime.datetime.utcnow()
        )
    else:
        dt = (
            datetime.datetime.now(pytz.timezone(LOCATION))
            if include_tz
            else datetime.datetime.now(pytz.timezone(LOCATION)).replace(tzinfo=None)
        )
    return dt if mode == "py" else np.datetime64(dt)


def get_timezone(
    at_time=datetime.datetime.utcnow(),
    location=LOCATION,
    input_tz="utc",
    output_type="str",
):
    """
    Given location and time, get timezone of that location at that utc time.
    :param at_time: datetime.datetime, utc time. If the input is in local time, specify input_tz.
    :param location: str, e.g. "America/Los_Angeles"
    :param input_tz: str, one of 'utc', 'local'. This determines how to interpret {at_time}.
    :param output_type: str, one of {'str', 'total_seconds'}
    :return: depending on output_type,
        - 'str': str, e.g. "-07:00"
        - 'total_seconds': float, e.g. 3600.
    """
    pytz_tz = pytz.timezone(location)
    localized_time = (
        pytz.utc.localize(at_time).astimezone(pytz_tz)
        if input_tz == "utc"
        else pytz_tz.localize(at_time).astimezone(pytz_tz)
    )
    tz_offset = localized_time.utcoffset().total_seconds()
    if output_type == "total_seconds":
        return tz_offset
    elif output_type == "str":
        sign = "-" if tz_offset <= 0 else "+"
        tz_offset = abs(tz_offset)
        hours = int(tz_offset / 3600)
        minutes = int((tz_offset - hours * 3600) / 60)
        return sign + "%02d" % hours + ":" + "%02d" % minutes
    else:
        raise ValueError(f"Invalid input: output_type={output_type}")


def utc2timezone(time_column, location=LOCATION, output_type="np"):
    """
    Given a numpy datetime64 array in utc time, return the timezone as 6-character string
    at the location specified.

    :param time_column: list of datetime.datetime or np.ndarray with dtype=datetime64
    :param location: str, e.g. 'America/Los_Angeles'
    :param output_type: str, one of {'np', 'py'}, numpy string array or pythong list of str.
    :return: np.ndarray, same shape as time_column, dtype of str; or list of str.
    """
    if type(time_column) is np.ndarray:
        time_column = time_column.astype("datetime64[us]").tolist()
    timezone_column = [get_timezone(t, location=location) for t in time_column]
    return (
        np.array(timezone_column, dtype=str) if output_type == "np" else timezone_column
    )


def local2utc(time_column, location=LOCATION, output_type="np", dtype=STD_DTYPE["t"]):
    """
    Given a time column in local time (without timezone info), convert to utc.
    This requires looking up time zone offset for each of the entry in time_column
    at the specified location.
    :param time_column: list of datetime.datetime or np.ndarray with dtype=datetime64
    :param location: str, e.g. 'America/Los_Angeles'
    :param output_type: str, one of {'np', 'py'}, numpy string array or pythong list of str.
    :param dtype: str, if output_type == 'np', specifies the output array dtype
    :return: np.ndarray, or list of datetime.datetime, same shape as time_column
    """
    time_column_np = None
    if type(time_column) is np.ndarray:
        time_column_np = time_column
        time_column = time_column.astype(datetime.datetime)
    timezone_offset = [
        get_timezone(
            t, location=location, input_tz="local", output_type="total_seconds"
        )
        for t in time_column
    ]
    if output_type == "np":
        time_column_np = (
            np.array(time_column, dtype=dtype)
            if (time_column_np is None)
            else time_column_np
        )
        return time_column_np - np.array(timezone_offset).astype("timedelta64[s]")
    elif output_type == "py":
        return [
            time_column[i] - datetime.timedelta(seconds=int(timezone_offset[i]))
            for i in range(len(timezone_offset))
        ]
    else:
        raise ValueError(f"Invalid input: output_type={output_type}")


def local2utc_singlestr(t_local, location=LOCATION):
    t_local = parser.parse(t_local)
    timezone_offset = get_timezone(
        t_local, location=location, input_tz="local", output_type="total_seconds"
    )
    t_utc = t_local - datetime.timedelta(seconds=int(timezone_offset))
    return strftime_(t_utc)


def utc2local(t, tz=None, output_type="np"):
    """
    Converts utc time to local time, dropping timezone information
    e.g. 2022-04-30T16:45:00.636 & "-07:00" --> 2022-04-30T23:45:00.636
    :param t: np.datetime64, timezone-unaware time
    :param tz: str, optional
    :returns: np.datetime64
    """
    tz = tz or utc2timezone([t.tolist()])[0]
    sign, hours, minutes = tz[0], tz[1:3], tz[4:6]
    hours, minutes = int(sign + hours), int(sign + minutes)
    timezone_timedelta = (
        np.timedelta64(1, "h") * hours + np.timedelta64(1, "m") * minutes
    )
    t_local = t + timezone_timedelta
    if output_type == "np":
        return t_local
    elif output_type == "py":
        return t_local.tolist()
    else:
        raise ValueError(f"Invalid input: output_type={output_type}")


def datetime_str2utc(time_str):
    """
    Converts a timezone-aware time to utc time.
    :param time_str: str, or numpy str array of timezone-aware time, e.g. "2022-04-30T23:45:00.636-07:00"
    :return:
        - utc, utc time with dtype 'datetime64[ms]'
        - timezone, str in format "-07:00"
    """

    def parse_dt_format(_time_str):
        # If this fails, then we have unknown input datetime string format time_str
        len_unit = {
            True: {25: "s", 29: "ms", 32: "us", 35: "ns"},
            False: {19: "s", 23: "ms", 26: "us", 29: "ns"},
        }
        _has_timezone = (_time_str[-6] in ("-", "+")) and (_time_str[-3] == ":")
        _unit = len_unit[_has_timezone][len(_time_str)]
        return _unit, _has_timezone

    # Handle single string
    if type(time_str) is str:
        unit, has_timezone = parse_dt_format(time_str)
        # Note: If time_str carries time zone, a numpy deprecated warning will be printed.
        out = np.datetime64(time_str, unit)  # This is slow
        return (out, time_str[-6:]) if has_timezone else out
    # Handle np/pandas/list string arrays
    else:
        if not (type(time_str) is np.ndarray):
            time_str = np.array(time_str, dtype=str)
        # If this fails, then we have unknown input datetime string format time_str
        unit, has_timezone = parse_dt_format(time_str[0])
        out = np.array(time_str, dtype=f"datetime64[{unit}]")
        return (
            (
                out,
                slice_np_string_array(time_str, len(time_str[0]) - 6, len(time_str[0])),
            )
            if has_timezone
            else out
        )


def utc2tz_aware(df, pop_timezone=True, return_str=True):
    """
    Converts utc time to time-zone-aware time, given timezone as string.
    Supports both array input and single string.
    e.g. 2022-04-30T16:45:00.636 & "-07:00" --> "2022-04-30T23:45:00.636-07:00"
    :param df: dict of numpy.ndarray, with entries
        't': numpy datetime64 object or a numpy array of datetime64, in utc time
        'timezone': string, e.g. "07:00", or an array of strings
    :param pop_timezone: boolean, whether to pop timezone from dictionary
    :return: df updated, with 't' entry now a string or numpy string array
        of timezone-aware time, i.e. "2022-04-30T23:45:00.636-07:00"
    """
    # An numpy datetime64 array to timezone-aware datetime string
    if (type(df) is dict) and (not df):
        return df
    elif type(df) is pd.DataFrame:
        if not df.empty:
            assert (df["t"].dt.tz is None) or (df["t"].dt.tz is pytz.UTC), df["t"].dt.tz
            df["t"] = df["t"].dt.tz_localize("utc").dt.tz_convert("America/Los_Angeles")
        return df
    elif type(df["t"]) is np.ndarray:
        if df_empty(df):
            if pop_timezone and ("timezone" in df):
                df.pop("timezone")
            return df
        tz_column = df["timezone"] if "timezone" in df else utc2timezone(df["t"])
        assert len(tz_column[0]) == 6, tz_column
        hours = np.array(slice_np_string_array(tz_column, 0, 3), dtype=int)
        sign = slice_np_string_array(tz_column, 0, 1)
        minutes = slice_np_string_array(tz_column, 4, 6)
        minutes = np.array(np.core.defchararray.add(sign, minutes), dtype=int)
        timezone_timedelta = (
            np.timedelta64(1, "h") * hours + np.timedelta64(1, "m") * minutes
        )
        if pop_timezone and ("timezone" in df):
            df.pop("timezone")
        if return_str:
            out_arr = np.datetime_as_string(df["t"] + timezone_timedelta)
            df["t"] = np.core.defchararray.add(out_arr, tz_column)
        else:
            df["t"] = df["t"] + timezone_timedelta
        return df
    # A single UTC datetime string to timezone-aware datetime string
    elif type(df["t"]) is np.datetime64:
        sign, hours, minutes = (
            df["timezone"][0],
            df["timezone"][1:3],
            df["timezone"][4:6],
        )
        hours, minutes = int(sign + hours), int(sign + minutes)
        timezone_timedelta = (
            np.timedelta64(1, "h") * hours + np.timedelta64(1, "m") * minutes
        )
        if return_str:
            return np.datetime_as_string(df["t"] + timezone_timedelta) + df["timezone"]
        else:
            return df["t"] + timezone_timedelta
    else:
        raise ValueError(f"Unsupported input type: {df}")


def np_datetime2py_datetime(t, tz=None):
    """
    Converts numpy datetime64 object to datetime.datetime object
    :param t: numpy.datetime64
    :param tz: str, length 6, e.g. '-07:00'
    :return: datetime.datetime
    """
    dt = t.astype(datetime.datetime)
    if tz:
        t_delta = datetime.datetime.strptime(tz[1:], "%H:%M")
        t += (
            datetime.timedelta(hours=t_delta.hour, minutes=t_delta.minute)
            if tz[0] == "-"
            else -datetime.timedelta(hours=t_delta.hour, minutes=t_delta.minute)
        )
    return dt


def pd_datetime2np_datetime(pd_series, return_timezone=False):
    """Converts pandas datetimeseries to numpy datetime array"""
    np_datetime_arr, np_timezone_arr = [], []
    if (pd_series.dtype == "O") and (type(pd_series[0]) is pd.Timestamp):
        for obj in pd_series:
            minutes = obj.tzinfo._offset.total_seconds() / 60
            sign = "+" if minutes > 0 else "-"
            hours = math.floor(abs(minutes) / 60)
            minutes = abs(minutes) - hours * 60
            np_datetime_arr.append(np.datetime64(obj))
            if return_timezone:
                np_timezone_arr.append(sign + "%02d" % hours + ":" + "%02d" % minutes)
        np_datetime_arr = np.array(np_datetime_arr, dtype="datetime64[ms]")
        if return_timezone:
            np_timezone_arr = np.array(np_timezone_arr)
    else:
        np_datetime_arr = pd_series.dt.tz_convert(None)
        sign = "+" if (pd_series.dt.tz._minutes > 0) else "-"
        hours = math.floor(abs(pd_series.dt.tz._minutes) / 60)
        minutes = abs(pd_series.dt.tz._minutes) - hours * 60
        if return_timezone:
            timezone_str = sign + "%02d" % hours + ":" + "%02d" % minutes
            np_timezone_arr = np.array([timezone_str] * len(pd_series), dtype=str)
        np_datetime_arr = np_datetime_arr.to_numpy(dtype="datetime64[ms]")

    if return_timezone:
        return np_datetime_arr, np_timezone_arr
    else:
        return np_datetime_arr


def timedelta_(unit_amount, mode, unit="us"):
    """
    :param unit_amount: dictionary, e.g. {'days': 2, 'seconds': 5} is 2 days 5 seconds.
        Alternatively, it can also be a datetime.timedelta already.
    :param mode: one of {'py', 'np', 'pd'}
    :param unit: str, pandas or numpy datetime unit
    :return: a timedelta object in the type of mode
    """
    if mode == "py":
        td = (
            datetime.timedelta(**unit_amount)
            if type(unit_amount) is dict
            else unit_amount
        )
        return td
    elif mode == "np":
        if type(unit_amount) is dict:
            td = np.timedelta64(0, "s").astype(f"timedelta64[{unit}]")
            for u, amount in unit_amount.items():
                td += np.timedelta64(amount, PYTHON2NUMPY_TIME_UNIT[u]).astype(
                    f"timedelta64[{unit}]"
                )
        else:
            td = np.timedelta64(unit_amount)
        return td
    elif mode == "pd":
        td = (
            datetime.timedelta(**unit_amount)
            if type(unit_amount) is dict
            else unit_amount
        )
        return pd.Timedelta(td, unit=unit)
    else:
        return ValueError("Unknown mode: ", mode)


def datetime_(unit_amount, mode):
    """
    :param unit_amount: dictionary, e.g. {'year': 1999, 'month': 4, 'day': 2, 'second': 5}
        for the datetime of 1998-04-02T00:00:05.000
    :param mode: one of {'py', 'np', 'pd'}
    :return: a datetime object in the type of mode
    """
    dt = datetime.datetime(**unit_amount)
    if mode == "py":
        return dt
    elif mode == "np":
        return np.array([dt], dtype=STD_DTYPE["t"])[0]
    elif mode == "pd":
        return pd.Timedelta(dt, unit="milliseconds")
    else:
        return ValueError("Unknown mode: ", mode)


def datetime_parse_unit(dtype):
    dtype = str(dtype)
    assert dtype[:11] == "datetime64[", f"Uknown dtype format: {dtype}"
    return dtype[11:13]


def strftime_(t):
    """Converts a python datetime object to string in our custom format (ms precision)"""
    t_str = datetime.datetime.strftime(t, "%Y-%m-%dT%H:%M:%S.%f%z")
    if len(t_str) == 31:
        return t_str[:29] + ":" + t_str[29:]
    elif len(t_str) == 26:
        return t_str[:23]
    elif len(t_str) == 24:
        return t_str[:22] + ":" + t_str[22:]
    elif len(t_str) == 19:
        return t_str
    else:
        raise ValueError(f"Unknown time string format: {t_str}")


def strptime_np(t, unit="us"):
    """Converts str time to numpy.datetime64[unit], the maximum resolution is us."""
    if t is None:
        return None
    else:
        if type(t) is np.datetime64:
            return t
        elif type(t) is str:
            return np.datetime64(parser.parse(t), unit)
        elif type(t) is datetime.datetime:
            return np.datetime64(t, unit)
        else:
            raise TypeError(f"Input t: {t} has type {type(t)}.")


def strptime_dateonly(t_str, str_format="%Y-%m-%d"):
    """
    Get date from time string.
    :param t_str: str, time e.g. '2022-05-01T23:00:00.000-07:00'
    :param str_format: str
    :return: str, date. Note that timezone information is discarded.
        Output is in local time and timezone is ignored.
        To output date in utc time, convert t_str to utc time first.
    """
    return parser.parse(t_str).strftime(str_format)


def find_time(timestamps, target, start_idx):
    """
    Given an iterable of timestamps, find index of target time.
    :param timestamps: an iterable of timestamps
    :param target: np or python datetime object
    :param start_idx: int, start index
    :return: int, index of the closest timestamp to target.
        If target time is out of range, the end index (i.e. 0 or length-1) is returned.
    """
    now = timestamps[start_idx]
    diff = target - now
    curr_idx = start_idx
    if diff > datetime.timedelta(seconds=0):
        while (now < target) and (curr_idx < len(timestamps) - 1):
            curr_idx += 1
            now = timestamps[curr_idx]
    else:
        while (now > target) and (curr_idx > 0):
            curr_idx -= 1
            now = timestamps[curr_idx]
    return curr_idx


def round_time(dt, round_to):
    """
    Round a datetime object to any time in seconds
    :param dt: datetime.datetime object, default now.
    :param round_to: int, number of seconds to round to.
    :returns: datetime.datetime
    """
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def round_time_np(dt, round_to):
    """
    Round a numpy datetime64 to any time unit.
    :param dt: numpy datetime64 object
    :param round_to: str, unit in numpy datetime format ('m', 's', etc.)
    :return: numpy datetime64 object, the original dtype is preserved
    """
    if round_to is None:
        return dt
    elif type(round_to) is str:
        dtype = dt.dtype
        timedelta_or_datetime = str(dtype)[: str(dtype).find("[")]
        assert (
            "timedelta" in timedelta_or_datetime or "datetime" in timedelta_or_datetime
        ), dtype
        # By default, numpy casting between timestamps is equivalent to 'floor'
        base = dt.astype(f"{timedelta_or_datetime}[{round_to}]").astype(dtype)
        one_unit = np.ones(1, dtype=f"timedelta64[{round_to}]").astype(
            str(dtype).replace("datetime", "timedelta")
        )
        return (base + one_unit)[0] if (dt - base) / one_unit >= 0.5 else base
    elif type(round_to) is dict:
        assert len(round_to) == 1
        dtype = dt.dtype
        timedelta_or_datetime = str(dtype)[: str(dtype).find("[")]
        unit = list(round_to.keys())[0]
        unit_idx = PYTHON_TIME_UNITS.index(unit) - 1
        unit_next_up, unit_next_up_np = (
            PYTHON_TIME_UNITS[unit_idx],
            NUMPY_TIME_UNITS[unit_idx],
        )
        interval = timedelta_(round_to, "np")
        next_up_interval = timedelta_({unit_next_up: 1}, "np")
        num_intervals = next_up_interval / interval
        assert (
            int(num_intervals) == num_intervals
        ), f"Only integer number of intervals is supported: {round_to}, {num_intervals}"
        base = dt.astype(f"{timedelta_or_datetime}[{unit_next_up_np}]").astype(dtype)
        candidates = base + np.arange(int(num_intervals) + 1) * interval
        idx = np_searchsorted(candidates, dt, mode="nearest")
        return candidates[idx]
    else:
        raise ValueError(f"Unknown input type: {round_to}")


def determine_interval_size(timestamps, err_tolerance=(0.1, 0.6), use_pandas=False):
    """
    :param timestamps: a vector of timestamps
    :param err_tolerance: a tuple of two floats, for error checking.
        - in range(0,1), range in which we consider time intervals are "regular",
            in fractions of the median
        - in range (0,1) the percentage of samples that must fall in these std
        E.g. (0.1, 0.6) means that at least 60% of the time samples are within
            plus/minus 10% of median time interval.
    :return: a tuple of
        - timedelta, median time interval
        - intervals, vector of time interval sizes
        - int, error code 0 - no error, 1 - too many samples are on irregular
            intervals and time interval cannot be determined.
    """
    # Come on, give me at least 3 data points for each period, set error flag
    if len(timestamps) < 3:
        return None, None, 1

    intervals = difference(timestamps, use_pandas)
    # assert (intervals > 0).all(), "Unsorted timeseries or duplicate timestamp."
    med_time_interval = median_(intervals, use_pandas)
    if not med_time_interval:
        return med_time_interval, intervals, 1

    # Check if the intervals are regular enough, set error flag
    interval_size_deviation = abs_(intervals - med_time_interval, use_pandas)
    if (
        interval_size_deviation <= (med_time_interval * err_tolerance[0])
    ).mean() < err_tolerance[1]:
        return med_time_interval, intervals, 1
    else:
        return med_time_interval, intervals, 0


def resample(
    in_df,
    interval_size=None,
    intervals=None,
    max_interval=None,
    start=None,
    end=None,
    time_new=None,
    out_of_bound=None,
):
    """
    Converts irregular-interval timeseries data into regular interval data.
    This may require both up-sampling (interpolation) and down-sampling.
    :param in_df: pandas dataframe
    :param interval_size: numpy timedelta64
    :param max_interval: dictionary, e.g. {'hours': 1} or float or None.
        If it is float, it should be positive, and means multiples of interval_size.
        If supplied, any missing data interpolated from an interval larger
        than max_interval will carry error flags.
    :param start: datetime object, the start time. If not supplied, we'll
        use the first timestamp in input data.
    :param end: datetime object, the end time. If not supplied, we'll
        use the last timestamp in input data.
    :param time_new: numpy datetime64 array. The output time column if supplied.
    :param out_of_bound: object, if specified, out-of-bound samples will be set to this,
        otherwise the values of first and last entries are used. This applies to
        linear interpolate only and not nearest-neighbor interpolate.
    :param intervals: numpy.ndarray, optional
    :return: a tuple of
        - dict, dataframe with the same keys as original dataframe. For resample error, if the original
            data contains error column, error value will be written to the existing column, otherwise
            an 'err' column (i.e. key) will be created.
        - int, error code
    """
    out_df = {}

    # Determine regular time interval size
    if interval_size is None:
        interval_size, intervals, err = determine_interval_size(in_df["t"])
        if err:
            return out_df, err
    elif max_interval:
        intervals = difference(in_df["t"], False) if intervals is None else intervals

    # Generate regular timestamps
    if time_new is None:
        start = in_df["t"][0] if start is None else start
        end = in_df["t"][len(in_df["t"]) - 1] if end is None else end
        # Alternatively, use math.floor so the last sample is before the last available data point
        num_samples = round((end - start) / interval_size) + 1
        out_df["t"] = np.arange(num_samples) * interval_size + start
    else:
        num_samples = len(time_new)
        out_df["t"] = time_new
    time_orig_as_float = in_df["t"].astype(out_df["t"].dtype).astype(float)
    time_new_as_float = out_df["t"].astype(float)

    # Linear interpolate (out-of-bound values are 0)
    for k in ("v", "sum", "mean"):
        if k in in_df:
            if out_of_bound:
                out_df[k] = np.interp(
                    time_new_as_float,
                    time_orig_as_float,
                    in_df[k],
                    left=out_of_bound,
                    right=out_of_bound,
                )
            else:
                out_df[k] = np.interp(time_new_as_float, time_orig_as_float, in_df[k])
    # Nearest-neighbor interpolate
    indices = None
    for k in [
        "s",
        "r",
        "err",
        "info",
        "timezone",
        "meters down",
        "meters up",
        "EM down",
        "HW down",
        "CHW down",
        "EM up",
        "HW up",
        "CHW up",
    ]:
        if k in in_df:
            if indices is None:
                _, indices = scipy.spatial.KDTree(time_orig_as_float[:, None]).query(
                    time_new_as_float[:, None]
                )
            out_df[k] = np.take(in_df[k], indices)

    # Check if all columns are accounted for
    for k in in_df:
        assert (
            k in out_df
        ), f"Column {k} is not converted due to lack of rule (utils:resample)"

    err = np.zeros(num_samples, dtype=bool)
    # Time samples out of bound
    err |= out_df["t"] < (in_df["t"][0] - interval_size)
    err |= out_df["t"] > (in_df["t"][-1] + interval_size)
    # Intervals too large
    if max_interval:
        period_mask = intervals > (max_interval * interval_size)
        err_intervals = period_mask.nonzero()[0]
        if len(err_intervals) == len(intervals):
            print(f"utils.py:resample All data intervals are too large.\n{max_interval}, {interval_size}\n{intervals}")
        for i in err_intervals:
            sample_mask = (out_df["t"] > in_df["t"][i]) & (
                out_df["t"] < in_df["t"][i + 1]
            )
            err |= sample_mask
    # Populate error mask
    if not ("err" in out_df):
        out_df["err"] = np.ones(num_samples, dtype=STD_DTYPE["err"]) * STD_VALUE["err"]
    out_df["err"][err != 0] = ERR_CODES["resample error"]
    return out_df, 0


def select_sample(df, time_column):
    """
    Given a new time column, find the rows that are closest in time to the new time column.
    :param df: dict, custom dataframe
    :param time_column: numpy.ndarray, new time column
    :return: df, dict, custom dataframe
    """
    left_idx = np.searchsorted(df['t'], time_column, side="left")
    right_idx = np.searchsorted(df['t'], time_column, side="left")
    left_diff = np.abs(df['t'][left_idx] - time_column)
    right_diff = np.abs(df['t'][right_idx] - time_column)
    idx = np.unique(np.where(left_diff < right_diff, left_idx, right_idx))
    return {k: df[k][idx] for k in df}
    

def downsample(
    in_df,
    interval_size,
    datetimespan=None,
    interval_round_to=None,
    start_end_round_to=None,
):
    """
    Down-samples timeseries data into regular interval data using mean.
    The new time column indicates the average value for the succeeding interval.
    :param in_df: dict, custom dataframe
    :param interval_size: numpy timedelta64
    :param datetimespan: tuple of two str, specifying the start (inclusive) and end time.
    :param interval_round_to: str, unit in numpy datetime format ('m', 's', etc.)
    :param start_end_round_to: str, unit in numpy datetime format ('m', 's', etc.)
    :return: dict, custom dataframe
    """
    interval_size_orig, intervals, err = determine_interval_size(in_df["t"])
    interval_size_orig = (
        round_time_np(interval_size_orig, interval_round_to)
        if interval_round_to
        else interval_size_orig
    )
    if err:
        raise RuntimeError(f"determine_interval_size returned with error code: {err}")
    if not (intervals == interval_size).all():
        start = (
            strptime_np(datetimespan[0])
            if datetimespan
            else round_time_np(in_df["t"][0], start_end_round_to)
        )
        end = (
            strptime_np(datetimespan[1])
            if datetimespan
            else round_time_np(in_df["t"][-1], start_end_round_to)
        )
        has_err = "err" in in_df
        in_df = resample(in_df, interval_size=interval_size_orig, start=start, end=end)[
            0
        ]
        if not has_err:
            in_df.pop("err")
    # Take average over each period
    num_samples = int((in_df["t"][-1] - in_df["t"][0]) / interval_size)
    num_samples_interval = interval_size / interval_size_orig
    assert (
        abs(round(num_samples_interval) - num_samples_interval) == 0
    ), f"Interval size {interval_size} must be integer multiples of "
    num_samples_interval = int(num_samples_interval)
    out_df = {"t": np.arange(num_samples) * interval_size + in_df["t"][0]}
    # Downsample for each column
    for k in in_df:
        # Select every k rows
        if k in {
            "s",
            "r",
            "err",
            "info",
            "timezone",
            "meters down",
            "meters up",
            "EM down",
            "HW down",
            "CHW down",
            "EM up",
            "HW up",
            "CHW up",
        }:
            out_df[k] = in_df[k][: num_samples * num_samples_interval][
                ::num_samples_interval
            ]
        elif k == "t":
            continue
        # Take mean
        else:
            out_df[k] = (
                in_df[k][: num_samples * num_samples_interval]
                .reshape(num_samples, num_samples_interval)
                .mean(axis=1)
            )
    return out_df


def union_time_intervals(time_intervals_list):
    """
    :param: time_intervals_list: list of tuple, (start_time, end_time)
        start and end times can be datetime.datetime or numpy.datetime64
    """
    # First sort time intervals by start time
    time_intervals_list = sorted(time_intervals_list, key=lambda it: it[0])
    # Merge overlapping time intervals
    time_intervals_merged = [time_intervals_list[0]]
    for t0, t1 in time_intervals_list[1:]:
        t0_prev, t1_prev = time_intervals_merged[-1]
        if t0 <= t1_prev:
            time_intervals_merged[-1] = (t0_prev, max(t1_prev, t1))
        else:
            time_intervals_merged.append((t0, t1))
    return time_intervals_merged


"""Functions with pandas-numpy dual support & numpy specific helper functions"""


def abs_(arr, use_pandas=None):
    if use_pandas or type(arr) is pd.DataFrame:
        return arr.abs()
    else:
        return np.abs(arr)


def median_(arr, use_pandas=None):
    if use_pandas or type(arr) is pd.DataFrame:
        return arr.median()
    else:
        return np.median(arr)


def concatenate(arr_list, use_pandas):
    return pd.concat(arr_list) if use_pandas else np.concatenate(arr_list)


def derivative(df, use_pandas, timescale):
    return difference(df["v"], use_pandas) / (
        difference(df["t"], use_pandas) / timescale
    )


def argmax(arr, use_pandas):
    return arr.idxmax() if use_pandas else arr.argmax()


def argmin(arr, use_pandas):
    return arr.idxmin() if use_pandas else arr.argmin()


def convolve(arr, kernel, use_pandas):
    if use_pandas:
        return pd.Series(np.convolve(arr.to_numpy(), kernel, mode="same"))
    else:
        return np.convolve(arr, kernel, mode="same")


def smooth(arr, kernel_size=10, mode="same"):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(arr, kernel, mode=mode)


def difference(data, use_pandas=False):
    """This is to help make our code pandas-numpy agnostic."""
    if use_pandas:
        return data[1:].reset_index(drop=True) - data[:-1].reset_index(drop=True)
    else:
        return data[1:] - data[:-1]


def zeros_array(size, dtype, use_pandas):
    zeros = np.zeros(size, dtype=dtype)
    return pd.Series(zeros) if use_pandas else zeros


def arr_is_float(arr):
    return np.vectorize(is_float, otypes=[bool])(arr)


def slice_np_string_array(a, start, end):
    a = np.ascontiguousarray(a, dtype=a.dtype)
    b = a.view((str, 1)).reshape(len(a), -1)[:, start:end]
    return np.array(b).view((str, end - start)).flatten()


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


"""General python helper functions"""


def list_index(l, element, non_existent=sys.maxsize):
    """Python list index with support for index for non-existant values."""
    try:
        return l.index(element)
    except:
        return non_existent


def listdir(dir, recursive=False):
    """
    :param dir: str, path to direcotry
    :param recursive: bool, whether to recurse into all sub-folders and list
    :returns: list of os.path.Path, full path to file includign argument dir
    """
    if os.path.exists(dir):
        if recursive:
            out_files = []
            for root, _, files in os.walk(dir):
                out_files += [os.path.join(root, f) for f in files]
            return out_files
        else:
            return [os.path.join(dir, f) for f in os.listdir(dir)]
    else:
        return []


def decode(code, base=2):
    """
    :param code: int
    :param base: int
    :return: list of int
    """
    out = []
    while code > 0:
        d = math.floor(math.log(code, base))
        out.append(d)
        code -= 2**d
    return sorted(out)


def remove_from_list(l, indices):
    """Remove items from list by a collection of indices"""
    cnt = 0
    for i in sorted(indices):
        l.pop(i - cnt)
        cnt += 1
    return l


def update_dict_keys(d, old_new):
    """
    Given a dictionary, update its key names with another dictionary.
    :param d: dictionary to be updated
    :param old_new: dictionary, where keys are old names, and values are new names
    :return: updated dictionary d
    """
    for o in old_new:
        if o in d:
            d[old_new[o]] = d.pop(o)
    return d


def update_ordered_set(a, b):
    """
    Update an iterable (a) with unique, ordered elements with an iterable (b).
    :param a: list or str, to be updated
    :param b: list or str, to be added to a
    """
    a += [element for element in b if element not in a]
    return a


def locate_by_name(list_of_dict, name):
    for d in list_of_dict:
        if d["name"] == name:
            return d


def is_float(val):
    try:
        float(val)
    except ValueError:
        return False
    else:
        return True


def mkdir(dir):
    """Make directory recursively if directory does not exist."""
    os.makedirs(dir, exist_ok=True)


def rmdir(dir):
    """Remove directory recursively if directory does not exist."""
    if os.path.exists(dir):
        shutil.rmtree(dir)


def delete_output(
    rootdir, folder_suffix, file_suffix="", safe_ext=("png", "jpg", "jpeg", "json")
):
    """
    There are two modes:
        - If out_folder_suffix == "", recursively delete files ending with file_suffix.
        - Otherwise, recursively delete folders ending with out_folder_suffix.
    :param rootdir:
    :param folder_suffix: string, delete folders matching this suffix.
    :param file_suffix: string, delete files matching this suffix.
        If folder_suffix is not "" or None, file_suffix is ignored.
    :param safe_ext: tuple of string, extensions that are safe to delete.
        This is only active when file_suffix and folder_suffix are both empty strings.
    """
    rootdir = rootdir.removesuffix("/")
    if not folder_suffix:
        for root, subdirs, files in os.walk(rootdir):
            for f in files:
                if (
                    len(f.split(".")[-2]) > len(file_suffix)
                    and (f.split(".")[-2][-len(file_suffix) :] == file_suffix)
                ) or ((file_suffix == "") and (f.split(".")[-1] in safe_ext)):
                    os.remove(os.path.join(root, f))
    else:
        if os.path.exists(rootdir + folder_suffix):
            shutil.rmtree(rootdir + folder_suffix)
        for root, subdirs, files in os.walk(rootdir):
            for subdir in subdirs:
                if subdir[-len(folder_suffix) :] == folder_suffix:
                    shutil.rmtree(os.path.join(root, subdir))
    print("Deleted existing output")


def print_if_unique(outstr, strset):
    if not (outstr in strset):
        print(outstr)
        strset.add(outstr)


def make_nested_dict(keys, element):
    d = element
    for k in keys[::-1]:
        d = {k: d}
    return d


def list_remove_if_exists(l, element):
    if element in l:
        l.remove(element)
    return l


def insert_dict(d, keys, element, mode="insert"):
    """
    Insert an element into a multiple nested dictionary of arbitrary nested levels.
    d[keys[1]][keys[2]]...[keys[n]] = element
    :param d: dict, dictionary to insert element into
    :param keys: list of keys from the outermost level to the innermost level
    :param element: final element to insert
    :param mode: str, one of {'insert', 'update'}, if 'update', then element must be a dictionary.
    :return: d, updated dict
    """
    sub_d = d
    for i in range(len(keys)):
        if i == (len(keys) - 1):
            if mode == "update":
                sub_d[keys[i]].update(element)
            else:
                sub_d[keys[i]] = element
        else:
            if (keys[i] not in sub_d) and (type(sub_d) is not list):
                sub_d[keys[i]] = {}
            sub_d = sub_d[keys[i]]
    return d


def insert_dict_list(d, keys, element):
    """
    Same as inser_dict, but the element is part of a list.
    Appends element to the existing list, otherwise create a new list.
    :param d: dict, dictionary to insert element into
    :param keys: list of keys from the outermost level to the innermost level
    :param element: list, or anything, element(s) to insert.
        If a list is provided, the list will be concatenated with the existing list.
    :return: d, updated dict
    """
    element = [element] if not (type(element) is list) else element
    if dict_exists(d, keys):
        return insert_dict(d, keys, dict_get(d, keys) + element)
    else:
        return insert_dict(d, keys, element)


def pop_dict(d, keys):
    if len(keys) == 1:
        d.pop(keys[0])
    else:
        pop_dict(d[keys[0]], keys[1:])
        if not len(d[keys[0]]):
            d.pop(keys[0])


def dict_increment(d, keys, increment=1):
    if not dict_exists(d, keys):
        return insert_dict(d, keys, increment)
    else:
        innermost_d = d
        for i in range(len(keys) - 1):
            innermost_d = d[keys[i]]
        innermost_d[keys[-1]] += increment
        return d


def count_dict(d, levels):
    """
    Count number of entires in nested dictionary
    :param d: dict
    :param levels: int, number of nested levels to count
    :return: int, count
    """
    sum = 0
    levels -= 1
    if levels <= 0:
        return len(d)
    else:
        for k in d:
            sum += count_dict(d[k], levels - 1)
        return sum


def sort_dict(d):
    return {k: d[k] for k in sorted(d.keys())}


def dict_exists(d, keys):
    """
    Check if nested keys exist in nested dictionary
    :param d: dict
    :param keys: list
    :return: bool
    """
    if keys[0] in d:
        if len(keys) == 1:
            return True
        else:
            return dict_exists(d[keys[0]], keys[1:])
    else:
        return False


def dict_get(d, keys, raise_err=True):
    try:
        out = d
        for k in keys:
            out = out[k]
        return out
    except Exception as e:
        if raise_err:
            raise e
        return None


def which_exists(l, elements):
    for e in elements:
        if e in l:
            return e
    return None


def load_json(file_path, raise_err=False):
    try:
        with open(Path(file_path).expanduser(), "r") as f:
            return json.load(f)
    except:
        if raise_err:
            raise OSError("Non-existent or malformed file:", file_path)
        return {}


def save_json(file_path, d, indent=2):
    mkdir(os.path.dirname(file_path))
    with open(file_path, "w+") as f:
        f.write(json.dumps(d, indent=indent))


def list2dict(list_of_dict, key, raise_err=True):
    """
    Given a list of dictionaries, convert to a dictionary of dictionaries, using one of the
    inner dictionary keys as the outer keys. It must exist and be unique for each dictionary.
    :param list_of_dict: list of dictionaries, each dictionary must contain key
    :param key: key to make the common key of the new dictionary
    :param raise_err: bool, whether to allow missing key in list elements.
        If False, elements with missing key are discarded.
    :return: dictionary
    """
    if raise_err:
        return {d[key]: d for d in list_of_dict}
    else:
        return {d[key]: d for d in list_of_dict if key in d}


def list2dict_nonunique(list_of_dict, key):
    """
    Same as above, except key can be non-unique. The values are then a list of elements.
    :param list_of_dict: list of dictionaries, each dictionary must contain key
    :param key: key to make the common key of the new dictionary
    :return: dictionary
    """
    out_dict = {}
    for d in list_of_dict:
        insert_dict_list(out_dict, [d[key]], d)
    return out_dict


def change_dict_key(dict_of_dict, key, orig_key=None):
    """
    Given a dictionary of dictionary, make one of the keys of the nested dictionary
        the new key of the outer dictionary.
    :param dict_of_dict: dict of dict, k --> k --> v
    :param key: key to make the common key of the new outer dictionary
    :param orig_key: hashable, if supplied, gives a name to the original outer key
    :return: dictionary of dictionary
    """
    orig_key = orig_key if orig_key else "orig_key"
    out_dict = {}
    for outer_key in dict_of_dict:
        dict_of_dict[outer_key][orig_key] = outer_key
        out_dict[dict_of_dict[outer_key][key]] = dict_of_dict[outer_key]
    return out_dict


def flip_nested_dict(dict_of_dict):
    """
    Given
    {
        'a': {
            '1': x1,
            '2': x2,
        },
        'b': {
            '1': x3,
            '2': x4,
        }
    }
    outputs
    {
        '1': {
            'a': x1,
            'b': x3
        },
        '2': {
            'a': x2,
            'b': x4
        }
    }
    :param dict_of_dict: dict, with one nested structure (can also be dict of list/tuple)
    :return: dict, nested but flipped
    """
    return_dict = {}
    for outer_key in dict_of_dict:
        for inner_key in dict_of_dict[outer_key]:
            insert_dict(
                return_dict, [inner_key, outer_key], dict_of_dict[outer_key][inner_key]
            )
    return return_dict


def flip_dict_key_value(d):
    """
    Given {'a': 1, 'b': 2, 'c': 2}
    outputs {
        1: ['a'],
        2: ['b', 'c']
    }
    :param d: dict
    :return: dict, flipped
    """
    return dict([(v, [k for k, v1 in d.items() if v1 == v]) for v in set(d.values())])


def dict_parents(d, output_d=None, parents=list(), include_self=True):
    """
    With input:
    {'a': {
            'b': {
                'c': {}
            },
            'd': {}
        }
    }
    output dictionary will be:
    {
        'a': [],
        'b': ['a'],
        'c': ['b', 'a'],
        'd': ['a']
    }
    :param d: dict, all keys must be unique
    :param output_d: dict or None, only for recursion use
    :param parents: list or None, only for recursion use
    :param include_self: bool, whether to include the element itself as its own parent
    :return: dict, keys: each key in d, values: list of its parents,
        in the order of outermost to innermost
    """
    output_d = {} if output_d is None else output_d
    for k in d:
        output_d[k] = parents + [k] if include_self else parents
        dict_parents(
            d[k], output_d=output_d, parents=parents + [k], include_self=include_self
        )
    return output_d


def dict_children(d, output_d=None, children_lists=list(), include_self=True):
    """
    Exactly mirrors dict_parents function.
    :param d: dict, all keys must be unique
    :param output_d: dict or None, only for recursion use
    :param children_lists: list of list, only for recursion use
    :param include_self: bool, whether to include the element itself as its own parent
    :return: dict, keys: each key in d, values: list of its children,
        in the order of outermost to innermost
    """
    output_d = {} if output_d is None else output_d
    for k in d:
        for l in children_lists:
            l.append(k)
        output_d[k] = [k] if include_self else []
        dict_children(
            d[k],
            output_d=output_d,
            children_lists=children_lists + [output_d[k]],
            include_self=include_self,
        )
    return output_d


def traverse_dict_apply_lambda(d, fxn):
    """
    Traverse dictionary or list and convert all Decimal type to python float.
    Uses recursion to handle nested list and dictionaries.
    NOT TESTED.
    :param d: dict
    :return: dict, updated
    """
    if type(d) is dict:
        keys = d.keys()
    elif type(d) is list:
        keys = range(len(d))
    else:
        raise ValueError(f"Unsupported input type {type(d)}")
    # Recursion or appply lambda
    for i in keys:
        if type(d[i]) in (dict, list):
            traverse_dict_apply_lambda(d[i], fxn)
        else:
            d[i] = fxn(d[i])


def update_nested_list(l, mapping, inverse=False):
    for i, el in enumerate(l):
        if type(el) in (list, set):
            update_nested_list(el, mapping)
        else:
            l[i] = mapping.index(l[i]) if inverse else mapping[l[i]]


def flatten_iterable(ite, nested_levels):
    """Flatten (i.e. concatenate) a list of lists into a single list."""
    if nested_levels == 1:
        return list(ite)
    else:
        out = []
        for i in ite:
            out += flatten_iterable(i, nested_levels - 1)
        return out


def in_list(elements, list_):
    """Check if at least one of elements is in list_"""
    count = 0
    for e in elements:
        if e in list_:
            count += 1
    return count


def list_mean(l):
    if len(l):
        return sum(l) / float(len(l))
    else:
        return None


def list_items_equal(l):
    item = l[0]
    for i in l:
        if i != item:
            return False
    return True


def nchoose2(l):
    res = []
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            res.append((l[i], l[j]))
    return res


def ceildiv(a, b):
    """Ceiling division"""
    return -(a // -b)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_header(file_path):
    """
    Returns header of file.

    :param file_path: string, path to file
    :return header: string, header string ending with '\n'
    """
    with open(file_path, "r") as f:
        header = f.readline()
    return header


def keyboard_input_query(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


"""TIMESTAMP RELATED"""


def ntp_now(client=NTPClient(), tz=None, fallback_to_system_time=True):
    try:
        utc_timestamp = client.request("2.north-america.pool.ntp.org", version=3).tx_time
        dt = datetime.datetime.fromtimestamp(
            utc_timestamp, tz=tz or datetime.timezone.utc
        )
        return dt if tz else dt.replace(tzinfo=None)
    except Exception as e:
        print("Get NTP time failed.")
        print(e)
        return datetime.datetime.now() if fallback_to_system_time else None


def get_timestamp(datapoint):
    """
    Returns timestamp of a datapoint, assuming it is the first value.

    :param datapoint: string, values are comma-delimited
    :return: string, timestamp of datapoint
    """
    return datapoint.split(",")[0]


def check_timestamp_format(timestamp, timestamp_format):
    """
    Checks whether timestamp matches regex format defined by timestamp_format.

    :param timestamp: string, timestamp
    :param timestamp_format: regex format
    :return: boolean, whether timestamp matches regex format defined by timestamp_format
    """
    format = re.compile(timestamp_format)
    return format.match(timestamp)


def datetime2iso(dt: datetime.datetime, tod="start"):
    """
    Convert datetime to iso formatted timestamp for beginning or end of day.

    :param dt: datetime object with no time information
    :param tod (time of day): string, 'start'|'end'
        If 'start', returned string's time is midnight.
        If 'end', returned string's time is 1 microsecond before midnight of the next day
        (i.e. 23:59:59.999)
    :return: string, iso timestamp
    """
    if tod == "end":
        dt = dt + datetime.timedelta(days=1) - datetime.timedelta(microseconds=1)
    return dt.isoformat(sep="T", timespec="milliseconds")


def dt_is_tz_naive(dt):
    return dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None


def compare_timestamps(timestamp_1, op, timestamp_2):
    """
    Performs a relational operation on two iso timestamps.
    Timestamps can be of different timezones.
    If any one of the timestamps are timezone naive, they are assumed to be UTC time.

    :param timestamp_1: string, first timestamp
    :param op: operator module comparison function
    :param timestamp_2: string, second timestamp
    """
    # Convert timestamps to datetime
    dt_1 = parser.parse(timestamp_1)
    dt_2 = parser.parse(timestamp_2)

    # If datetime is timezone naive, assume it is in UTC time
    if dt_is_tz_naive(dt_1):
        dt_1 = pytz.timezone("UTC").localize(dt_1)

    if dt_is_tz_naive(dt_2):
        dt_2 = pytz.timezone("UTC").localize(dt_2)

    return op(dt_1, dt_2)


def get_timestamp_delta(timestamp_1, timestamp_2, use_abs=True):
    """
    Returns number of milliseconds between two iso timestamps.
    If any one of the timestamps are timezone naive, they are assumed to be UTC time.

    :param timestamp_1: string
    :param timestamp_2: string
    :param use_abs: boolean, whether to return absolute value of delta
    :return: float
    """
    # Convert timestamps to datetime
    dt_1 = parser.parse(timestamp_1)
    dt_2 = parser.parse(timestamp_2)

    # If datetime is timezone naive, assume it is in UTC time
    if dt_is_tz_naive(dt_1):
        dt_1 = pytz.timezone("UTC").localize(dt_1)

    if dt_is_tz_naive(dt_2):
        dt_2 = pytz.timezone("UTC").localize(dt_2)

    delta = dt_1 - dt_2
    return (
        abs(delta / datetime.timedelta(milliseconds=1))
        if use_abs
        else delta / datetime.timedelta(milliseconds=1)
    )


""" Finding datapoint location/info related (file objects are opened in binary mode)"""


def get_eof_byte(file, path_or_obj):
    """
    Return byte location for end of file.

    :param file: string|file object
    :param path_or_obj, string, 'path'|'obj', whether param 'file' is a file path or file object
    :return: byte location for end of file
    """
    if path_or_obj not in ["path", "obj"]:
        raise ValueError
    if path_or_obj == "path":
        with open(file, "rb") as f:
            f.seek(0, os.SEEK_END)
            return f.tell()
    else:
        file.seek(0, os.SEEK_END)
        return file.tell()


def get_front_byte(byte, file_obj, suppress_warning=False):
    """
    Returns start of current row's byte position and row's timestamp. Moves cursor beginning of next row.
    (Current row = whatever row byte's position is in).
    Only should be called on datapoint rows.

    :param byte: int, current byte position of cursor in relation to top of file
    :param file_obj: file object, file object of opened file
    :return byte_start: int, byte position of front of row
    :return: string, row's timestamp
    """
    try:
        # increments of two chars we read forward one
        file_obj.seek(byte, os.SEEK_SET)  # move readhead to two characters before EOF
        while (
            file_obj.read(1) != b"\n"
        ):  # move readhead back two characters until the first '\n' from EOF is found
            file_obj.seek(-2, os.SEEK_CUR)
    except OSError:  # called on header row
        if not suppress_warning:
            print("Called get_front_byte() on the header row")
        raise  # should never logically occur
    byte_start = file_obj.tell()
    return byte_start, get_timestamp(
        file_obj.readline().decode()
    )  # decode() because file is in binary mode


def get_datapoint_info(byte, file_obj, suppress_warning=False):
    """
    Returns start and end of current row's byte position and row's timestamp. Moves cursor beginning of next row.
    (Current row = whatever row byte's position is in).
    Only should be called on datapoint rows.

    :param byte: int, current byte position of cursor in relation to top of file
    :param file_obj: file object, file object of opened file
    :return byte_start: int, byte position of front of row
    :return byte_end: int, byte position of byte after end of row
    :return datapoint_time: string, row's timestamp
    """
    file_obj.seek(byte, os.SEEK_SET)
    byte_start = byte
    datapoint = file_obj.readline().decode()
    datapoint_time = get_timestamp(datapoint)
    byte_end = file_obj.tell()
    if not check_timestamp_format(datapoint_time, ISO_TIMESTAMP_FORMAT_MS):
        byte_start, datapoint_time = get_front_byte(
            byte, file_obj, suppress_warning=suppress_warning
        )
        byte_end = file_obj.tell()
    return byte_start, byte_end, datapoint_time


def get_previous_datapoint_info(
    byte, file_obj, include_timestamp=True, suppress_warning=False
):
    """
    Returns start of previous row's byte position and row's timestamp. Moves cursor beginning of current row.
    (Current row = whatever row byte's position is in).
    Only should be called on datapoint rows, excluding the first datapoint.

    :param byte: int, current byte position of cursor in relation to top of file
    :param file_obj: file object, file object of opened file
    :return byte_start: int, byte position of front of previous row
    :return datapoint_time: string, row's timestamp
    """
    try:
        # increments of two chars since '\n' is two chars
        file_obj.seek(byte)
        file_obj.seek(-2, os.SEEK_CUR)  # move readhead before previous row's '\n'
        while (
            file_obj.read(1) != b"\n"
        ):  # move readhead back two characters until the row before previous row's '\n' is reached
            file_obj.seek(-2, os.SEEK_CUR)
        byte_start = file_obj.tell()
    except OSError:  # reached header row
        if not suppress_warning:
            print(
                "Called get_previous_datapoint_info() on header row or first datapoint."
            )
        raise
    if include_timestamp:
        return byte_start, get_timestamp(
            file_obj.readline().decode()
        )  # decode() because file is in binary mode
    else:
        return byte_start


def get_next_datapoint_info(
    byte, file_obj, include_timestamp=True, suppress_warning=False
):
    """
    Returns beginning of next row's byte position and row's timestamp. Moves cursor to end of next row.
    Only should be called on datapoint rows, excluding the first datapoint.

    :param byte: int, current byte position of cursor in relation to top of file
    :param file_obj: file object, file object of opened file in binary mode
    :param include_timestamp: bool, if True,  returns the timestamp of the row, otherwise only the byte location
    :return byte_start: int, byte position of beginning of next row
    :return datapoint_time: string, row's timestamp
    """
    file_obj.seek(0, os.SEEK_END)
    eof = file_obj.tell()
    file_obj.seek(byte)
    file_obj.readline()
    byte_start = file_obj.tell()
    if byte_start == eof:
        if not suppress_warning:
            print("Called get_next_datapoint_info() on last row.")
        raise OSError
    if include_timestamp:
        return byte_start, get_timestamp(
            file_obj.readline().decode()
        )  # decode() because file is in binary mode
    else:
        return byte_start


def get_next_datapoint_info_safe(byte, file_path):
    """
    A safe wrapper around get_next_datapoint_info.
    :param byte: int, current byte location
    :param file_path: str, path to csv file
    :return:
        byte_location, int
        end_of_file, bool
    """
    try:
        with open(file_path, "r") as f:
            return (
                get_next_datapoint_info(
                    byte, f, include_timestamp=False, suppress_warning=True
                ),
                False,
            )
    except OSError:
        return os.path.getsize(file_path), True


def get_first_datapoint_info(file_path):
    """
    Returns the first datapoint of file and byte position of beginning of first datapoint.
    If there are no datapoints, returns 'HEADER ROW'.
    If the file is empty, returns 'EMPTY FILE'.

    :param file: string, path to file
    :return byte_start: byte position of beginning of first datapoint
    :return: string, first datapoint|'HEADER ROW'|'EMPTY FILE'
    """
    with open(file_path, "rb") as f:
        try:
            f.readline()  # skips header row
            try:
                byte_start = f.tell()
                return byte_start, f.readline().decode()
            except StopIteration:
                return 0, "HEADER ROW"
        except StopIteration:
            return 0, "EMPTY FILE"


def get_last_datapoint_info(file_path):
    """
    Returns the last datapoint of file and byte position of beginning of last datapoint.
    If there are no datapoints, returns 'HEADER ROW'.
    If the file is empty, returns 'EMPTY FILE'.

    :param file: string, path to file
    :return byte_start: byte position of beginning of last datapoint
    :return: a tupel of (int, str), error code and last datapoint|'HEADER ROW'|'EMPTY FILE'
    """
    with open(
        file_path, "rb"
    ) as f:  # must be in binary mode to do cursor-relative seeks
        # check if file is empty
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return 0, "EMPTY FILE"
        try:
            f.seek(
                -2, os.SEEK_END
            )  # move cursor to two characters before EOF to skip last '\n' in file
            # move cursor back (net) 1 character until the second to last line's '\n' is found
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            byte_start = f.tell()
            return (
                byte_start,
                f.readline().decode(),
            )  # decode() because file is in binary mode
        except OSError:  # file only has header row
            return 0, "HEADER ROW"


def count_lines_in_file(file_path):
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


def binary_search_timestamp(
    file_path,
    target_timestamp,
    find_closest=False,
    return_time=False,
    suppress_warning=False,
):
    """
    Returns byte position for the front of the target_timestamp (get_end_byte=False) OR
    for the end of the target_timestamp (get_end_byte=True).
    If target_timestamp is not found and find_closest is False, then None is returned.
    If target_timestamp is not found and find_closest is True, then the byte location of
    the absolute closest row is returned.

    :param file_path: string, path of file
    :param target_timestamp: string, timestamp to search for, in format TIMESTAMP_FORMAT
    :param find_closest: boolean, whether to search for the closest timestamp if target_timestamp isn't found
    :param return_time: bool, whether to return timestamp along with byte location
    :param suppress_warning: bool
    :return target_byte: int, byte position
    """
    first_datapoint_byte_start, first_datapoint = get_first_datapoint_info(file_path)
    last_datapoint_byte_start, last_datapoint = get_last_datapoint_info(file_path)
    first_timestamp = get_timestamp(first_datapoint)
    last_timestamp = get_timestamp(last_datapoint)
    if compare_timestamps(target_timestamp, operator.lt, first_timestamp):
        if find_closest:
            return (
                (first_datapoint_byte_start, first_datapoint.split(",")[0])
                if return_time
                else first_datapoint_byte_start
            )
        return (None, None) if return_time else None
    elif compare_timestamps(last_timestamp, operator.lt, target_timestamp):
        if find_closest:
            return (
                (last_datapoint_byte_start, last_datapoint.split(",")[0])
                if return_time
                else last_datapoint_byte_start
            )
        return (None, None) if return_time else None

    with open(file_path, "rb") as f:
        f.readline()  # skip header row
        low = f.tell()
        high = get_eof_byte(f, "obj") - 1
        while low <= high:
            mid_byte_calc = (low + high) // 2
            mid_byte_start, mid_byte_end, mid_time = get_datapoint_info(
                mid_byte_calc, f, suppress_warning=suppress_warning
            )
            if compare_timestamps(mid_time, operator.lt, target_timestamp):
                low = mid_byte_end
            elif compare_timestamps(target_timestamp, operator.lt, mid_time):
                high = mid_byte_start - 1
            else:  # target_timestamp found
                target_byte = mid_byte_start
                return (target_byte, mid_time) if return_time else target_byte

        # target_timestamp not found
        if find_closest:
            target_temp_time = mid_time
            target_byte = mid_byte_start
            return_t = mid_time
            while compare_timestamps(target_temp_time, operator.gt, target_timestamp):
                prev_byte_start, prev_time = get_previous_datapoint_info(
                    mid_byte_start, f, suppress_warning=suppress_warning
                )
                target_temp_time = prev_time
                target_byte = prev_byte_start
                return_t = prev_time
            next_byte_start, next_time = get_next_datapoint_info(
                target_byte, f, suppress_warning=suppress_warning
            )
            prev_delta = get_timestamp_delta(
                target_timestamp, target_temp_time, use_abs=True
            )
            next_delta = get_timestamp_delta(target_timestamp, next_time, use_abs=True)
            if prev_delta > next_delta:
                target_byte = next_byte_start
                return_t = next_time
            else:
                pass  # keep target_byte = prev_byte_start
        else:
            target_byte = None
            return_t = None
    return (target_byte, return_t) if return_time else target_byte


def read_file_section(file_path, section_byte_start, section_byte_end=None):
    """
    Returns string of file data from section_byte_start to section_byte_end.
    If section_byte_end is None, function reads until end of file.

    :param file_path: string, path of file
    :param section_byte_start: int, byte location to start reading from
    :param section_byte_end: int, byte location to read to
    :return timeseries_string: string, string of file data from section_byte_start to section_byte_end
    """
    with open(file_path, "rb") as f:
        # set section_byte_end to eof
        if section_byte_end is None:
            f.seek(0, os.SEEK_END)
            section_byte_end = f.tell()

        f.seek(section_byte_start)
        timeseries_string = f.read(section_byte_end - section_byte_start).decode()
    return timeseries_string


def rsync(src, dst, flags=["-ar"]):
    if str(src)[-1] != "/":
        print(f"[WARNING] Source directory does not end with /: {src}")
    mkdir(dst)
    cmd = ["rsync"] + flags + [str(src), str(dst)]
    print(" ".join(cmd))
    subprocess.check_output(cmd)


def cp(src, dst, flags=["-ru"]):
    src = str(src) + "*" if str(src)[-1] == "/" else str(src)
    mkdir(dst)
    cmd = " ".join(["cp"] + flags + [str(src), str(dst)])
    print(cmd)
    subprocess.run(cmd, shell=True)
    

"""CSVMapper"""
class CSVMapper:
    """
    Convert csv data (row-oriented) to json-style data.
    This can be used to create mappings, e.g. from Jace names to building names.
    Duplicates and missing entries (i.e. non-bijective mappings) are supported.
    """

    def __init__(self, csv_file, joined_cols=tuple()):
        """
        :param csv_file: str, path to file
        :param joined_cols: tuple of str, column headers of columns that contain entries that are
            joined by JOIN_CHAR. These entries will be broken apart
        """
        self.csv_file = csv_file
        self.csv_data = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
        self.joined_cols = joined_cols
        self.mapping_cache = {}
        self._available_src_keys = {}

    def populate_mapping(self, src, dst):
        """
        Populate mapping cache for mapping from src column to dst column in csv file.
        This creates a dictionary of mappings, stored under self.mapping_cache[src][dst].
        :param src: str or list/tuple of str, column headers of source column in csv file
        :param dst: str, column header of destination column in csv file
        """
        assert dst in self.csv_data, f"Non-existing dst column name: {dst}"
        if type(src) is str:
            assert src in self.csv_data, f"Non-existing src column name: {src}"
        else:
            for c in src:
                assert c in self.csv_data, f"Non-existing src column name: {src}"
            src = tuple(src)

        # If we've never mapped from src before
        if not (src in self.mapping_cache):
            self.mapping_cache[src] = {}
            self.populate_src_keys(src)

        # If we've never mapped from src to dst before
        if not (dst in self.mapping_cache[src]):
            self.mapping_cache[src][dst] = {}
            for i in range(df_len(self.csv_data)):
                # Retrieve source column data
                if type(src) is str:
                    s = self.csv_data[src][i]
                else:
                    # All source columns must be filled
                    s = [self.csv_data[c][i] for c in src]
                    s = tuple(s) if all([bool(v) for v in s]) else None
                # Retrieve destination column data
                if self.csv_data[dst][i]:
                    d = (
                        set(self.csv_data[dst][i].split(JOIN_CHAR))
                        if dst in self.joined_cols
                        else {self.csv_data[dst][i]}
                    )
                else:
                    d = set()
                # Store a mapping if src is not empty string
                # Alternatively, Store a mapping if both src and dst data are not empty string
                # if s and d:
                if s:
                    if not (s in self.mapping_cache[src][dst]):
                        self.mapping_cache[src][dst][s] = d
                    else:
                        self.mapping_cache[src][dst][s] = self.mapping_cache[src][dst][
                            s
                        ].union(d)

    def populate_src_keys(self, src):
        """
        Populate self._available_src_keys, which is used to efficiently check
        if an entry exists in the csv data.
        :param: src, str or tuple of str, column header(s) in csv file
        """
        if not (type(src) is str):
            if not (src in self._available_src_keys):
                src_cols = self.csv_data[list(src)]
                valid_rows = src_cols[(src_cols != "").all(axis=1)]
                self._available_src_keys[tuple(src)] = set(
                    [tuple(row.tolist()) for _, row in valid_rows.iterrows()]
                )
        else:
            if not (src in self._available_src_keys):
                self._available_src_keys[src] = set(list(self.csv_data[src]))

    def src_keys(self, src):
        self.populate_src_keys(src)
        return self._available_src_keys[src]

    def map(self, src, dst, src_element, print_err=False):
        """
        Caution: If you would like to directly operate on the function output,
        use copy.deepcopy, as the set returned is passed by reference.
        Three mapping modes:
            - Mode 1: From one column to another column (src --> dst)
                src is str, dst is str, src_element is str, output is a set of str
            - Mode 2: From multiple columns to another column ([src1, src2, ...] --> dst).
                e.g. Map 'BTU_CHW' (Meter name) and 'AN16_1' (Jace) to a Building String.
                src_element and src both must be iterables of equal length.
            - Mode 3: From multiple entries in one column to another column ([src1, src2, ...] --> dst).
                e.g. Map 3 meters to the union of their respective buildings.
                src is a str, and src_element is a list/set/tuple.
        :param src: str or tuple of str, column header(s) in csv file
        :param dst: str, column header(s) in csv file
        :param src_element: str, or a tuple of string, the element we wish to map
        :param print_err: bool, whether to print warning messages
        :return: dst_elements, set of string(s) in the destination column,
            If the set has length 1, the mapping is 1-to-1.
            If the set has length 0, no mapping is found (data src and/or dst is missing).
            If the set has length 2, the mapping is not 1-to-1.
        """
        self.populate_mapping(src, dst)

        # Modes 1 & 2
        if (src_element is str) or (not (src is str)):
            if print_err and (not (src_element in self.src_keys(src))):
                print(
                    f"[Warning][map] Source element: {src_element} is not in {self.csv_file} column {src}"
                )
            return (
                copy.deepcopy(self.mapping_cache[src][dst][src_element])
                if src_element in self.mapping_cache[src][dst]
                else set()
            )
        # Mode 3
        else:
            out = set()
            for e in src_element:
                out = out.union(
                    self.mapping_cache[src][dst][e]
                    if e in self.mapping_cache[src][dst]
                    else set()
                )
            return copy.deepcopy(out)

    def save_as_json(self, src, dst, json_path):
        with open(json_path, "w+") as f:
            f.write(json.dumps(self.mapping_cache[src][dst], indent=2))

    def save_as_csv(self, src, dst, csv_path, mode):
        """
        Save dictionary as csv
        :param src: source column key
        :param dst: destination column key
        :param csv_path: str, path to save file
        :param mode: string, one of {JOIN_CHAR, 'new row'}.
        - JOIN_CHAR: join multiple outputs with JOIN_CHAR symbol in one row
        - 'new row': create a new row for each output
        :return:
        """
        self.dict_as_df(self.mapping_cache[src][dst], src, dst, mode).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )

    @staticmethod
    def dict_as_df(dictionary, src, dst, mode, type="pd", preserve_empty_mapping=True):
        """
        Save a dictionary as csv file.
        :param dictionary: dict containing mapping
        :param src: str, source column header
        :param dst: str, destination column header
        :param type: str, one of 'pd' or 'dict'
        :param mode: string, one of {JOIN_CHAR, 'new row'}.
            - JOIN_CHAR: join multiple outputs with JOIN_CHAR symbol in one row
            - 'new row': create a new row for each output
        :param preserve_empty_mapping: boolean, whether to store/skip empty mapping as a row
        :return: pandas dataframe, csv can be saved using df.to_csv()
        """
        df = {src: [], dst: []}
        for s in dictionary:
            if dictionary[s]:
                if mode == "new row":
                    for d in dictionary[s]:
                        df[src].append(s)
                        df[dst].append(d)
                elif mode == JOIN_CHAR:
                    df[src].append(s)
                    df[dst].append(JOIN_CHAR.join(list(dictionary[s])))
                else:
                    raise ValueError(f"Unknown input option: {mode}")
            elif preserve_empty_mapping:
                df[src].append(s)
                df[dst].append("")
        return pd.DataFrame(df) if type == "pd" else df


def load_waveforms(path_waveforms, timestamp, meter_dirs, return_pandas=True, internal_call=False):
    """
    Load waveforms captured at the same time (raw data)
    :param path_waveforms: string, the path to sample_dataset/waveforms
    :param timestamp: string, the desired data timestamp, formatted like '2024-12-02T17:49:50'
    :param meter_dirs: list of string, the names of meters to include
    :param return_pandas: boolean, whether to return data in pandas dataframe format
        If False, data is returned as a dictionary of numpy arrays
    :return: nested dictionary, waveforms at the closest time to timestamp
        where data from all meters are available, 
        {'2024-12-02T17:49:50': {
            'meter_1': {'t': array, 'v': array},
            'meter_2': {'t': array, 'v': array},
            ...},
        }
    """
    input_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    input_date = input_time.strftime("%Y-%m-%d")
    result = {}
    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2}\.\d+)\.parquet")

    # YYYY-MM-DD folder
    def parse_folder_dates(base_path):
        return sorted(
            [
                folder
                for folder in os.listdir(base_path)
                if re.match(r"\d{4}-\d{2}-\d{2}", folder) and os.path.isdir(os.path.join(base_path, folder))
            ]
        )

    # get parquet files under specific path
    def get_parquet_files(folder_path):
        parquet_files = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".parquet")
        ]
        parsed_files = []
        for file_path in parquet_files:
            match = timestamp_pattern.search(os.path.basename(file_path))
            if match:
                file_date = match.group(1)
                file_time = match.group(2).replace("-",":")
                full_timestamp = f"{file_date}T{file_time}"
                try:
                    file_datetime = datetime.datetime.strptime(full_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
                    parsed_files.append((file_datetime, file_path))
                except ValueError:
                    continue
        return sorted(parsed_files, key=lambda x: x[0])

    # nearest and second-nearest
    def find_nearest_files(files, target_time):
        if not files:
            return None, None
        diffs = [(abs((file_time - target_time).total_seconds()), file_time, file_path) for file_time, file_path in files]
        diffs.sort(key=lambda x: x[0])
        return diffs[0], (diffs[1] if len(diffs) > 1 else None)

    nearest_files = {}
    nearest2_files = {}

    for meter_dir in meter_dirs:
        base_path = os.path.join(path_waveforms, meter_dir)
        if not internal_call:
            folder_dates = parse_folder_dates(base_path)
            if not folder_dates:
                raise ValueError(f"No date folders found in {base_path}")

            if input_date < folder_dates[0]:
                target_folders = [folder_dates[0]]
            elif input_date > folder_dates[-1]:
                target_folders = [folder_dates[-1]]
            else:
                date_diffs = [(abs((datetime.datetime.strptime(folder_date, "%Y-%m-%d") - datetime.datetime.combine(input_time.date(), datetime.datetime.min.time())).days), folder_date)
                            for folder_date in folder_dates]
                date_diffs.sort(key=lambda x: x[0])
                closest_date = date_diffs[0][1]
                target_folders = [closest_date]
                idx = folder_dates.index(closest_date)

                files_in_current = get_parquet_files(os.path.join(base_path, closest_date))
                if files_in_current:
                    min_time = files_in_current[0][0]
                    max_time = files_in_current[-1][0]
                    if input_time < min_time and idx > 0:
                        target_folders.insert(0, folder_dates[idx - 1])
                    if input_time > max_time and idx < len(folder_dates) - 1:
                        target_folders.append(folder_dates[idx + 1])
        else:
            target_folders = [input_date]

        all_files = []
        for folder in target_folders:
            folder_path = os.path.join(base_path, folder)
            all_files += get_parquet_files(folder_path)
        nearest, second_nearest = find_nearest_files(all_files, input_time)
        if not nearest:
            raise ValueError(f"No valid parquet files found near {timestamp} in {meter_dir}")
        nearest_files[meter_dir] = nearest
        if second_nearest:
            nearest2_files[meter_dir] = second_nearest

    # timestamps of the first meter
    base_meter, base_file_info = list(nearest_files.items())[0]
    _, _, base_file_path = base_file_info
    base_df = pd.read_parquet(base_file_path)
    base_min_time, base_max_time = pd.to_datetime(base_df['t']).agg(['min', 'max'])
    base_first_timestamp = pd.to_datetime(base_df.iloc[0]['t']).strftime('%Y-%m-%dT%H:%M:%S')
    result[base_first_timestamp] = {}

    file_cache = {}
    for meter_dir, (_, _, file_path) in nearest_files.items():
        candidate_df = pd.read_parquet(file_path)
        file_cache[file_path] = candidate_df
        candidate_min_time, candidate_max_time = pd.to_datetime(candidate_df['t']).agg(['min', 'max'])
        while not (candidate_max_time >= base_min_time and candidate_min_time <= base_max_time):
            if meter_dir not in nearest2_files:
                print(f"Warning: No overlapping time range found between {base_meter} and {meter_dir}. Returning no data.")
                return {}
            _, _, next_file_path = nearest2_files.pop(meter_dir)
            candidate_df = pd.read_parquet(next_file_path)
            file_cache[next_file_path] = candidate_df
            candidate_min_time, candidate_max_time = pd.to_datetime(candidate_df['t']).agg(['min', 'max'])
            nearest_files[meter_dir] = (_, _, next_file_path)

    for meter_dir, (_, _, file_path) in nearest_files.items():
        result[base_first_timestamp][meter_dir] = file_cache[file_path]
    
    if not return_pandas:
        result = {base_first_timestamp: {meter_dir: \
            {'t': df['t'].to_numpy(), 'v': df['v'].to_numpy()} \
            for meter_dir, df in result[base_first_timestamp].items()
        }}
    return result


def load_waveforms_timerange(path_waveforms, timestamp_range, meter_dirs, return_pandas=True):
    """
    Load multiple waveforms captured at multiple times (a range of time input)
    (raw sampled data, not perfectly time-aligned)
    :param path_waveforms: string, the path to sample_dataset/waveforms
    :param timestamp_range: tuple of string, timestamps of start and end times,
                           formatted like ('2024-12-02T17:49:50', '2024-12-02T17:52:50')
    :param meter_dirs: list of string, the names of meters to include
    :param return_pandas: boolean, whether to return data in pandas dataframe format
        If False, data is returned as a dictionary of numpy arrays
    :return: nested dictionary, all available data within the timerange, e.g.
        {'2024-12-02T17:49:50': {
            'meter_1': {'t': array, 'v': array},
            'meter_2': {'t': array, 'v': array},
            ...},
        '2024-12-02T17:51:50': {
            'meter_1': {'t': array, 'v': array},
            'meter_2': {'t': array, 'v': array},
            ...},
        ...}
    """
    start_time = datetime.datetime.strptime(timestamp_range[0], "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(timestamp_range[1], "%Y-%m-%dT%H:%M:%S")

    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2})\.\d+\.parquet")

    # set the first meter as reference
    base_meter_dir = meter_dirs[0] # reference meter (can be replaced with others)
    base_path = os.path.join(path_waveforms, base_meter_dir)

    current_date = start_time
    all_dates = []
    while current_date <= end_time:
        all_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += datetime.timedelta(days=1)

    base_files = []
    for date_str in all_dates:
        date_folder = os.path.join(base_path, date_str)
        if os.path.exists(date_folder) and os.path.isdir(date_folder):
            for root, _, files in os.walk(date_folder):
                base_files.extend([os.path.join(root, f) for f in files if f.endswith(".parquet")])
        else:
            print(f"Skipping {date_str}: No data")
    if not base_files:
        raise ValueError(f"No parquet files found in {base_meter_dir}")

    base_valid_files = []
    for file_path in base_files:
        match = timestamp_pattern.search(os.path.basename(file_path))
        if match:
            file_date = match.group(1)
            file_time = match.group(2).replace("-", ":")
            full_timestamp = f"{file_date}T{file_time}"    
            try:
                file_datetime = datetime.datetime.strptime(full_timestamp, "%Y-%m-%dT%H:%M:%S")
                if start_time <= file_datetime <= end_time:
                    base_valid_files.append((file_datetime, file_path))
            except ValueError as e:
                raise ValueError(f"Error parsing timestamp from file {file_path}: {e}")
    if not base_valid_files:
        raise ValueError(f"No valid parquet files found in {base_meter_dir} for the given timestamp range")

    base_valid_files.sort(key=lambda x: x[0])
    base_timestamps = list(dt for dt, _ in base_valid_files)

    result = {}
    for timestamp in base_timestamps:
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            timestamp_data = load_waveforms(path_waveforms, timestamp_str, meter_dirs, return_pandas=return_pandas, internal_call=True)
            if not timestamp_data:
                print(f"Skipping empty data for timestamp {timestamp_str}")
                continue
            data_key = next(iter(timestamp_data.keys()))
            if data_key in result:
                print(f"Skipping duplicate timestamp {timestamp_str}")
                continue
            result[timestamp_str] = timestamp_data[timestamp_str]
        except ValueError as e:
            print(f"Skipping timestamp {timestamp_str} due to error: {e}")
    return result


def resample_waveforms(data, freq="400us"):
    """
    Re-sample multiple waveforms with regular-interval time-aligned timestamps.
    Note: For non-overlapping time ranges, NaN values are filled in.
    :param data: nested dictionary, output of load_waveforms or load_waveforms_timerange
    :return: nested dictionary of pandas DataFrame, e.g. 
        {'2024-12-02T17:49:50': {
            'meter_1': {'t': array, 'v': array},
            'meter_2': {'t': array, 'v': array},
            ...},
        '2024-12-02T17:51:50': {
            'meter_1': {'t': array, 'v': array},
            'meter_2': {'t': array, 'v': array},
            ...},
        ...}
    """
    data_copy = copy.deepcopy(data)
    for timestamp, dfs in data_copy.items():
        time_ranges = [
            (pd.to_datetime(df['t']).min(), pd.to_datetime(df['t']).max())
            for df in dfs.values()
        ]
        min_time = max(start for start, _ in time_ranges)
        max_time = min(end for _, end in time_ranges)

        if min_time >= max_time:
            raise ValueError(f"No overlapping time range for timestamp {timestamp}")

        new_time_index = pd.date_range(start=min_time, end=max_time, freq=freq)

        for key, df in dfs.items():
            df['t'] = pd.to_datetime(df['t'])
            missing_times = new_time_index[~new_time_index.isin(df['t'])]
            if len(missing_times) > 0:
                missing_data = pd.DataFrame({'t':missing_times, 'v': [np.nan]*len(missing_times)})
                df = pd.concat([df, missing_data]).sort_values('t').reset_index(drop=True)
            df = df.set_index('t').sort_index()
            df['v'] = df['v'].interpolate(method='time')
            df = df.reset_index()
            df_aligned = df[df['t'].isin(new_time_index)].reset_index(drop=True)
            dfs[key] = df_aligned

    return data_copy
