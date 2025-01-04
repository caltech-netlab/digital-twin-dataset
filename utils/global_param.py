"""Global Variables (@yxie20)"""

# Third-party imports
import os
from pathlib import Path
import numpy as np

# np.seterr(all="raise")

REPO_DIR = Path(__file__).parent.parent.resolve()
FILE_PATHS = {
    "magnitudes": os.path.join(REPO_DIR, "sample_dataset", "magnitudes"),
    "phasors": os.path.join(REPO_DIR, "sample_dataset", "phasors"),
    "waveforms": os.path.join(REPO_DIR, "sample_dataset", "waveforms"),
    "element_inheritance": os.path.join(REPO_DIR, "utils", "element_inheritance.json"),
    "cable_info": os.path.join(REPO_DIR, "utils", "cable_info.csv"),
    "net_files": os.path.join(REPO_DIR, "sample_dataset", "topology", "network_files"),
    "net_params": os.path.join(REPO_DIR, "sample_dataset", "topology", "parameter_timeseries"),
    "net_metadata": os.path.join(REPO_DIR, "sample_dataset", "topology", "metadata"),
}

# CSV data file column convention
COLUMNS_ORDER = [
    "t",
    "v",
    "r",
    "s",
    "err",
    "info",
    "sum",
    "mean",
    "meters down",
    "meters up",
    "EM down",
    "HW down",
    "CHW down",
    "EM up",
    "HW up",
    "CHW up",
]

F = 60.0
LOCATION = "America/Los_Angeles"
ISO_TIMESTAMP_FORMAT_MS = (
    "[\d]{4}-[\d]{2}-[\d]{2}T[\d]{2}:[\d]{2}:[\d]{2}.[\d]{3}[+-][\d]{2}:[\d]{2}"
)

# Print options
np.set_printoptions(precision=2, linewidth=100)

# Time-related
TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"
ONE_MS = np.timedelta64(1, "ms")
ONE_SEC = np.timedelta64(1, "s")
ONE_DAY = np.timedelta64(1, "D")

# Custom dataframe default dtype and values
STD_DTYPE = {
    "t": "datetime64[us]",
    "v": "float",
    "r": "int",
    "s": "int",
    "err": "int",
    "info": "int",
    "timezone": "U6",
    "sum": "float",
    "mean": "float",
    "meters down": "int",
    "meters up": "int",
    "EM down": "int",
    "HW down": "int",
    "CHW down": "int",
    "EM up": "int",
    "HW up": "int",
    "CHW up": "int",
    "name": "<U64",
    "phase_angle": "float",
    "frequency": "float",
    "rms": "float",
    "t0": "datetime64[us]",
    "t_zero_crossing": "datetime64[us]",
    "t_local": "datetime64[ms]",
    "a": "csingle",
    "b": "csingle",
    "c": "csingle",
    "an": "csingle",
    "bn": "csingle",
    "cn": "csingle",
    "ag": "csingle",
    "bg": "csingle",
    "cg": "csingle",
    "datetime": "datetime64[s]",
    "tempmax": "float",
    "tempmin": "float",
    "temp": "float",
    "feelslikemax": "float",
    "feelslikemin": "float",
    "feelslike": "float",
    "dew": "float",
    "humidity": "float",
    "precip": "float",
    "precipprob": "float",
    "precipcover": "float",
    "preciptype": "<U32",
    "snow": "float",
    "snowdepth": "float",
    "windgust": "float",
    "windspeed": "float",
    "winddir": "float",
    "sealevelpressure": "float",
    "cloudcover": "float",
    "visibility": "float",
    "solarradiation": "float",
    "solarenergy": "float",
    "uvindex": "float",
    "severerisk": "float",
    "sunrise": "datetime64[s]",
    "sunset": "datetime64[s]",
    "moonphase": "float",
    "conditions": "<U64",
    "description": "<U128",
    "icon": "<U32",
    "stations": "<U128",
}
STD_VALUE = {
    "t": np.datetime64("NaT"),
    "v": float("NaN"),
    "r": 0,
    "s": 0,
    "err": 0,
    "info": 0,
    "timezone": "",
    "sum": float("NaN"),
    "mean": float("NaN"),
    "meters down": 0,
    "meters up": 0,
    "EM down": 0,
    "HW down": 0,
    "CHW down": 0,
    "EM up": 0,
    "HW up": 0,
    "CHW up": 0,
    "name": "",
    "phase_angle": float("NaN"),
    "frequency": float("NaN"),
    "t0": np.datetime64("NaT"),
    "t_zero_crossing": np.datetime64("NaT"),
    "t_local": np.datetime64("NaT"),
    "datetime": np.datetime64("NaT"),
    "tempmax": float("NaN"),
    "tempmin": float("NaN"),
    "temp": float("NaN"),
    "feelslikemax": float("NaN"),
    "feelslikemin": float("NaN"),
    "feelslike": float("NaN"),
    "dew": float("NaN"),
    "humidity": float("NaN"),
    "precip": float("NaN"),
    "precipprob": float("NaN"),
    "precipcover": float("NaN"),
    "preciptype": "",
    "snow": float("NaN"),
    "snowdepth": float("NaN"),
    "windgust": float("NaN"),
    "windspeed": float("NaN"),
    "winddir": float("NaN"),
    "sealevelpressure": float("NaN"),
    "cloudcover": float("NaN"),
    "visibility": float("NaN"),
    "solarradiation": float("NaN"),
    "solarenergy": float("NaN"),
    "uvindex": float("NaN"),
    "severerisk": float("NaN"),
    "sunrise": np.datetime64("NaT"),
    "sunset": np.datetime64("NaT"),
    "moonphase": float("NaN"),
    "conditions": "",
    "description": "",
    "icon": "",
    "stations": "",
}

# @yxie20 below
# Defines the meaning of error and info code (number) in csv files
ERR_CODES = {
    "missing data": 0,
    "zero subseq.": 1,
    "spikes": 2,
    "out of bound": 3,
    # Faulty meters below (entire timeseries is faulty):
    "empty input data": 4,
    "constant": 5,
    "too many anomalies": 6,
    # Others
    "resample error": 7,  # Gap is too big in resampling
}
# Note that the output in csv will be binary-encoded, can start with 0 since 2**0 = 1.
# e.g. 5 = 4 + 1 = 2**2 + 2**0 --> meaning 'faulty meter' and 'spiking constant'
INFO_CODES = {
    "faulty meter": 0,
    "daily pattern": 1,
    "spiking const.": 2,
    "mean shift": 3,
}

PYTHON2NUMPY_TIME_UNIT = {
    "years": "Y",
    "months": "M",
    "weeks": "W",
    "days": "D",
    "hours": "h",
    "minutes": "m",
    "seconds": "s",
    "milliseconds": "ms",
    "microseconds": "us",
    "nanoseconds": "ns",
    "picoseconds": "ps",
    "femtosecond": "fs",
    "attosecond": "as",
}
NUMPY_TIME_UNITS = [
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "fs",
    "as",
]
PYTHON_TIME_UNITS = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
    "picoseconds",
    "femtosecond",
    "attosecond",
]
# Special character for joining multiple outputs in one csv entry
JOIN_CHAR = "|"
DIR_CHAR = "->"
METERNAME_DELIMITERS = r"_|-|\."
