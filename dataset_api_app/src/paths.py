# Third-party imports
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATASET_API_APP_DIR = Path(__file__).parent.parent.resolve()
"""Root path for the Dataset API App."""

LOGS_DIR = DATASET_API_APP_DIR / "logs"
"""Directory to write log files to."""

API_USAGE_LOG_PATH = LOGS_DIR / "api_usage.log"
"""Path to the API usage log file."""

REPLACEMENT_LOOKUP_FILE = DATASET_API_APP_DIR / "replacement_lookup.json"
"""Path to the file containing a mapping from real to anonymized element names."""

FLASK_TEMPLATES_DIR = DATASET_API_APP_DIR / "templates"
"""Directory containing Flask templates."""

USERS_DB_PATH = DATASET_API_APP_DIR / "users.db"
"""Path to the users database."""

MAGNITUDES_DIR = Path(os.environ.get("MAGNITUDES_DIR", "/data/magnitudes"))
"""Path to the directory containing raw magnitude data."""

PHASORS_DIR = Path(os.environ.get("PHASORS_DIR", "/data/phasors"))
"""Path to the directory containing raw phasor data."""

WAVEFORMS_DIR = Path(os.environ.get("WAVEFORMS_DIR", "/data/waveforms"))
"""Path to the directory containing raw waveform data after 11/01/2024."""

WAVEFORMS_2024_10_DIR = Path(
    os.environ.get("WAVEFORMS_2024_10_DIR", "/data/waveforms_2024_10")
)
"""Path to the directory containing raw waveform data before 11/01/2024."""

WAVEFORMS_2024_10_CUTOFF = datetime(2024, 11, 1)
"""Use `WAVEFORMS_2024_10_DIR` for waveforms before this date."""

WAVEFORMS_2025_03_DIR = Path(
    os.environ.get("WAVEFORMS_2025_03_DIR", "/data/waveforms_2025_03")
)
"""Path to the directory containing raw waveform data from 11/01/2024 to 04/01/2025."""

WAVEFORMS_2025_03_CUTOFF = datetime(2025, 4, 1)
"""Use `WAVEFORMS_2025_03_DIR` for waveforms before this date (and after `WAVEFORMS_2024_10_CUTOFF`)."""
