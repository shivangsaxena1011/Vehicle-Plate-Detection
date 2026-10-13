"""CSV demo database for vehicle lookup.

All bundled records are SYNTHETIC DEMO DATA — they do not represent
real vehicles or real registrations.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "registration_number", "make", "model", "fuel", "vehicle_class", "status"
]


def normalize(value: str) -> str:
    """Normalize a registration string for comparison.

    Strips all non-alphanumeric characters and upper-cases.
    """
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def load_vehicle_database(path: str = "data/vehicles.csv") -> pd.DataFrame:
    """Load the CSV vehicle database.

    Returns an empty DataFrame with the correct columns if the file
    is missing, and raises ValueError if required columns are absent.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Vehicle database not found at %s — using empty database.", p)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    try:
        frame = pd.read_csv(p, dtype=str).fillna("")
    except Exception as exc:
        logger.error("Failed to parse vehicle CSV at %s: %s", p, exc)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Vehicle CSV at '{p}' is missing required columns: {', '.join(missing)}"
        )

    # Drop exact duplicate rows
    frame = frame.drop_duplicates(subset=["registration_number"], keep="first")

    return frame


def lookup_vehicle(plate: str, records: pd.DataFrame) -> Dict[str, Any]:
    """Look up a registration number in the demo database.

    Returns ``{"status": "found", "vehicle": {...}}`` on match,
    or ``{"status": "not_found"}`` on miss.

    A miss means the plate is not in the demo CSV.  It does **not** mean
    the vehicle is officially unregistered.
    """
    if records is None or records.empty or not plate:
        return {"status": "not_found"}

    target = normalize(plate)
    if not target:
        return {"status": "not_found"}

    matches = records[records["registration_number"].map(normalize) == target]

    if matches.empty:
        return {"status": "not_found"}

    record = matches.iloc[0].to_dict()
    record["_data_source"] = "SYNTHETIC DEMO DATA"
    return {"status": "found", "vehicle": record}
