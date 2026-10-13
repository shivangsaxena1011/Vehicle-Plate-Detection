"""Pluggable vehicle-data provider interface.

The default provider uses the local CSV demo database.
A ``RealProvider`` stub shows how to integrate an authorized external API
once proper credentials and permissions are obtained.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .database import load_vehicle_database, lookup_vehicle

logger = logging.getLogger(__name__)


@dataclass
class VehicleLookupResult:
    """Standardized lookup result from any provider."""
    status: str                          # "found", "not_found", "provider_unavailable", "error"
    source: str = "unknown"              # e.g. "demo_csv", "authorized_api"
    vehicle: Optional[Dict[str, Any]] = None
    message: str = ""
    is_demo_data: bool = False


class VehicleProvider(abc.ABC):
    """Abstract base class for vehicle-data providers."""

    @abc.abstractmethod
    def lookup(self, plate_text: str) -> VehicleLookupResult:
        ...

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        ...


class DemoCSVProvider(VehicleProvider):
    """Looks up vehicle records in the local CSV demo database.

    All records are clearly labeled as SYNTHETIC DEMO DATA.
    A lookup miss reports that no demo record was found — it never claims
    the vehicle is officially unregistered.
    """

    def __init__(self, csv_path: str = "data/vehicles.csv"):
        self.csv_path = csv_path
        self._records = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            self._records = load_vehicle_database(self.csv_path)
            logger.info("Demo CSV loaded from %s (%d records)",
                        self.csv_path, len(self._records))
        except Exception as exc:
            self._load_error = f"Failed to load demo database: {exc}"
            logger.error(self._load_error, exc_info=True)

    @property
    def provider_name(self) -> str:
        return "Local CSV Demo Database"

    def lookup(self, plate_text: str) -> VehicleLookupResult:
        if self._records is None:
            return VehicleLookupResult(
                status="error",
                source="demo_csv",
                message=self._load_error or "Demo database not loaded.",
                is_demo_data=True,
            )

        result = lookup_vehicle(plate_text, self._records)

        if result["status"] == "found":
            vehicle = result["vehicle"]
            vehicle["_data_source"] = "SYNTHETIC DEMO DATA"
            return VehicleLookupResult(
                status="found",
                source="demo_csv",
                vehicle=vehicle,
                message="Matching record found in the local demo database.",
                is_demo_data=True,
            )

        return VehicleLookupResult(
            status="not_found",
            source="demo_csv",
            message=(
                "No matching record in the local demo database. "
                "This does NOT mean the vehicle is unregistered."
            ),
            is_demo_data=True,
        )


class AuthorizedAPIProvider(VehicleProvider):
    """Placeholder for a real, authorized vehicle-data provider.

    This is intentionally non-functional.  Integration requires:
    1. Documented API access approval (e.g. VAHAN/Parivahan portal, or
       an authorized commercial aggregator).
    2. Valid API credentials stored in environment variables.
    3. Verified API documentation and permitted data fields.
    4. Rate-limiting, audit logging, and data-retention compliance.

    Do NOT scrape, reverse-engineer, or bypass CAPTCHAs/access controls
    on any government or private service.
    """

    def __init__(self):
        logger.info("AuthorizedAPIProvider instantiated (not configured).")

    @property
    def provider_name(self) -> str:
        return "Authorized Vehicle API (NOT CONFIGURED)"

    def lookup(self, plate_text: str) -> VehicleLookupResult:
        return VehicleLookupResult(
            status="provider_unavailable",
            source="authorized_api",
            message=(
                "No authorized vehicle-data provider is configured. "
                "To enable real lookups, obtain proper API access and credentials, "
                "then implement this provider. See README for integration guidance."
            ),
        )
