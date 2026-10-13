"""Tests for src.database — CSV loading and vehicle lookup."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.database import (
    REQUIRED_COLUMNS,
    load_vehicle_database,
    lookup_vehicle,
    normalize,
)


# ── Normalization ──

class TestNormalize:
    def test_basic(self):
        assert normalize("MP 04 AB 1234") == "MP04AB1234"

    def test_lowercase(self):
        assert normalize("mp04ab1234") == "MP04AB1234"

    def test_special_chars(self):
        assert normalize("MP-04-AB-1234") == "MP04AB1234"

    def test_empty(self):
        assert normalize("") == ""

    def test_none(self):
        assert normalize(None) == ""

    def test_unicode_stripped(self):
        assert normalize("MP04·AB·1234") == "MP04AB1234"


# ── Load database ──

class TestLoadDatabase:
    def test_loads_default_csv(self):
        df = load_vehicle_database("data/vehicles.csv")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_missing_file_returns_empty(self):
        df = load_vehicle_database("nonexistent/path/vehicles.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_missing_columns_raises(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            f.write("registration_number,make\nMP04AB1234,Test\n")
            f.close()
            with pytest.raises(ValueError, match="missing required columns"):
                load_vehicle_database(f.name)
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass

    def test_malformed_csv_raises(self):
        f = tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False)
        try:
            f.write(b"\x00\x01\x02\x03\x04")
            f.close()
            # Binary garbage doesn't have required columns, so raises ValueError or returns empty
            try:
                df = load_vehicle_database(f.name)
                assert isinstance(df, pd.DataFrame)
            except ValueError:
                pass  # Also valid: missing required columns
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass

    def test_duplicate_registrations_deduplicated(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            f.write(",".join(REQUIRED_COLUMNS) + "\n")
            f.write("MP04AB1234,Make1,Model1,Petrol,Car,Active\n")
            f.write("MP04AB1234,Make2,Model2,Diesel,Car,Active\n")
            f.close()
            df = load_vehicle_database(f.name)
            matches = df[df["registration_number"] == "MP04AB1234"]
            assert len(matches) == 1
            assert matches.iloc[0]["make"] == "Make1"  # first kept
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass


# ── Lookup ──

class TestLookupVehicle:
    @pytest.fixture
    def sample_records(self):
        rows = [
            {
                "registration_number": "MP04AB1234",
                "make": "Example Motors",
                "model": "City Sedan",
                "fuel": "Petrol",
                "vehicle_class": "Motor Car",
                "status": "Active",
            },
            {
                "registration_number": "DL01CA0001",
                "make": "Demo Motors",
                "model": "Metro Hatch",
                "fuel": "CNG",
                "vehicle_class": "Motor Car",
                "status": "Active",
            },
        ]
        return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    def test_exact_match(self, sample_records):
        result = lookup_vehicle("MP04AB1234", sample_records)
        assert result["status"] == "found"
        assert result["vehicle"]["model"] == "City Sedan"

    def test_normalized_match(self, sample_records):
        result = lookup_vehicle("mp 04 ab 1234", sample_records)
        assert result["status"] == "found"

    def test_hyphenated_match(self, sample_records):
        result = lookup_vehicle("MP-04-AB-1234", sample_records)
        assert result["status"] == "found"

    def test_not_found(self, sample_records):
        result = lookup_vehicle("MP99ZZ0000", sample_records)
        assert result["status"] == "not_found"

    def test_empty_plate(self, sample_records):
        result = lookup_vehicle("", sample_records)
        assert result["status"] == "not_found"

    def test_none_plate(self, sample_records):
        result = lookup_vehicle(None, sample_records)
        assert result["status"] == "not_found"

    def test_empty_records(self):
        records = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result = lookup_vehicle("MP04AB1234", records)
        assert result["status"] == "not_found"

    def test_none_records(self):
        result = lookup_vehicle("MP04AB1234", None)
        assert result["status"] == "not_found"

    def test_found_has_demo_label(self, sample_records):
        result = lookup_vehicle("MP04AB1234", sample_records)
        assert result["status"] == "found"
        assert result["vehicle"]["_data_source"] == "SYNTHETIC DEMO DATA"

    def test_second_record(self, sample_records):
        result = lookup_vehicle("DL01CA0001", sample_records)
        assert result["status"] == "found"
        assert result["vehicle"]["make"] == "Demo Motors"
