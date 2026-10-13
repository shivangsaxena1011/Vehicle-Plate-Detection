"""Tests for src.vehicle_provider — demo CSV provider and API stub."""

import os
import tempfile

import pandas as pd
import pytest

from src.vehicle_provider import (
    DemoCSVProvider,
    AuthorizedAPIProvider,
    VehicleLookupResult,
)
from src.database import REQUIRED_COLUMNS


class TestDemoCSVProvider:
    def test_lookup_found(self):
        provider = DemoCSVProvider(csv_path="data/vehicles.csv")
        result = provider.lookup("MP04AB1234")
        assert result.status == "found"
        assert result.is_demo_data is True
        assert result.vehicle is not None
        assert result.vehicle["_data_source"] == "SYNTHETIC DEMO DATA"

    def test_lookup_not_found(self):
        provider = DemoCSVProvider(csv_path="data/vehicles.csv")
        result = provider.lookup("ZZ99ZZ9999")
        assert result.status == "not_found"
        assert "does NOT mean" in result.message

    def test_lookup_normalized(self):
        provider = DemoCSVProvider(csv_path="data/vehicles.csv")
        result = provider.lookup("mp 04 ab 1234")
        assert result.status == "found"

    def test_missing_csv(self):
        provider = DemoCSVProvider(csv_path="nonexistent.csv")
        result = provider.lookup("MP04AB1234")
        assert result.status == "not_found"

    def test_provider_name(self):
        provider = DemoCSVProvider()
        assert "Demo" in provider.provider_name or "CSV" in provider.provider_name

    def test_malformed_csv(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            f.write("bad_col1,bad_col2\nval1,val2\n")
            f.close()
            provider = DemoCSVProvider(csv_path=f.name)
            result = provider.lookup("TEST")
            # Should handle error gracefully
            assert result.status in ("not_found", "error")
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass


class TestAuthorizedAPIProvider:
    def test_always_unavailable(self):
        provider = AuthorizedAPIProvider()
        result = provider.lookup("MP04AB1234")
        assert result.status == "provider_unavailable"
        assert "no authorized" in result.message.lower()

    def test_provider_name(self):
        provider = AuthorizedAPIProvider()
        assert "NOT CONFIGURED" in provider.provider_name
