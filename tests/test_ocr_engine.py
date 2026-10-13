"""Tests for src.ocr_engine — OCR text normalization and format checks.

Heavy OCR model tests are mocked so the suite runs without downloading
EasyOCR models or requiring a GPU.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ocr_engine import (
    normalize_plate_text,
    looks_like_indian_plate,
    OCREngine,
    OCRResult,
    OCRItem,
)


# ── Text normalization ──

class TestNormalizePlateText:
    def test_basic(self):
        assert normalize_plate_text("MP 04 AB 1234") == "MP04AB1234"

    def test_lowercase(self):
        assert normalize_plate_text("mp04ab1234") == "MP04AB1234"

    def test_hyphens(self):
        assert normalize_plate_text("MH-12-DE-1433") == "MH12DE1433"

    def test_dots(self):
        assert normalize_plate_text("KA.05.MH.7777") == "KA05MH7777"

    def test_empty(self):
        assert normalize_plate_text("") == ""

    def test_none(self):
        assert normalize_plate_text(None) == ""

    def test_special_chars(self):
        assert normalize_plate_text("DL·01·CA·0001") == "DL01CA0001"

    def test_preserves_characters_without_unintended_substitutions(self):
        # Verify that letters like 'O', 'I', 'S', 'B' and digits '0', '1', '5', '8'
        # are preserved literally and not blindly replaced
        assert normalize_plate_text("DL 01 SO 180B") == "DL01SO180B"
        assert normalize_plate_text("KA 55 II 0001") == "KA55II0001"


# ── Indian plate format heuristic ──

class TestLooksLikeIndianPlate:
    @pytest.mark.parametrize("plate", [
        "MP04AB1234",
        "DL01CA0001",
        "KA05MH7777",
        "MH12DE1433",
        "TN10AQ2345",
        "RJ14SB6789",
        "UP32GH9012",
    ])
    def test_valid_formats(self, plate):
        assert looks_like_indian_plate(plate) is True

    @pytest.mark.parametrize("plate", [
        "",
        "AB",
        "1234",
        "ABCDEFGHIJKLMN",  # too long
        "1234ABCD",        # starts with digit
    ])
    def test_invalid_formats(self, plate):
        assert looks_like_indian_plate(plate) is False


# ── OCRResult helper ──

class TestOCRResult:
    def test_best_prefers_plausible(self):
        r = OCRResult(items=[
            OCRItem("A", "GARBAGE", confidence=0.9, plausible_indian=False),
            OCRItem("B", "MH12DE1433", confidence=0.7, plausible_indian=True),
        ])
        assert r.best.normalized_text == "MH12DE1433"

    def test_best_highest_confidence_when_no_plausible(self):
        r = OCRResult(items=[
            OCRItem("A", "XYZ", confidence=0.5),
            OCRItem("B", "ABC", confidence=0.8),
        ])
        assert r.best.normalized_text == "ABC"

    def test_best_none_when_empty(self):
        r = OCRResult()
        assert r.best is None


# ── OCR Engine with mocking ──

class TestOCREngineMocked:
    """Test OCR engine with mocked EasyOCR to avoid model downloads."""

    @patch("src.ocr_engine._easyocr", None)
    def test_unavailable_easyocr(self):
        engine = OCREngine.__new__(OCREngine)
        engine.languages = ["en"]
        engine.gpu = False
        engine._reader = None
        engine._init_error = "EasyOCR is not installed."
        assert engine.is_ready is False
        assert "not installed" in engine.init_error

    def test_read_without_reader(self):
        engine = OCREngine.__new__(OCREngine)
        engine.languages = ["en"]
        engine.gpu = False
        engine._reader = None
        engine._init_error = "Test: no reader"
        img = np.zeros((50, 150, 3), dtype=np.uint8)
        result = engine.read(img)
        assert result.error is not None

    def test_read_with_mocked_reader(self):
        engine = OCREngine.__new__(OCREngine)
        engine.languages = ["en"]
        engine.gpu = False
        engine._init_error = None
        mock_reader = MagicMock()
        # Simulate EasyOCR returning detections
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "MH 12 DE 1433", 0.85),
            ([[0, 30], [100, 30], [100, 60], [0, 60]], "INDIA", 0.40),
        ]
        engine._reader = mock_reader
        img = np.zeros((60, 200, 3), dtype=np.uint8)
        result = engine.read(img)
        assert result.error is None
        assert len(result.items) >= 1
        # Best should be the high-confidence Indian plate
        best = result.best
        assert best is not None
        assert best.normalized_text == "MH12DE1433"
        assert best.confidence == pytest.approx(0.85)

    def test_read_handles_ocr_exception(self):
        engine = OCREngine.__new__(OCREngine)
        engine.languages = ["en"]
        engine.gpu = False
        engine._init_error = None
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("OCR crash")
        engine._reader = mock_reader
        img = np.zeros((50, 150, 3), dtype=np.uint8)
        result = engine.read(img)
        assert result.error is not None
        assert "crash" in result.error.lower() or "failed" in result.error.lower()
