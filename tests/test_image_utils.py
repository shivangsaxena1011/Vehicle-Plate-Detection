"""Tests for src.image_utils — validation, decoding, preprocessing."""

import io
import struct

import cv2
import numpy as np
import pytest

from src.image_utils import (
    validate_file_extension,
    validate_file_size,
    decode_image,
    enhance_plate_crop,
    deskew_plate,
    draw_boxes,
    MAX_UPLOAD_BYTES,
)


# ── Extension validation ──

class TestValidateFileExtension:
    @pytest.mark.parametrize("name", ["photo.jpg", "photo.jpeg", "photo.png", "photo.webp"])
    def test_allowed(self, name):
        ok, msg = validate_file_extension(name)
        assert ok is True

    @pytest.mark.parametrize("name", ["doc.pdf", "archive.zip", "script.py", "photo.bmp"])
    def test_disallowed(self, name):
        ok, msg = validate_file_extension(name)
        assert ok is False

    def test_empty_filename(self):
        ok, msg = validate_file_extension("")
        assert ok is False

    def test_case_insensitive(self):
        ok, _ = validate_file_extension("PHOTO.JPG")
        assert ok is True


# ── Size validation ──

class TestValidateFileSize:
    def test_small_file(self):
        ok, _ = validate_file_size(b"x" * 100)
        assert ok is True

    def test_empty_file(self):
        ok, _ = validate_file_size(b"")
        assert ok is False

    def test_oversized_bytes(self):
        ok, msg = validate_file_size(b"x" * (MAX_UPLOAD_BYTES + 1))
        assert ok is False
        assert "exceeds" in msg.lower() or "limit" in msg.lower()

    def test_bytesio(self):
        buf = io.BytesIO(b"x" * 500)
        ok, _ = validate_file_size(buf)
        assert ok is True


# ── Image decoding ──

def _make_test_png(width: int = 100, height: int = 80) -> bytes:
    """Create a minimal valid PNG in memory."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return encoded.tobytes()


class TestDecodeImage:
    def test_valid_png(self):
        data = _make_test_png(200, 150)
        img, msg = decode_image(data)
        assert img is not None
        assert img.shape == (150, 200, 3)

    def test_valid_bytesio(self):
        data = _make_test_png()
        img, msg = decode_image(io.BytesIO(data))
        assert img is not None

    def test_corrupt_data(self):
        img, msg = decode_image(b"\x00\x01\x02\x03\x04")
        assert img is None
        assert "corrupt" in msg.lower() or "decode" in msg.lower()

    def test_empty_data(self):
        img, msg = decode_image(b"")
        assert img is None

    def test_too_small(self):
        # 10x10 is below MIN_DIMENSION=32
        data = _make_test_png(10, 10)
        img, msg = decode_image(data)
        assert img is None
        assert "small" in msg.lower()


# ── Preprocessing ──

class TestEnhancePlateCrop:
    def test_returns_grayscale(self):
        crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
        result = enhance_plate_crop(crop)
        assert len(result.shape) == 2  # grayscale

    def test_upscales_small_crop(self):
        crop = np.random.randint(0, 255, (20, 60, 3), dtype=np.uint8)
        result = enhance_plate_crop(crop)
        assert result.shape[1] >= 60  # should be upscaled

    def test_handles_grayscale_input(self):
        crop = np.random.randint(0, 255, (40, 120), dtype=np.uint8)
        result = enhance_plate_crop(crop)
        assert len(result.shape) == 2


class TestDeskewPlate:
    def test_small_angle_corrected(self):
        crop = np.random.randint(0, 255, (60, 200, 3), dtype=np.uint8)
        result = deskew_plate(crop)
        assert result.shape[:2] == crop.shape[:2]

    def test_returns_same_shape(self):
        crop = np.random.randint(0, 255, (50, 150), dtype=np.uint8)
        result = deskew_plate(crop)
        assert result.shape == crop.shape


class TestDrawBoxes:
    def test_draws_boxes(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        boxes = [(10, 10, 100, 50), (200, 100, 350, 180)]
        result = draw_boxes(img, boxes)
        # Should return a copy, not modify original
        assert not np.array_equal(result, img)
        assert result.shape == img.shape

    def test_empty_boxes(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_boxes(img, [])
        assert np.array_equal(result, img)


# ── Video extension validation ──

class TestValidateVideoExtension:
    def test_allowed_video_extensions(self):
        from src.video import validate_video_extension
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            ok, msg = validate_video_extension(f"sample_traffic{ext}")
            assert ok is True
            assert msg == "OK"

    def test_disallowed_video_extensions(self):
        from src.video import validate_video_extension
        for ext in [".exe", ".sh", ".jpg", ".flv", ".wmv"]:
            ok, msg = validate_video_extension(f"video{ext}")
            assert ok is False
            assert "Unsupported" in msg

    def test_empty_filename(self):
        from src.video import validate_video_extension
        ok, msg = validate_video_extension("")
        assert ok is False
