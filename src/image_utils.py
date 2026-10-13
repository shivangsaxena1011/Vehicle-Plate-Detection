"""Image validation, preprocessing and enhancement utilities for ANPR.

All functions work on numpy arrays in RGB format unless stated otherwise.
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------- constants ----------
MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024          # 20 MB
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
MIN_DIMENSION: int = 32                            # reject tiny images
MAX_DIMENSION: int = 8192                          # reject enormous images
PLATE_MIN_WIDTH: int = 40                          # minimum plate crop width (px)
PLATE_UPSCALE_TARGET_WIDTH: int = 500              # upscale target for small crops


# ---------- validation ----------

def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """Check whether *filename* has an allowed image extension.

    Returns (ok, message).
    """
    if not filename:
        return False, "No filename provided."
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    return True, "OK"


def validate_file_size(data: bytes | io.BytesIO, max_bytes: int = MAX_UPLOAD_BYTES) -> Tuple[bool, str]:
    """Reject files exceeding *max_bytes*."""
    if isinstance(data, (bytes, bytearray)):
        size = len(data)
    elif hasattr(data, "seek") and hasattr(data, "tell"):
        pos = data.tell()
        data.seek(0, 2)
        size = data.tell()
        data.seek(pos)
    else:
        return True, "OK"  # cannot determine size; let decode catch bad data
    if size > max_bytes:
        return False, f"File size ({size / 1024 / 1024:.1f} MB) exceeds the {max_bytes / 1024 / 1024:.0f} MB limit."
    if size == 0:
        return False, "File is empty (0 bytes)."
    return True, "OK"


def decode_image(data: bytes | io.BytesIO) -> Tuple[Optional[np.ndarray], str]:
    """Decode raw bytes into an RGB numpy array.

    Returns (image_or_None, message).
    """
    try:
        if isinstance(data, io.BytesIO):
            data.seek(0)
            raw = data.read()
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            return None, "Unsupported data type for image decoding."

        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == 0:
            return None, "Image data is empty after reading."

        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, "Could not decode the image. The file may be corrupt or in an unsupported format."

        h, w = bgr.shape[:2]
        if h < MIN_DIMENSION or w < MIN_DIMENSION:
            return None, f"Image is too small ({w}×{h}). Minimum dimension is {MIN_DIMENSION} px."
        if h > MAX_DIMENSION or w > MAX_DIMENSION:
            return None, f"Image is too large ({w}×{h}). Maximum dimension is {MAX_DIMENSION} px."

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, "OK"
    except Exception as exc:
        logger.exception("Image decode error")
        return None, f"Image decode error: {exc}"


def pil_to_rgb(pil_image) -> np.ndarray:
    """Convert a PIL Image to an RGB numpy array."""
    import numpy as _np
    return _np.array(pil_image.convert("RGB"))


# ---------- preprocessing ----------

def enhance_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance a cropped plate region for better OCR.

    Pipeline: upscale small crops → grayscale → CLAHE → light denoise.
    Returns a single-channel (grayscale) enhanced image.
    """
    h, w = crop.shape[:2]

    # 1. Upscale small crops
    if w < PLATE_UPSCALE_TARGET_WIDTH:
        scale = min(4.0, PLATE_UPSCALE_TARGET_WIDTH / max(w, 1))
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # 2. Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop

    # 3. CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Light bilateral denoise — preserves edges better than Gaussian
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

    return enhanced


def deskew_plate(crop: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Attempt to correct small rotations in a plate crop.

    Uses the minimum-area rotated rectangle of contours.
    Returns the deskewed image or the original if angle detection fails.
    """
    try:
        if len(crop.shape) == 3:
            work = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            work = crop.copy()

        _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop

        # Use the largest contour
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[1][0]  # width angle

        # minAreaRect returns angles in [-90, 0); normalize
        angle = rect[2]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        if abs(angle) > max_angle or abs(angle) < 0.5:
            return crop

        h, w = crop.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            crop, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception:
        logger.debug("Deskew failed; returning original crop", exc_info=True)
        return crop


def draw_boxes(image: np.ndarray, boxes: list[Tuple[int, int, int, int]],
               color: Tuple[int, int, int] = (0, 200, 80),
               thickness: int = 3) -> np.ndarray:
    """Draw bounding boxes on an RGB image. Returns a copy."""
    annotated = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        # Add label background
        label = "PLATE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated
