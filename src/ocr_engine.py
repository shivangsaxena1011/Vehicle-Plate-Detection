"""OCR engine for license plate text recognition.

Uses EasyOCR by default.  Handles initialization failures gracefully.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .image_utils import enhance_plate_crop, deskew_plate

logger = logging.getLogger(__name__)

# Try importing EasyOCR; set to None if unavailable.
try:
    import easyocr as _easyocr
except ImportError:
    _easyocr = None


# ---------- Indian plate format ----------

# Common Indian registration patterns (not exhaustive):
#   XX 00 XX 0000  — state(2) district(2) series(1-3) number(1-4)
# We use a loose regex that accepts most real plates without being so strict
# that minor OCR mistakes cause rejection.
_INDIAN_PLATE_RE = re.compile(
    r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$"
)


def normalize_plate_text(text: str) -> str:
    """Normalize raw OCR output to a clean registration string.

    Operations:
    - Converts input text to uppercase.
    - Strips spaces, hyphens, periods, and all non-alphanumeric characters.
    - Preserves letters (A-Z) and digits (0-9) without positional substitutions,
      avoiding miscorrections when vehicle plate format is ambiguous.
    """
    text = (text or "").upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", text)
    return cleaned


def looks_like_indian_plate(text: str) -> bool:
    """Heuristic check for Indian registration format.

    This is NOT proof of registration.  It merely filters implausible OCR noise.
    """
    if not text or len(text) < 5 or len(text) > 13:
        return False
    return bool(_INDIAN_PLATE_RE.match(text))


@dataclass
class OCRItem:
    """A single OCR reading."""
    raw_text: str
    normalized_text: str
    confidence: Optional[float] = None
    plausible_indian: bool = False


@dataclass
class OCRResult:
    """Result of running OCR on one image region."""
    items: List[OCRItem] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def best(self) -> Optional[OCRItem]:
        """Return the highest-confidence plausible item, or just the best item."""
        plausible = [i for i in self.items if i.plausible_indian]
        pool = plausible if plausible else self.items
        if not pool:
            return None
        return max(pool, key=lambda i: i.confidence if i.confidence is not None else 0.0)


class OCREngine:
    """EasyOCR-based text recognition for license plates.

    Parameters
    ----------
    languages : list[str]
        Languages for EasyOCR.  Default ``["en"]`` covers Latin-alphabet Indian plates.
    gpu : bool
        Whether to attempt GPU acceleration.
    """

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None
        self._init_error: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        if _easyocr is None:
            self._init_error = (
                "EasyOCR is not installed. Install via: pip install easyocr"
            )
            logger.warning(self._init_error)
            return
        try:
            self._reader = _easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR reader initialized (languages=%s, gpu=%s)",
                        self.languages, self.gpu)
        except Exception as exc:
            self._init_error = f"EasyOCR initialization failed: {exc}"
            logger.error(self._init_error, exc_info=True)

    @property
    def is_ready(self) -> bool:
        return self._reader is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def read(self, image: np.ndarray, enhance: bool = True,
             attempt_deskew: bool = True) -> OCRResult:
        """Run OCR on an image region (RGB or grayscale).

        Parameters
        ----------
        image : np.ndarray
            The image crop to read.
        enhance : bool
            Apply CLAHE / denoise preprocessing.
        attempt_deskew : bool
            Try to correct small rotations before OCR.

        Returns
        -------
        OCRResult with zero or more OCRItem entries.
        """
        result = OCRResult()

        if self._reader is None:
            result.error = self._init_error or "OCR engine not loaded."
            return result

        try:
            processed = image.copy()

            # Deskew before enhancement
            if attempt_deskew:
                processed = deskew_plate(processed)

            # Enhance for OCR
            if enhance:
                processed = enhance_plate_crop(processed)

            detections = self._reader.readtext(processed, detail=1, paragraph=False)
        except Exception as exc:
            result.error = f"OCR processing failed: {exc}"
            logger.error(result.error, exc_info=True)
            return result

        for item in detections:
            if len(item) < 2:
                continue
            raw = str(item[1])
            normalized = normalize_plate_text(raw)
            try:
                conf = float(item[2]) if len(item) >= 3 else None
            except (TypeError, ValueError):
                conf = None

            if not normalized:
                continue

            result.items.append(OCRItem(
                raw_text=raw,
                normalized_text=normalized,
                confidence=conf,
                plausible_indian=looks_like_indian_plate(normalized),
            ))

        # Sort by confidence descending
        result.items.sort(
            key=lambda i: i.confidence if i.confidence is not None else 0.0,
            reverse=True,
        )
        return result
