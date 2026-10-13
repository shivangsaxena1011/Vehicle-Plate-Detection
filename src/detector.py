"""YOLO-based license plate detector.

Wraps ultralytics YOLO for plate-region detection.  Falls back gracefully
when weights or the ultralytics package are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try importing ultralytics; set to None if unavailable.
try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    _YOLO = None


@dataclass
class Detection:
    """A single detected plate region."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    crop: np.ndarray  # RGB crop

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectorResult:
    """Result of running the plate detector on one image."""
    detections: List[Detection] = field(default_factory=list)
    model_loaded: bool = False
    error: Optional[str] = None
    model_path: Optional[str] = None

    @property
    def found(self) -> bool:
        return len(self.detections) > 0


class PlateDetector:
    """License-plate detector backed by a YOLO model.

    Parameters
    ----------
    model_path : str or Path
        Path to a YOLO ``.pt`` file trained for license-plate detection.
    confidence : float
        Minimum detection confidence threshold.
    """

    def __init__(self, model_path: str = "models/license_plate.pt",
                 confidence: float = 0.40):
        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self._model = None
        self._load_error: Optional[str] = None
        self._load()

    # ---- loading ----

    def _load(self) -> None:
        path = Path(self.model_path)
        if _YOLO is None:
            self._load_error = (
                "The 'ultralytics' package is not installed. "
                "Install it via: pip install ultralytics"
            )
            logger.warning(self._load_error)
            return

        if not path.is_file():
            self._load_error = (
                f"No plate-detection model found at '{path}'. "
                "Download a license-plate-specific YOLO .pt file and place it there, "
                "or enter a valid path in the sidebar."
            )
            logger.warning(self._load_error)
            return

        try:
            self._model = _YOLO(str(path))
            logger.info("Loaded plate detector from %s", path)
        except Exception as exc:
            self._load_error = f"Failed to load YOLO model from '{path}': {exc}"
            logger.error(self._load_error, exc_info=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ---- detection ----

    def detect(self, rgb: np.ndarray) -> DetectorResult:
        """Run plate detection on an RGB image.

        Returns a DetectorResult with zero or more Detection objects.
        """
        result = DetectorResult(model_path=self.model_path)

        if self._model is None:
            result.error = self._load_error or "Detector not loaded."
            return result

        result.model_loaded = True

        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            predictions = self._model.predict(
                source=bgr,
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:
            result.error = f"Detection inference failed: {exc}"
            logger.error(result.error, exc_info=True)
            return result

        if not predictions:
            return result

        pred = predictions[0]
        if pred.boxes is None:
            return result

        h, w = rgb.shape[:2]

        for box in pred.boxes:
            try:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = coords
                # Clamp to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box

                conf = float(box.conf[0].cpu().numpy())
                crop = rgb[y1:y2, x1:x2].copy()

                result.detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf, crop=crop,
                ))
            except Exception as exc:
                logger.warning("Skipping malformed box: %s", exc)

        # Sort by confidence descending
        result.detections.sort(key=lambda d: d.confidence, reverse=True)
        return result
