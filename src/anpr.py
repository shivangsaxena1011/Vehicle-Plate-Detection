"""End-to-end ANPR pipeline.

Orchestrates detection → cropping → preprocessing → OCR → normalization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .detector import PlateDetector, DetectorResult
from .image_utils import draw_boxes, enhance_plate_crop
from .ocr_engine import OCREngine, OCRResult, OCRItem

logger = logging.getLogger(__name__)


@dataclass
class PlateReading:
    """One recognized plate from the image."""
    text: str
    ocr_confidence: Optional[float] = None
    detector_confidence: Optional[float] = None
    crop: Optional[np.ndarray] = None
    plausible_indian: bool = False


@dataclass
class ANPRResult:
    """Full pipeline result for one image."""
    plates: List[PlateReading] = field(default_factory=list)
    annotated_image: Optional[Image.Image] = None
    detection_method: str = "none"
    status: str = "not_processed"
    notes: List[str] = field(default_factory=list)
    detector_loaded: bool = False

    @property
    def best_plate(self) -> Optional[PlateReading]:
        """Return the plate reading with highest OCR confidence."""
        if not self.plates:
            return None
        return max(
            self.plates,
            key=lambda p: p.ocr_confidence if p.ocr_confidence is not None else 0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for display / logging (no numpy arrays)."""
        best = self.best_plate
        return {
            "plate_text": best.text if best else "",
            "ocr_confidence": best.ocr_confidence if best else None,
            "detector_confidence": best.detector_confidence if best else None,
            "detection_method": self.detection_method,
            "status": self.status,
            "notes": "; ".join(self.notes) if self.notes else None,
            "num_plates": len(self.plates),
            "detector_loaded": self.detector_loaded,
        }


class ANPRPipeline:
    """Main ANPR pipeline.

    Parameters
    ----------
    model_path : str
        Path to a YOLO .pt model trained for license-plate detection.
    detection_confidence : float
        Minimum detection confidence.
    allow_full_image_ocr : bool
        If True and no detector is available, run OCR on the full image
        as a limited fallback.
    """

    def __init__(
        self,
        model_path: str = "models/license_plate.pt",
        detection_confidence: float = 0.40,
        allow_full_image_ocr: bool = True,
    ):
        self.model_path = model_path
        self.detection_confidence = detection_confidence
        self.allow_full_image_ocr = allow_full_image_ocr

        # Initialize sub-components
        self._detector = PlateDetector(
            model_path=model_path,
            confidence=detection_confidence,
        )
        self._ocr = OCREngine(languages=["en"], gpu=False)

    @property
    def detector_loaded(self) -> bool:
        return self._detector.is_loaded

    @property
    def ocr_ready(self) -> bool:
        return self._ocr.is_ready

    def process(self, pil_image: Image.Image) -> ANPRResult:
        """Run the full pipeline on a PIL Image.

        Returns an ANPRResult with detected plates, annotated image, and status.
        """
        result = ANPRResult()
        result.detector_loaded = self._detector.is_loaded

        # Convert to RGB numpy array
        rgb = np.array(pil_image.convert("RGB"))

        # ── Step 1: Detection ──
        det_result: DetectorResult = self._detector.detect(rgb)
        boxes = [d.box for d in det_result.detections]

        if det_result.found:
            result.detection_method = "YOLO plate detection + EasyOCR"
            result = self._process_detections(rgb, det_result, result)
        elif self.allow_full_image_ocr:
            result.detection_method = "Full-image OCR fallback (no plate detector)"
            if det_result.error:
                result.notes.append(det_result.error)
            result.notes.append(
                "⚠️ Running OCR on the entire image because no plate-detection "
                "model is available. This is a limited fallback — not reliable ANPR."
            )
            result = self._process_full_image(rgb, result)
        else:
            result.detection_method = "none"
            result.status = "no_detector"
            if det_result.error:
                result.notes.append(det_result.error)
            result.notes.append("No plate detected and full-image OCR fallback is disabled.")

        # ── Step 2: Annotated image ──
        if boxes:
            annotated = draw_boxes(rgb, boxes)
        else:
            annotated = rgb.copy()
        result.annotated_image = Image.fromarray(annotated)

        # ── Step 3: Final status ──
        if result.plates:
            result.status = "text_recognized"
        elif det_result.found:
            result.status = "plate_detected_but_unreadable"
        elif not det_result.model_loaded and not self.allow_full_image_ocr:
            result.status = "no_detector"
        else:
            result.status = "not_recognized"

        return result

    def _process_detections(
        self, rgb: np.ndarray, det_result: DetectorResult, result: ANPRResult
    ) -> ANPRResult:
        """OCR each detected plate region."""
        for detection in det_result.detections:
            ocr_result: OCRResult = self._ocr.read(detection.crop)

            if ocr_result.error:
                result.notes.append(f"OCR error on crop: {ocr_result.error}")
                continue

            best = ocr_result.best
            if best:
                result.plates.append(PlateReading(
                    text=best.normalized_text,
                    ocr_confidence=best.confidence,
                    detector_confidence=detection.confidence,
                    crop=detection.crop,
                    plausible_indian=best.plausible_indian,
                ))
            else:
                # Detection found a region but OCR couldn't read it
                result.plates.append(PlateReading(
                    text="",
                    detector_confidence=detection.confidence,
                    crop=detection.crop,
                ))
                result.notes.append(
                    "A plate region was detected but OCR could not extract text from it."
                )

        return result

    def _process_full_image(
        self, rgb: np.ndarray, result: ANPRResult
    ) -> ANPRResult:
        """Run OCR on the full image as a fallback."""
        ocr_result: OCRResult = self._ocr.read(rgb, enhance=True, attempt_deskew=False)

        if ocr_result.error:
            result.notes.append(f"OCR error: {ocr_result.error}")
            return result

        # Collect all plausible plate readings
        for item in ocr_result.items:
            if item.plausible_indian or (item.confidence and item.confidence > 0.3):
                result.plates.append(PlateReading(
                    text=item.normalized_text,
                    ocr_confidence=item.confidence,
                    plausible_indian=item.plausible_indian,
                ))

        # If nothing plausible, take the best raw reading
        if not result.plates and ocr_result.items:
            best = ocr_result.items[0]
            result.plates.append(PlateReading(
                text=best.normalized_text,
                ocr_confidence=best.confidence,
                plausible_indian=best.plausible_indian,
            ))

        return result
