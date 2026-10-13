"""Tests for src.anpr — end-to-end pipeline with mocked detector and OCR."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

from src.anpr import ANPRPipeline, ANPRResult, PlateReading
from src.detector import DetectorResult, Detection
from src.ocr_engine import OCRResult, OCRItem


def _make_test_image(w: int = 640, h: int = 480) -> Image.Image:
    """Create a simple test PIL image."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestANPRResult:
    def test_best_plate(self):
        r = ANPRResult(plates=[
            PlateReading(text="ABC", ocr_confidence=0.5),
            PlateReading(text="DEF", ocr_confidence=0.9),
        ])
        assert r.best_plate.text == "DEF"

    def test_best_plate_empty(self):
        r = ANPRResult()
        assert r.best_plate is None

    def test_to_dict(self):
        r = ANPRResult(
            plates=[PlateReading(text="MP04AB1234", ocr_confidence=0.8)],
            detection_method="test",
            status="text_recognized",
        )
        d = r.to_dict()
        assert d["plate_text"] == "MP04AB1234"
        assert d["ocr_confidence"] == pytest.approx(0.8)
        assert d["detection_method"] == "test"


class TestPipelineWithMocks:
    """Test the pipeline with fully mocked detector and OCR engine."""

    def _make_pipeline_with_mocks(self, det_result: DetectorResult, ocr_result: OCRResult):
        """Create an ANPRPipeline with mocked sub-components."""
        pipeline = ANPRPipeline.__new__(ANPRPipeline)
        pipeline.model_path = "models/test.pt"
        pipeline.detection_confidence = 0.4
        pipeline.allow_full_image_ocr = True

        mock_detector = MagicMock()
        mock_detector.is_loaded = det_result.model_loaded
        mock_detector.detect.return_value = det_result

        mock_ocr = MagicMock()
        mock_ocr.is_ready = True
        mock_ocr.read.return_value = ocr_result

        pipeline._detector = mock_detector
        pipeline._ocr = mock_ocr
        return pipeline

    def test_successful_detection_and_ocr(self):
        crop = np.zeros((40, 120, 3), dtype=np.uint8)
        det = DetectorResult(
            detections=[Detection(10, 20, 130, 60, 0.92, crop)],
            model_loaded=True,
        )
        ocr = OCRResult(items=[
            OCRItem("MH 12 DE 1433", "MH12DE1433", 0.88, True),
        ])
        pipeline = self._make_pipeline_with_mocks(det, ocr)
        result = pipeline.process(_make_test_image())

        assert result.status == "text_recognized"
        assert len(result.plates) == 1
        assert result.plates[0].text == "MH12DE1433"
        assert result.plates[0].ocr_confidence == pytest.approx(0.88)
        assert result.detection_method.startswith("YOLO")

    def test_detection_but_ocr_fails(self):
        crop = np.zeros((40, 120, 3), dtype=np.uint8)
        det = DetectorResult(
            detections=[Detection(10, 20, 130, 60, 0.85, crop)],
            model_loaded=True,
        )
        ocr = OCRResult(error="OCR crashed")
        pipeline = self._make_pipeline_with_mocks(det, ocr)
        result = pipeline.process(_make_test_image())

        assert result.status in ("not_recognized", "plate_detected_but_unreadable")
        assert any("OCR" in n for n in result.notes)

    def test_no_detector_fallback_ocr(self):
        det = DetectorResult(
            model_loaded=False,
            error="No model file",
        )
        ocr = OCRResult(items=[
            OCRItem("DL01CA0001", "DL01CA0001", 0.75, True),
        ])
        pipeline = self._make_pipeline_with_mocks(det, ocr)
        result = pipeline.process(_make_test_image())

        assert "fallback" in result.detection_method.lower()
        assert len(result.plates) >= 1

    def test_no_detector_no_fallback(self):
        det = DetectorResult(model_loaded=False, error="No model")
        ocr = OCRResult()
        pipeline = self._make_pipeline_with_mocks(det, ocr)
        pipeline.allow_full_image_ocr = False
        result = pipeline.process(_make_test_image())

        assert result.status == "no_detector"
        assert any("No model" in n for n in result.notes)

    def test_no_plates_recognized(self):
        det = DetectorResult(model_loaded=True)
        ocr = OCRResult()
        pipeline = self._make_pipeline_with_mocks(det, ocr)
        pipeline.allow_full_image_ocr = True
        result = pipeline.process(_make_test_image())

        assert result.plates == [] or all(p.text == "" for p in result.plates)

    def test_multiple_detections(self):
        crop1 = np.zeros((40, 120, 3), dtype=np.uint8)
        crop2 = np.zeros((40, 120, 3), dtype=np.uint8)
        det = DetectorResult(
            detections=[
                Detection(10, 20, 130, 60, 0.95, crop1),
                Detection(200, 100, 350, 140, 0.80, crop2),
            ],
            model_loaded=True,
        )

        call_count = [0]
        def mock_read(img, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return OCRResult(items=[OCRItem("MP04AB1234", "MP04AB1234", 0.90, True)])
            return OCRResult(items=[OCRItem("DL01CA0001", "DL01CA0001", 0.70, True)])

        pipeline = self._make_pipeline_with_mocks(det, OCRResult())
        pipeline._ocr.read.side_effect = mock_read
        result = pipeline.process(_make_test_image())

        assert len(result.plates) == 2

    def test_annotated_image_returned(self):
        det = DetectorResult(model_loaded=True)
        ocr = OCRResult()
        pipeline = self._make_pipeline_with_mocks(det, ocr)
        result = pipeline.process(_make_test_image())

        assert result.annotated_image is not None
        assert isinstance(result.annotated_image, Image.Image)
