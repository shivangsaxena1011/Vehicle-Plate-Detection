"""Video and webcam processing for ANPR.

Processes video files frame-by-frame with configurable sampling,
duplicate suppression, and timestamp tracking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv"}
MAX_VIDEO_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB


def validate_video_extension(filename: str) -> Tuple[bool, str]:
    """Validate video file extension against allowed formats."""
    if not filename:
        return False, "No filename provided."
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return False, (
            f"Unsupported video format '{suffix}'. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )
    return True, "OK"


@dataclass
class FrameResult:
    """Result of processing a single video frame."""
    frame_number: int
    timestamp_sec: float
    plates: List[str] = field(default_factory=list)
    confidences: List[Optional[float]] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class VideoProcessingConfig:
    """Configuration for video processing."""
    frame_skip: int = 5              # Process every Nth frame
    max_frames: int = 500            # Maximum frames to process
    duplicate_window_sec: float = 3.0  # Suppress same plate within N seconds
    min_confidence: float = 0.3      # Minimum OCR confidence to report


class VideoProcessor:
    """Process video files or webcam streams for license plates.

    Parameters
    ----------
    pipeline : ANPRPipeline
        A configured ANPR pipeline instance.
    config : VideoProcessingConfig
        Processing configuration.
    """

    def __init__(self, pipeline, config: Optional[VideoProcessingConfig] = None):
        self.pipeline = pipeline
        self.config = config or VideoProcessingConfig()
        self._recent_plates: dict[str, float] = {}  # plate -> last_seen_timestamp

    def _is_duplicate(self, plate: str, timestamp: float) -> bool:
        """Check if this plate was seen recently."""
        if plate in self._recent_plates:
            last_seen = self._recent_plates[plate]
            if timestamp - last_seen < self.config.duplicate_window_sec:
                return True
        return False

    def _record_plate(self, plate: str, timestamp: float) -> None:
        """Record a plate sighting."""
        self._recent_plates[plate] = timestamp

    def process_video_file(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int, Optional[FrameResult]], None]] = None,
    ) -> List[FrameResult]:
        """Process a video file frame-by-frame.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        progress_callback : callable, optional
            Called with (current_frame, total_frames, frame_result_or_None).

        Returns
        -------
        list[FrameResult]
            Results for frames where plates were found.
        """
        self._recent_plates.clear()
        results: List[FrameResult] = []

        if not os.path.isfile(video_path):
            logger.error("Video file not found: %s", video_path)
            return results

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file via OpenCV: %s", video_path)
            return results

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_number = 0
        processed_count = 0

        try:
            while cap.isOpened() and processed_count < self.config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # Skip frames per config
                if frame_number % self.config.frame_skip != 0:
                    continue

                processed_count += 1
                timestamp = frame_number / fps

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    anpr_result = self.pipeline.process(pil_img)

                    frame_res = FrameResult(
                        frame_number=frame_number,
                        timestamp_sec=round(timestamp, 2),
                    )

                    if anpr_result.plates:
                        for plate_reading in anpr_result.plates:
                            if not plate_reading.text:
                                continue
                            if (plate_reading.ocr_confidence is not None and
                                    plate_reading.ocr_confidence < self.config.min_confidence):
                                continue
                            if self._is_duplicate(plate_reading.text, timestamp):
                                continue

                            self._record_plate(plate_reading.text, timestamp)
                            frame_res.plates.append(plate_reading.text)
                            frame_res.confidences.append(plate_reading.ocr_confidence)

                    if frame_res.plates:
                        if anpr_result.annotated_image:
                            frame_res.annotated_frame = np.array(anpr_result.annotated_image)
                        results.append(frame_res)

                    if progress_callback:
                        progress_callback(frame_number, total_frames, frame_res if frame_res.plates else None)

                except Exception as exc:
                    logger.warning("Error processing frame %d: %s", frame_number, exc, exc_info=True)
                    if progress_callback:
                        err_res = FrameResult(
                            frame_number=frame_number,
                            timestamp_sec=round(timestamp, 2),
                            error="Frame processing error",
                        )
                        progress_callback(frame_number, total_frames, err_res)

        finally:
            cap.release()

        return results

    def process_frames_generator(
        self, video_path: str
    ) -> Generator[Tuple[int, int, Optional[FrameResult]], None, None]:
        """Generator that yields (frame_number, total_frames, result) tuples.

        Useful for Streamlit progress bars.
        """
        self._recent_plates.clear()

        if not os.path.isfile(video_path):
            logger.error("Video file not found: %s", video_path)
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file via OpenCV: %s", video_path)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_number = 0
        processed_count = 0

        try:
            while cap.isOpened() and processed_count < self.config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                if frame_number % self.config.frame_skip != 0:
                    continue

                processed_count += 1
                timestamp = frame_number / fps

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    anpr_result = self.pipeline.process(pil_img)

                    frame_res = FrameResult(
                        frame_number=frame_number,
                        timestamp_sec=round(timestamp, 2),
                    )

                    if anpr_result.plates:
                        for pr in anpr_result.plates:
                            if not pr.text:
                                continue
                            if (pr.ocr_confidence is not None and
                                    pr.ocr_confidence < self.config.min_confidence):
                                continue
                            if self._is_duplicate(pr.text, timestamp):
                                continue
                            self._record_plate(pr.text, timestamp)
                            frame_res.plates.append(pr.text)
                            frame_res.confidences.append(pr.ocr_confidence)

                    if frame_res.plates:
                        if anpr_result.annotated_image:
                            frame_res.annotated_frame = np.array(anpr_result.annotated_image)

                    yield (frame_number, total_frames, frame_res if frame_res.plates else None)

                except Exception as exc:
                    logger.warning("Frame %d processing failed: %s", frame_number, exc, exc_info=True)
                    yield (frame_number, total_frames, FrameResult(
                        frame_number=frame_number,
                        timestamp_sec=round(timestamp, 2),
                        error="Processing error on frame",
                    ))
        finally:
            cap.release()
