"""Evaluation script for ANPR pipeline recognition accuracy.

Requires a labeled test dataset (directory of images with ground-truth
plate annotations).

Usage:
    python evaluate.py --data-dir path/to/test_images --labels path/to/labels.csv

Labels CSV format:
    filename,plate_text
    img001.jpg,MH12DE1433
    img002.jpg,KA05MH7777
    img003.jpg,           (empty = negative sample, no plate present)

Metrics computed and reported:
    - manifest_records: Total rows in the ground-truth manifest
    - processed_images: Number of images successfully loaded and processed
    - missing_images: Number of images in manifest that were not found on disk
    - unreadable_images: Number of images that failed during file read or decoding
    - positive_images: Expected positive plate records in the manifest
    - positive_processed: Expected positive images successfully loaded and processed
    - negative_images: Expected negative records in the manifest
    - negative_processed: Expected negative images successfully loaded and processed
    - positive_recognized: Successful positive recognitions (plate text extracted)
    - exact_matches: Positive recognitions where text exactly matches ground truth
    - recognition_rate_all_positive: positive_recognized / positive_images
      (Recognition rate across ALL expected positive records in manifest)
    - recognition_rate_processed_positive: positive_recognized / positive_processed
      (Recognition rate strictly among successfully processed positive images)
    - exact_match_accuracy_all: exact_matches / positive_images
    - exact_match_accuracy_processed: exact_matches / positive_processed
    - character_error_rate: Levenshtein distance sum / ground-truth character sum
    - false_positives: Erroneously recognized text on processed negative images
    - false_positive_rate: false_positives / negative_processed
    - latency: Mean and median latency per processed image

NOTE: This script measures end-to-end string extraction accuracy.
It does NOT report spatial bounding-box detection recall (mAP / IoU), which
requires ground-truth coordinate annotations [x1, y1, x2, y2].
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def load_labels(labels_path: str) -> Dict[str, str]:
    """Load ground-truth labels from CSV.

    Returns dict mapping filename -> plate_text (normalized, uppercase, alphanumeric only).
    """
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames or "plate_text" not in reader.fieldnames:
            raise ValueError("Labels CSV must have 'filename' and 'plate_text' columns.")
        for row in reader:
            fname = row["filename"].strip()
            plate = "".join(c for c in row["plate_text"].strip().upper() if c.isalnum())
            labels[fname] = plate
    return labels


def compute_evaluation_metrics(
    manifest: Dict[str, str],
    predictions: Dict[str, str],
    missing_images: Optional[Sequence[str]] = None,
    unreadable_images: Optional[Sequence[str]] = None,
    latencies: Optional[Sequence[float]] = None,
    errors: Optional[Sequence[str]] = None,
) -> Dict:
    """Compute comprehensive evaluation metrics safely.

    Parameters
    ----------
    manifest : dict
        Mapping filename -> ground truth string (empty string for negative images).
    predictions : dict
        Mapping filename -> predicted plate string (only for successfully processed images).
    missing_images : sequence of str, optional
        Files listed in manifest but absent on disk.
    unreadable_images : sequence of str, optional
        Files that failed during file read, decode, or processing.
    latencies : sequence of float, optional
        Per-sample execution latencies in seconds.
    errors : sequence of str, optional
        Detailed error log entries.

    Returns
    -------
    dict
        Evaluation metrics with separate tallies for positive/negative,
        all-manifest vs processed rates, and zero-safe denominators.
    """
    missing_set = set(missing_images or [])
    unreadable_set = set(unreadable_images or [])

    manifest_records = len(manifest)
    missing_count = len(missing_set)
    unreadable_count = len(unreadable_set)
    processed_count = len(predictions)

    positive_images = 0
    negative_images = 0
    positive_processed = 0
    negative_processed = 0
    positive_recognized = 0
    exact_matches = 0
    char_errors = 0
    char_total = 0
    false_positives = 0

    for filename, gt_plate in manifest.items():
        is_positive = bool(gt_plate)
        if is_positive:
            positive_images += 1
        else:
            negative_images += 1

        # Check if successfully processed
        if filename in predictions:
            pred = predictions[filename]
            if is_positive:
                positive_processed += 1
                if pred:
                    positive_recognized += 1
                if pred == gt_plate:
                    exact_matches += 1
                char_errors += edit_distance(pred, gt_plate)
                char_total += len(gt_plate)
            else:
                negative_processed += 1
                if pred:
                    false_positives += 1

    # Safe denominator calculations
    rec_rate_all = (positive_recognized / positive_images) if positive_images > 0 else 0.0
    rec_rate_proc = (positive_recognized / positive_processed) if positive_processed > 0 else 0.0

    exact_acc_all = (exact_matches / positive_images) if positive_images > 0 else 0.0
    exact_acc_proc = (exact_matches / positive_processed) if positive_processed > 0 else 0.0

    cer = (char_errors / char_total) if char_total > 0 else 0.0
    fpr = (false_positives / negative_processed) if negative_processed > 0 else 0.0

    mean_lat = float(np.mean(latencies)) if latencies and len(latencies) > 0 else 0.0
    med_lat = float(np.median(latencies)) if latencies and len(latencies) > 0 else 0.0

    return {
        "manifest_records": manifest_records,
        "processed_images": processed_count,
        "missing_images": missing_count,
        "unreadable_images": unreadable_count,
        "positive_images": positive_images,
        "positive_processed": positive_processed,
        "negative_images": negative_images,
        "negative_processed": negative_processed,
        "positive_recognized": positive_recognized,
        "exact_matches": exact_matches,
        "false_positives": false_positives,
        "recognition_rate_all_positive": rec_rate_all,
        "recognition_rate_processed_positive": rec_rate_proc,
        "exact_match_accuracy_all": exact_acc_all,
        "exact_match_accuracy_processed": exact_acc_proc,
        "character_error_rate": cer,
        "false_positive_rate": fpr,
        "mean_latency_sec": mean_lat,
        "median_latency_sec": med_lat,
        "missing_image_list": sorted(list(missing_set)),
        "unreadable_image_list": sorted(list(unreadable_set)),
        "errors": list(errors or []),
    }


def run_evaluation(
    data_dir: str,
    labels_path: str,
    model_path: str = "models/license_plate.pt",
    confidence: float = 0.40,
) -> Dict:
    """Run evaluation on the test set."""
    from src.anpr import ANPRPipeline

    labels = load_labels(labels_path)
    data_path = Path(data_dir)

    if not data_path.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    pipeline = ANPRPipeline(
        model_path=model_path,
        detection_confidence=confidence,
        allow_full_image_ocr=True,
    )

    logger.info("=" * 65)
    logger.info("ANPR Pipeline Recognition Evaluation")
    logger.info("=" * 65)
    logger.info("Data directory:    %s", data_dir)
    logger.info("Labels manifest:   %s (%d records)", labels_path, len(labels))
    logger.info("Model path:        %s (detector_loaded=%s)", model_path, pipeline.detector_loaded)
    logger.info("Confidence thresh: %.2f", confidence)
    logger.info("-" * 65)

    predictions: Dict[str, str] = {}
    missing_images: List[str] = []
    unreadable_images: List[str] = []
    latencies: List[float] = []
    errors: List[str] = []

    for filename, gt_plate in labels.items():
        img_path = data_path / filename
        if not img_path.is_file():
            missing_images.append(filename)
            err_msg = f"Missing file: {filename}"
            errors.append(err_msg)
            logger.warning("  %-25s | MISSING FILE", filename)
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            unreadable_images.append(filename)
            err_msg = f"Cannot read/decode image '{filename}': {exc}"
            errors.append(err_msg)
            logger.warning("  %-25s | UNREADABLE IMAGE (%s)", filename, exc)
            continue

        try:
            start = time.perf_counter()
            result = pipeline.process(pil_img)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            best = result.best_plate
            pred = best.text if best else ""
            predictions[filename] = pred

            is_match = (pred == gt_plate) if gt_plate else (pred == "")
            status_sym = "✅" if is_match else "❌"

            logger.info(
                "  %-20s | GT: %-12s | Pred: %-12s | %s | %.2fs",
                filename,
                gt_plate or "(none)",
                pred or "(none)",
                status_sym,
                elapsed,
            )
        except Exception as exc:
            unreadable_images.append(filename)
            err_msg = f"Processing failure on '{filename}': {exc}"
            errors.append(err_msg)
            logger.warning("  %-20s | PIPELINE ERROR (%s)", filename, exc)

    metrics = compute_evaluation_metrics(
        manifest=labels,
        predictions=predictions,
        missing_images=missing_images,
        unreadable_images=unreadable_images,
        latencies=latencies,
        errors=errors,
    )

    logger.info("-" * 65)
    logger.info("EVALUATION SUMMARY")
    logger.info("-" * 65)
    logger.info("Manifest records:                 %d", metrics["manifest_records"])
    logger.info("Processed images:                 %d", metrics["processed_images"])
    logger.info("Missing images:                   %d", metrics["missing_images"])
    logger.info("Unreadable images:                %d", metrics["unreadable_images"])
    logger.info("Positive records (expected):      %d", metrics["positive_images"])
    logger.info("Positive processed:               %d", metrics["positive_processed"])
    logger.info("Negative records:                 %d", metrics["negative_images"])
    logger.info("Negative processed:               %d", metrics["negative_processed"])
    logger.info("Positive text recognized:         %d", metrics["positive_recognized"])
    logger.info("Exact text matches:               %d", metrics["exact_matches"])
    logger.info("False positives (on negatives):   %d", metrics["false_positives"])
    logger.info("Recognition rate (all positive):  %.1f%%", metrics["recognition_rate_all_positive"] * 100)
    logger.info("Recognition rate (proc positive): %.1f%%", metrics["recognition_rate_processed_positive"] * 100)
    logger.info("Exact match accuracy (all):       %.1f%%", metrics["exact_match_accuracy_all"] * 100)
    logger.info("Exact match accuracy (proc):      %.1f%%", metrics["exact_match_accuracy_processed"] * 100)
    logger.info("Character error rate:             %.3f", metrics["character_error_rate"])
    logger.info("False positive rate:              %.1f%%", metrics["false_positive_rate"] * 100)
    logger.info("Mean latency:                     %.2f s", metrics["mean_latency_sec"])
    logger.info("Median latency:                   %.2f s", metrics["median_latency_sec"])
    if errors:
        logger.info("Errors encountered:               %d", len(errors))
    logger.info("=" * 65)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ANPR pipeline recognition accuracy")
    parser.add_argument("--data-dir", required=True, help="Directory containing test images")
    parser.add_argument("--labels", required=True, help="CSV with filename,plate_text columns")
    parser.add_argument("--model", default="models/license_plate.pt", help="Path to YOLO model")
    parser.add_argument("--confidence", type=float, default=0.40, help="Detection confidence")
    args = parser.parse_args()

    metrics = run_evaluation(
        data_dir=args.data_dir,
        labels_path=args.labels,
        model_path=args.model,
        confidence=args.confidence,
    )

    if metrics["positive_images"] > 0 and metrics["exact_match_accuracy_all"] < 0.01:
        logger.warning("⚠️ Very low accuracy — check model, image quality, and annotations.")


if __name__ == "__main__":
    main()
