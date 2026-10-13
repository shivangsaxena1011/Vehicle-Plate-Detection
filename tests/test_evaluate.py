"""Tests for evaluate.py metrics, zero-division handling, and missing image tracking."""

import os
import tempfile
import pytest

from evaluate import edit_distance, compute_evaluation_metrics, load_labels


class TestEditDistance:
    def test_exact_match(self):
        assert edit_distance("MH12DE1433", "MH12DE1433") == 0

    def test_single_substitution(self):
        assert edit_distance("MH12DE1433", "MH12DE1438") == 1

    def test_insertion_and_deletion(self):
        assert edit_distance("MH12", "MH12A") == 1
        assert edit_distance("MH12A", "MH12") == 1

    def test_empty_strings(self):
        assert edit_distance("", "") == 0
        assert edit_distance("ABC", "") == 3
        assert edit_distance("", "ABC") == 3


class TestComputeEvaluationMetrics:
    def test_all_positive_exact_match(self):
        manifest = {"img1.jpg": "MH12DE1433", "img2.jpg": "KA05MH7777"}
        predictions = {"img1.jpg": "MH12DE1433", "img2.jpg": "KA05MH7777"}
        metrics = compute_evaluation_metrics(manifest, predictions)

        assert metrics["manifest_records"] == 2
        assert metrics["processed_images"] == 2
        assert metrics["missing_images"] == 0
        assert metrics["unreadable_images"] == 0
        assert metrics["positive_images"] == 2
        assert metrics["positive_processed"] == 2
        assert metrics["negative_images"] == 0
        assert metrics["negative_processed"] == 0
        assert metrics["positive_recognized"] == 2
        assert metrics["exact_matches"] == 2
        assert metrics["recognition_rate_all_positive"] == 1.0
        assert metrics["recognition_rate_processed_positive"] == 1.0
        assert metrics["exact_match_accuracy_all"] == 1.0
        assert metrics["exact_match_accuracy_processed"] == 1.0
        assert metrics["character_error_rate"] == 0.0
        assert metrics["false_positives"] == 0
        assert metrics["false_positive_rate"] == 0.0

    def test_missing_and_unreadable_images_tracked_separately(self):
        # Manifest has 4 positive images and 1 negative image
        manifest = {
            "pos1.jpg": "MH12DE1433",
            "pos2.jpg": "KA05MH7777",
            "pos_missing.jpg": "DL01CA0001",
            "pos_corrupt.jpg": "RJ14SB6789",
            "neg1.jpg": "",
        }
        # Only pos1, pos2, and neg1 were processed
        predictions = {
            "pos1.jpg": "MH12DE1433",  # exact match
            "pos2.jpg": "KA05MH7778",  # recognized, but 1 char error
            "neg1.jpg": "",            # true negative
        }
        missing = ["pos_missing.jpg"]
        unreadable = ["pos_corrupt.jpg"]
        errors = ["Missing file: pos_missing.jpg", "Cannot read image: pos_corrupt.jpg"]

        metrics = compute_evaluation_metrics(
            manifest=manifest,
            predictions=predictions,
            missing_images=missing,
            unreadable_images=unreadable,
            latencies=[0.5, 0.4, 0.2],
            errors=errors,
        )

        assert metrics["manifest_records"] == 5
        assert metrics["processed_images"] == 3
        assert metrics["missing_images"] == 1
        assert metrics["unreadable_images"] == 1
        assert metrics["positive_images"] == 4
        assert metrics["positive_processed"] == 2
        assert metrics["negative_images"] == 1
        assert metrics["negative_processed"] == 1

        # Both pos1 and pos2 returned text -> 2 positive recognitions
        assert metrics["positive_recognized"] == 2

        # Recognition rate across ALL expected positive records: 2 / 4 = 50%
        assert metrics["recognition_rate_all_positive"] == 0.50

        # Recognition rate among PROCESSED positive images: 2 / 2 = 100%
        assert metrics["recognition_rate_processed_positive"] == 1.00

        # Exact match accuracy all: 1 / 4 = 25%
        assert metrics["exact_match_accuracy_all"] == 0.25

        # Exact match accuracy processed: 1 / 2 = 50%
        assert metrics["exact_match_accuracy_processed"] == 0.50

        # Error tracking preserved
        assert len(metrics["errors"]) == 2
        assert "pos_missing.jpg" in metrics["missing_image_list"]
        assert "pos_corrupt.jpg" in metrics["unreadable_image_list"]

    def test_empty_dataset_safe_denominators(self):
        metrics = compute_evaluation_metrics(manifest={}, predictions={})

        assert metrics["manifest_records"] == 0
        assert metrics["processed_images"] == 0
        assert metrics["positive_images"] == 0
        assert metrics["positive_processed"] == 0
        assert metrics["recognition_rate_all_positive"] == 0.0
        assert metrics["recognition_rate_processed_positive"] == 0.0
        assert metrics["exact_match_accuracy_all"] == 0.0
        assert metrics["exact_match_accuracy_processed"] == 0.0
        assert metrics["character_error_rate"] == 0.0
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["mean_latency_sec"] == 0.0
        assert metrics["median_latency_sec"] == 0.0

    def test_only_negative_images_safe_denominators(self):
        manifest = {"neg1.jpg": "", "neg2.jpg": ""}
        predictions = {"neg1.jpg": "", "neg2.jpg": "FALSEPLATE"}

        metrics = compute_evaluation_metrics(manifest=manifest, predictions=predictions)

        assert metrics["manifest_records"] == 2
        assert metrics["positive_images"] == 0
        assert metrics["positive_processed"] == 0
        assert metrics["negative_images"] == 2
        assert metrics["negative_processed"] == 2
        assert metrics["positive_recognized"] == 0
        assert metrics["recognition_rate_all_positive"] == 0.0
        assert metrics["recognition_rate_processed_positive"] == 0.0
        assert metrics["exact_match_accuracy_all"] == 0.0
        assert metrics["exact_match_accuracy_processed"] == 0.0
        assert metrics["false_positives"] == 1
        assert metrics["false_positive_rate"] == 0.50

    def test_positive_unrecognized_handling(self):
        manifest = {"pos1.jpg": "MH12DE1433"}
        predictions = {"pos1.jpg": ""}  # processed, but OCR returned empty string

        metrics = compute_evaluation_metrics(manifest=manifest, predictions=predictions)

        assert metrics["positive_images"] == 1
        assert metrics["positive_processed"] == 1
        assert metrics["positive_recognized"] == 0
        assert metrics["exact_matches"] == 0
        assert metrics["recognition_rate_all_positive"] == 0.0
        assert metrics["recognition_rate_processed_positive"] == 0.0
        assert metrics["exact_match_accuracy_all"] == 0.0
        assert metrics["character_error_rate"] == 1.0


class TestLoadLabels:
    def test_load_valid_csv(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            f.write("filename,plate_text\n")
            f.write("img1.jpg, mh-12-de 1433 \n")
            f.write("img2.jpg, \n")
            f.close()

            labels = load_labels(f.name)
            assert labels["img1.jpg"] == "MH12DE1433"
            assert labels["img2.jpg"] == ""
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass

    def test_load_csv_missing_columns_raises(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            f.write("image_id,label\nimg1.jpg,MH12DE1433\n")
            f.close()
            with pytest.raises(ValueError, match="filename"):
                load_labels(f.name)
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass
