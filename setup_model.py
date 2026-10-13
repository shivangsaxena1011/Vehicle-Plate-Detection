"""Helper script for license-plate detection model setup.

This utility inspects the models/ directory, verifies existing weights,
checks their SHA-256 integrity, or provides instructions for downloading
or fine-tuning a plate detector.

Licensing Notice:
- Framework: Ultralytics YOLOv8 software is licensed under AGPL-3.0.
  Commercial deployment without source disclosure requires an enterprise
  license purchased directly from Ultralytics.
- Model Checkpoint: The reference community checkpoint was trained on
  the Kaggle Indian License Plate Dataset. The source repository does not
  include an explicit open-source license grant. Because public redistribution
  rights cannot be established, these weights are NOT bundled in the clean
  release package. Users must obtain or train their own model checkpoint.

Usage:
    python setup_model.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# Optional community reference weights (for educational/research experimentation)
REFERENCE_MODEL_URL = (
    "https://raw.githubusercontent.com/Shrestha-Swami/automatic-number-plate-recognition"
    "/git_main/models/yolov8/best.pt"
)
REFERENCE_MODEL_SHA256 = "88d1ae65d8eafa4e9f2b824b48b9a7aa03c89a9cad669a8e95547b9a41cdfe45"
REFERENCE_MODEL_SIZE = 5620598  # bytes


def setup_model_directory() -> Path:
    """Create the models directory if it doesn't exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def check_existing_model(model_path: Path) -> bool:
    """Check if a model file exists and report its integrity."""
    if not model_path.is_file():
        return False

    size = model_path.stat().st_size
    size_mb = size / (1024 * 1024)
    file_hash = compute_file_sha256(model_path)

    print(f"File found at: {model_path} ({size_mb:.2f} MB)")
    print(f"SHA-256 hash:  {file_hash}")

    if file_hash == REFERENCE_MODEL_SHA256:
        print("Status: Matches community reference weights (Shrestha-Swami).")
    else:
        print("Status: Custom user-provided weights (differs from reference).")
    return True


def download_reference_weights(target_path: Path) -> bool:
    """Download the reference weights and validate size and SHA-256."""
    print(f"\nDownloading community reference weights from:\n  {REFERENCE_MODEL_URL}")
    print("NOTE: These weights are provided for research/educational testing.")
    temp_target = target_path.with_suffix(".tmp")
    try:
        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\rProgress: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(REFERENCE_MODEL_URL, temp_target, reporthook=_reporthook)
        print()

        downloaded_hash = compute_file_sha256(temp_target)
        if downloaded_hash != REFERENCE_MODEL_SHA256:
            print(f"ERROR: SHA-256 mismatch!\nExpected: {REFERENCE_MODEL_SHA256}\nGot:      {downloaded_hash}")
            temp_target.unlink(missing_ok=True)
            return False

        if target_path.exists():
            target_path.unlink()
        temp_target.rename(target_path)
        print(f"SUCCESS: Model saved to {target_path} and verified.")
        return True
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        if temp_target.exists():
            temp_target.unlink()
        return False


def main():
    print("=" * 70)
    print("  Vehicle ANPR — License Plate Detection Model Setup")
    print("=" * 70)
    print()
    print("LICENSING SUMMARY:")
    print("1. Framework License (Ultralytics):")
    print("   The YOLOv8 framework is licensed under GNU AGPL-3.0.")
    print("   Commercial use requires either releasing derivative source code")
    print("   or purchasing a commercial license from Ultralytics.")
    print()
    print("2. Checkpoint Licensing:")
    print("   Weights must have an explicit redistribution license to be bundled.")
    print("   The community reference checkpoint lacks an explicit OSS license.")
    print("   Therefore, it must be downloaded or trained separately by the user.")
    print("-" * 70)
    print()

    models_dir = setup_model_directory()
    model_path = models_dir / "license_plate.pt"

    has_model = check_existing_model(model_path)

    if has_model:
        print()
        response = input("A model file already exists. Re-download reference weights? (y/N): ").strip().lower()
        if response != "y":
            print("Keeping existing model.")
            return

    print("\nOptions for obtaining license-plate detector weights:")
    print("1. Download community reference weights (experimental, research only)")
    print("2. Train your own YOLOv8 model using an explicitly licensed dataset")
    print("   (e.g., CC BY 4.0 Indian Plate datasets on Roboflow Universe)")
    print()

    choice = input("Enter choice (1/2, default 1): ").strip() or "1"

    if choice == "1":
        ok = download_reference_weights(model_path)
        if not ok:
            print("\nDownload failed. You can manually download via:")
            print(f"  curl -L -o models/license_plate.pt {REFERENCE_MODEL_URL}")
    else:
        print("\nTraining guide:")
        print("1. Export an Indian License Plate dataset in YOLOv8 format (e.g., Roboflow CC BY 4.0).")
        print("2. Train via CLI: yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640")
        print(f"3. Place the resulting best.pt at: {model_path.resolve()}")


if __name__ == "__main__":
    main()
