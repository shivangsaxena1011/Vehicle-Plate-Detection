# Vehicle Number Plate Recognition (ANPR)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shivang-vehicle-plate-detection.streamlit.app/)

A modular Computer Vision and Deep Learning application for **Automatic Number Plate Recognition (ANPR)** targeting Indian vehicle registration plates.

🔗 **Live Web Application**: [https://shivang-vehicle-plate-detection.streamlit.app/](https://shivang-vehicle-plate-detection.streamlit.app/)  
📂 **GitHub Repository**: [https://github.com/shivangsaxena1011/Vehicle-Plate-Detection](https://github.com/shivangsaxena1011/Vehicle-Plate-Detection)

---

## Key Features

- **Multi-Format Input**: Upload images (JPG, JPEG, PNG, WEBP) or capture directly using your browser camera.
- **YOLOv8 Plate Detection**: Modular integration with YOLOv8 models trained specifically for license plate localization (distinguishing plate bounds from vehicles).
- **Computer Vision Enhancement**: OpenCV preprocessing pipeline including intelligent upscaling, grayscale conversion, Contrast Limited Adaptive Histogram Equalization (CLAHE), and bilateral filtering.
- **EasyOCR Recognition**: Deep learning text recognition engine with Latin-character normalization and Indian registration format heuristic checks (`XX 00 XX 0000`).
- **Pluggable Vehicle Provider**: Extensible vehicle lookup architecture with a local demo database (`data/vehicles.csv` labeled as **SYNTHETIC DEMO DATA**).
- **Video & Real-Time Stream Processing**: Frame-by-frame video processing (`.mp4`, `.avi`, `.mov`, `.mkv`) with frame sampling, timestamp recording, and consecutive-frame duplicate suppression.
- **Streamlit Web Dashboard**: Modern, responsive interface with separate tabs for Image recognition, Video processing, and Pipeline documentation.

---

## ⚠️ Important Privacy & Legal Notice

1. **Synthetic Demo Records**: All bundled vehicle records in `data/vehicles.csv` are **fictional demo data**. A database miss indicates no matching demo record exists — it **never** claims that the vehicle is unregistered.
2. **OCR Limitations**: OCR engines can make mistakes. High OCR confidence is a statistical indicator, not a guarantee of correctness.
3. **No Personal Data Exposure**: This system intentionally does **not** expose owner names, phone numbers, home addresses, or personal contact details.
4. **Authorized Registry Access Only**: Real-world registry lookups require documented, legally authorized API access (e.g. official MoRTH / NIC Parivahan gateway). Scraping or bypassing access controls on public portals is strictly prohibited.
5. **Permitted Use**: Process only images and videos you are authorized to analyze.

---

## Architecture & Project Structure

```
vehicle_anpr_project/
├── app.py                      # Main Streamlit web dashboard
├── evaluate.py                 # Evaluation script for benchmark datasets
├── setup_model.py              # Helper utility for model verification & setup
├── requirements.txt            # Streamlit Cloud & local headless dependencies
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── FINAL_TEST_REPORT.md        # Detailed test results & verification
├── data/
│   └── vehicles.csv            # Synthetic demo vehicle records
├── models/
│   └── (user-configured YOLO .pt weights)
├── src/
│   ├── __init__.py             # Package init
│   ├── anpr.py                 # End-to-end ANPR pipeline orchestration
│   ├── detector.py             # YOLOv8 plate detector module
│   ├── ocr_engine.py           # EasyOCR engine, text normalization & heuristics
│   ├── image_utils.py          # Image validation, decode, CLAHE, deskew, boxes
│   ├── database.py             # CSV loading, normalization & lookup operations
│   ├── vehicle_provider.py     # Pluggable provider interface (Demo / Authorized)
│   └── video.py                # Video & webcam processing with deduplication
└── tests/
    ├── __init__.py
    ├── test_database.py        # Database loading, normalization & lookup tests
    ├── test_image_utils.py     # Image & video validation, decoding & preprocessing
    ├── test_ocr_engine.py      # OCR normalization, format heuristic & reader tests
    ├── test_anpr.py            # End-to-end pipeline tests (mocked detector & OCR)
    ├── test_vehicle_provider.py# Pluggable provider interface tests
    └── test_evaluate.py        # Evaluation metrics and label parser tests
```

---

## Model Licensing & Dataset Provenance

| Component | Provenance & Licensing Status |
|---|---|
| **YOLOv8 Framework (`ultralytics`)** | Ultralytics YOLOv8 software framework is licensed under **GNU AGPL-3.0**. Commercial use without open-sourcing derivative works requires an enterprise license purchased from Ultralytics. |
| **Model Weights (`models/license_plate.pt`)** | **Checkpoint License Notice**: Because public redistribution rights for third-party community checkpoints cannot be formally established, pre-trained `.pt` weights are **intentionally excluded** from this public repository. The application is fully modular: it operates in graceful full-image OCR fallback mode without weights, and seamlessly loads any compatible YOLOv8 plate detection checkpoint placed in `models/` or specified in the sidebar. |
| **OCR Engine (`easyocr`)** | JaidedAI EasyOCR is licensed under **Apache 2.0**. Character recognition weights are fetched automatically to `~/.EasyOCR/` on first run. |
| **Demo Database (`data/vehicles.csv`)** | 13 synthetic demo vehicle records created specifically for this prototype. Dedicated to the public domain / MIT. |

### Recommended Datasets for Training / Fine-Tuning
1. **Roboflow Universe — Indian Number Plates**: [Link](https://universe.roboflow.com/datacluster-labs-agryi/indian-number-plates-9oobq) (CC BY 4.0 license, real-world images across India).
2. **Kaggle — Indian Vehicle Dataset**: [Link](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) (review dataset-specific terms).
3. **Hugging Face — Indian License Plate Detection**: [Link](https://huggingface.co/datasets/thundarstrom/indian-license-plate-detection).

---

## Streamlit Cloud Deployment Note

### Headless Linux Environment Resolution
On Streamlit Community Cloud (and headless Linux servers), installing standard GUI OpenCV builds can trigger:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
This repository resolves this issue by specifying **`ultralytics-opencv-headless`** alongside **`opencv-python-headless`** in `requirements.txt`. This ensures:
1. `from ultralytics import YOLO` and `import cv2` work seamlessly without requiring desktop OpenGL/X11 shared libraries (`libGL.so.1`).
2. Mutually incompatible GUI `opencv-python` wheels are not pulled into the deployment environment.

---

## Real Vehicle Registry Integration (Future Roadmap)

To replace the demo CSV with an authorized real-world vehicle registry:

1. **Obtain Official Authorization**: Register with the authorized authority (e.g., MoRTH/NIC Parivahan/VAHAN API through official government API gateways, or an authorized commercial data aggregator).
2. **Implement Provider**: Subclass `VehicleProvider` in [`src/vehicle_provider.py`](file:///c:/Users/Project/Vehicle%20ANPR/vehicle_anpr_project/src/vehicle_provider.py):
   ```python
   class ParivahanAuthorizedProvider(VehicleProvider):
       def lookup(self, plate_text: str) -> VehicleLookupResult:
           # Call authorized REST API with encrypted token
           ...
   ```
3. **Configure Secrets**: Set credentials in `.env` or Streamlit Cloud Secrets (never commit credentials to version control):
   ```
   AUTHORIZED_API_KEY=your_secret_key
   AUTHORIZED_API_URL=https://authorized-gateway.gov.in/api/v1/rc
   ```
4. **Security & Privacy Safeguards**:
   - Never log or store raw images of vehicles or owners.
   - Never display owner names, personal mobile numbers, or home addresses in general UIs.
   - Implement audit logs recording who queried what registration and when.
   - Enforce rate-limiting and HTTPS encryption in transit.
