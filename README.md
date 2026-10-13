# Vehicle ANPR — Automatic Number Plate Recognition System

A modular Python prototype for **Automatic Number Plate Recognition (ANPR)** targeting Indian vehicle registration plates:

- **Image & camera input**: Upload JPG, JPEG, PNG, or WEBP images, or capture directly using your browser camera.
- **Plate detection**: Uses a YOLOv8 model trained specifically on license plate regions (not a generic vehicle/car detector).
- **Image enhancement**: Upscaling, grayscale conversion, CLAHE contrast enhancement, and bilateral filtering for improved character visibility.
- **OCR recognition**: EasyOCR engine reading Latin-alphabet Indian registrations with clean character normalization and Indian format heuristics.
- **Demo vehicle lookup**: Pluggable provider interface with a local CSV demo database (clearly labeled as **SYNTHETIC DEMO DATA**).
- **Video processing**: Frame-by-frame video processing with duplicate suppression, extension validation, and timestamp tracking.
- **Streamlit dashboard**: Clean tabs for Image, Video, and Help with distinct status indicators, individual plate crops, and privacy notices.

---

## ⚠️ Important Privacy & Accuracy Notice

1. **Fictional Demo Data**: All bundled records in `data/vehicles.csv` are **synthetic demo data**. A database miss reports that no demo record was found — it **never** claims that the vehicle is officially unregistered.
2. **OCR Limitations**: OCR engines can and do make mistakes. High OCR confidence is **not** a calibrated guarantee of correctness.
3. **No Owner Doxxing**: This project intentionally does **not** expose owner names, phone numbers, addresses, or private details.
4. **Authorized Access Only**: Never scrape or bypass CAPTCHAs/access controls on official portals like VAHAN/Parivahan. Real integrations require documented, authorized API access with proper credentials.
5. **Permitted Use**: Process only images you have legal authorization to analyze.

---

## Architecture & Project Structure

```
vehicle_anpr_project/
├── app.py                      # Main Streamlit web application
├── evaluate.py                 # Evaluation script for benchmark datasets
├── setup_model.py              # Helper script for model verification & download
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # This documentation
├── FINAL_TEST_REPORT.md        # Detailed test results & verification
├── data/
│   └── vehicles.csv            # Synthetic demo vehicle records
├── models/
│   └── license_plate.pt        # YOLOv8 plate detector weights (~5.6 MB)
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

## Installation on Windows 11

### Prerequisites
- **Python 3.11** or **Python 3.12** installed and added to your `PATH`.
- **Git** (optional).
- Internet connection for initial package installation and OCR model download.

### Step 1: Open PowerShell
Open Windows PowerShell in the project directory:

```powershell
cd "c:\Users\Project\Vehicle ANPR\vehicle_anpr_project"
```

### Step 2: Create and activate virtual environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Note**: If PowerShell displays an execution policy error (`running scripts is disabled`), run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Or use Command Prompt (`cmd.exe`):
> ```cmd
> .venv\Scripts\activate.bat
> ```

### Step 3: Upgrade pip and install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Variables (`.env`)
Copy the example environment file if you wish to customize configuration:
```powershell
Copy-Item .env.example .env
```
Settings in `.env` (such as `PLATE_MODEL_PATH`) are automatically loaded by `app.py` via `python-dotenv`. You can also set environment variables in your active shell or system settings.

### Step 5: Verify or acquire plate detector weights
To inspect or verify the plate detector model weights:
```powershell
python setup_model.py
```
This script computes the SHA-256 hash of `models/license_plate.pt` and checks it against reference weights.

---

## Model Licensing & Dataset Provenance

| Component | Provenance & Licensing Status |
|---|---|
| **YOLOv8 Framework (`ultralytics`)** | Licensed by Ultralytics under **GNU AGPL-3.0**. Commercial use without open-sourcing derivative code requires an enterprise license directly from Ultralytics. |
| **Model Checkpoint (`models/license_plate.pt`)** | **Checkpoint License Notice**: Because public redistribution rights for third-party community weights cannot be formally established, pre-trained weights are **NOT bundled** in the clean redistributable ZIP. The codebase is fully modular and supports any YOLOv8 plate detector checkpoint. Users may acquire reference weights for educational testing via `python setup_model.py`, or train custom weights on an explicitly licensed dataset (e.g., CC BY 4.0). |
| **OCR Engine (`easyocr`)** | JaidedAI EasyOCR is licensed under **Apache 2.0**. Character recognition weights are fetched to `~/.EasyOCR/` on first use. |
| **Demo Vehicle Database (`data/vehicles.csv`)** | 13 synthetic demo vehicle records created specifically for this prototype. Dedicated to the public domain / MIT. |

### Recommended Datasets for Training / Fine-Tuning
1. **Roboflow Universe — Indian Number Plates**: [Link](https://universe.roboflow.com/datacluster-labs-agryi/indian-number-plates-9oobq) (CC BY 4.0 license, real-world images across India).
2. **Kaggle — Indian Vehicle Dataset**: [Link](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) (review dataset-specific terms).
3. **Hugging Face — Indian License Plate Detection**: [Link](https://huggingface.co/datasets/thundarstrom/indian-license-plate-detection).

---

## Running the Application

Start the Streamlit dashboard:

```powershell
streamlit run app.py
```

Streamlit will open in your browser at `http://localhost:8501`.

### Interface Overview:
1. **Sidebar**: Configurable YOLO model path, detection confidence slider (default: 0.40), full-image OCR fallback toggle, and data source indicator.
2. **Image Tab**: Upload JPG, JPEG, PNG, or WEBP images, or capture via webcam. Click **"Detect plate and look up vehicle"**.
   - Displays bounding box annotations on detected plate regions.
   - Shows extracted plate crops.
   - Displays recognized registration string, OCR confidence score, and format plausibility check.
   - Performs lookup in the local demo database (or indicates no demo record exists).
3. **Video Tab**: Upload `.mp4`, `.avi`, `.mov`, or `.mkv` video. Validates file extension and size, processes frames with configurable sampling, suppresses duplicate sightings, and displays timestamped observations.
4. **Help Tab**: Pipeline documentation, licensing considerations, and privacy notice.

---

## Running Automated Tests

Run the full pytest suite:

```powershell
pytest -v
```

All unit tests execute rapidly because AI models (YOLO and EasyOCR) are mocked in unit tests, allowing verification without GPU hardware or external downloads.

---

## Evaluating on a Labeled Dataset

To measure recognition performance on a labeled benchmark dataset:

```powershell
python evaluate.py --data-dir path/to/images --labels path/to/labels.csv --confidence 0.40
```

Where `labels.csv` contains:
```csv
filename,plate_text
img01.jpg,MH12DE1433
img02.jpg,KA05MH7777
img03.jpg,
```
*(Leave `plate_text` blank for negative sample images containing no plates).*

### Evaluation Metrics:
- **Manifest Records**: Total entries evaluated from the ground-truth manifest.
- **Processed Images**: Number of images successfully loaded and evaluated.
- **Missing Images**: Files present in the manifest but absent from disk (explicitly tracked).
- **Unreadable Images**: Files that failed decoding or image loading (explicitly tracked).
- **Positive Recognition Rate (All Expected Positive)**: Fraction of all manifest positive records where text was recognized (`positive_recognized / positive_images`). Does not silently drop missing or unreadable images.
- **Positive Recognition Rate (Processed Positive)**: Fraction of successfully processed positive images where text was recognized (`positive_recognized / positive_processed`).
- **Exact Match Accuracy (All vs. Processed)**: Fraction of plates where recognized text exactly matches ground truth.
- **Character Error Rate (CER)**: Total Levenshtein edit distance divided by total ground-truth characters.
- **False Positive Rate**: Fraction of processed negative images (no plate) where plate text was erroneously reported.
- **Processing Latency**: Mean and median processing duration per frame in seconds.

*(Note: This measures end-to-end character recognition. Object detection spatial IoU recall requires ground-truth bounding box coordinates [x1, y1, x2, y2]).*

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
3. **Configure Secrets**: Set credentials in `.env` (never commit `.env` to version control):
   ```
   AUTHORIZED_API_KEY=your_secret_key
   AUTHORIZED_API_URL=https://authorized-gateway.gov.in/api/v1/rc
   ```
4. **Security & Privacy Safeguards**:
   - Never log or store raw images of vehicles or owners.
   - Never display owner names, personal mobile numbers, or home addresses in general UIs.
   - Implement audit logs recording who queried what registration and when.
   - Enforce rate-limiting and HTTPS encryption in transit.

---

## Troubleshooting & FAQ

### 1. `ImportError: DLL load failed` or OpenCV issues
- Install `opencv-python-headless` (or `opencv-python`) compatible with your Windows environment.
- On Windows N editions, install the Windows Media Feature Pack.

### 2. Ultralytics / PyTorch Windows execution policies
- If PyTorch DLL loading reports an Application Control block, verify that your Windows security policy allows running Python libraries from your user virtual environment.

### 3. Slow CPU performance
- EasyOCR downloads PyTorch weights (~50 MB) to `~/.EasyOCR/` on its first run; subsequent runs are significantly faster.
- For video processing, increase `frame_skip` (e.g., 10 or 15) to minimize computation.

### 4. Port conflicts with Streamlit
If port 8501 is in use:
```powershell
streamlit run app.py --server.port 8502
```
