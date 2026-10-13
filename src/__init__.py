# Vehicle ANPR — source package
"""
Automatic Number Plate Recognition pipeline.

Modules:
    image_utils  — image validation, preprocessing, enhancement
    detector     — YOLO-based license plate detection
    ocr_engine   — EasyOCR text recognition and normalization
    anpr         — end-to-end pipeline orchestration
    database     — CSV demo database and vehicle lookup
    vehicle_provider — pluggable vehicle-data provider interface
    video        — video / webcam frame-by-frame processing
"""
