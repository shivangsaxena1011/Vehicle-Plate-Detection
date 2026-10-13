"""Vehicle ANPR — Streamlit dashboard.

Main entry point for the application. Handles image upload, camera capture,
plate detection, OCR, database lookup, and optional video processing.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

# Load local environment configuration (.env) if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.anpr import ANPRPipeline, ANPRResult
from src.image_utils import validate_file_extension, validate_file_size, decode_image
from src.vehicle_provider import DemoCSVProvider, VehicleLookupResult
from src.video import validate_video_extension, MAX_VIDEO_SIZE_BYTES, VideoProcessor, VideoProcessingConfig

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ──
st.set_page_config(page_title="Vehicle ANPR", page_icon="🚘", layout="wide")

# ── Permanent notice ──
st.title("🚘 Vehicle Number Plate Recognition")
st.caption("Image → plate detection → OCR → demo vehicle database lookup")

st.warning(
    "⚠️ **Privacy & accuracy notice:** OCR can make mistakes — recognized text "
    "is not proof of registration or identity. The bundled vehicle database "
    "contains **fictional demo records only**. Use only images you are "
    "authorized to process. Do not use this tool for surveillance or "
    "identification purposes without proper legal authorization."
)

# ── Sidebar settings ──
with st.sidebar:
    st.header("⚙️ Settings")

    model_path = st.text_input(
        "YOLO plate model path",
        value=os.getenv("PLATE_MODEL_PATH", "models/license_plate.pt"),
        help="Path to a YOLO model trained specifically to detect license plates. "
             "A general-purpose COCO model is NOT a plate detector.",
    )

    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05,
    )

    use_ocr_fallback = st.checkbox(
        "Try OCR on the full image if no plate model is available",
        value=True,
        help="When enabled, the app runs OCR on the entire image as a limited "
             "fallback. This is not reliable plate localization.",
    )

    st.markdown("---")
    st.markdown("**📊 Data source**")
    st.write("Local CSV demo database")
    st.caption(
        "All records are SYNTHETIC DEMO DATA. "
        "Replace with an authorized provider only after confirming "
        "permission and API access."
    )

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.caption(
        "ANPR prototype for educational and evaluation purposes. "
        "See README.md for model and dataset licensing considerations."
    )


# ── Cache the pipeline ──
@st.cache_resource(show_spinner="Loading OCR engine and detection model...")
def get_pipeline(path: str, conf: float, fallback: bool) -> ANPRPipeline:
    """Cache the ANPR pipeline to avoid reloading models on every interaction."""
    return ANPRPipeline(
        model_path=path,
        detection_confidence=conf,
        allow_full_image_ocr=fallback,
    )


@st.cache_resource
def get_vehicle_provider() -> DemoCSVProvider:
    """Cache the demo database provider."""
    return DemoCSVProvider(csv_path="data/vehicles.csv")


# ── Tabs ──
tab_image, tab_video, tab_help = st.tabs(["📷 Image", "🎥 Video", "❓ Help"])


# ═══════════════════════════════════════════════════════
#  IMAGE TAB
# ═══════════════════════════════════════════════════════
with tab_image:
    st.subheader("Upload or capture a vehicle image")

    col_upload, col_camera = st.columns(2)
    with col_upload:
        uploaded = st.file_uploader(
            "Upload a vehicle image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Accepted formats: JPG, JPEG, PNG, WEBP. Max 20 MB.",
        )
    with col_camera:
        camera_image = st.camera_input("Or capture a photo")

    image_source = uploaded if uploaded is not None else camera_image

    if image_source is not None:
        # ── Validate file ──
        filename = getattr(image_source, "name", "capture.jpg")

        # Extension check
        ext_ok, ext_msg = validate_file_extension(filename)
        if not ext_ok:
            st.error(f"❌ {ext_msg}")
            st.stop()

        # Size check
        raw_bytes = image_source.getvalue()
        size_ok, size_msg = validate_file_size(raw_bytes)
        if not size_ok:
            st.error(f"❌ {size_msg}")
            st.stop()

        # Decode check
        rgb_array, decode_msg = decode_image(raw_bytes)
        if rgb_array is None:
            st.error(f"❌ {decode_msg}")
            st.stop()

        pil_image = Image.fromarray(rgb_array)

        # ── Show input image ──
        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.markdown("**Input image**")
            st.image(pil_image, use_container_width=True)

        # ── Process button ──
        if st.button("🔍 Detect plate and look up vehicle", type="primary"):
            pipeline = get_pipeline(model_path, confidence, use_ocr_fallback)

            # Show detector status
            if not pipeline.detector_loaded:
                st.info(
                    "ℹ️ No plate-detection model loaded. "
                    "Running in OCR-only fallback mode. "
                    "For accurate localization, add a YOLO plate model (see setup_model.py)."
                )

            with st.spinner("Detecting plate and reading text..."):
                try:
                    result: ANPRResult = pipeline.process(pil_image)
                except Exception as exc:
                    logger.exception("Image recognition pipeline failed unexpectedly")
                    st.error(
                        "An unexpected error occurred while processing the image. "
                        "Please check the image and try again."
                    )
                    st.stop()

            # ── Detection results ──
            with right_col:
                st.markdown("**Recognition result**")

                if result.annotated_image is not None:
                    st.image(
                        result.annotated_image,
                        caption="Detected plate region(s)",
                        use_container_width=True,
                    )

                # Status display
                status_map = {
                    "text_recognized": ("✅ Text recognized", "success"),
                    "plate_detected_but_unreadable": ("⚠️ Plate detected but text unreadable", "warning"),
                    "not_recognized": ("❌ No plate text recognized", "error"),
                    "no_detector": ("ℹ️ No detector available", "info"),
                }
                status_text, status_type = status_map.get(
                    result.status, (result.status, "info")
                )
                getattr(st, status_type)(status_text)

                st.write(f"**Detection method:** {result.detection_method}")
                st.write(f"**Plates found:** {len(result.plates)}")

                # Notes
                for note in result.notes:
                    st.info(note)

            # ── Per-plate results ──
            if result.plates:
                st.markdown("---")
                st.subheader("📋 Plate Readings")

                for i, plate_reading in enumerate(result.plates):
                    plate_label = plate_reading.text or "(unreadable)"

                    with st.expander(
                        f"Plate {i + 1}: **{plate_label}**",
                        expanded=(i == 0),
                    ):
                        pcol1, pcol2 = st.columns([1, 2])

                        with pcol1:
                            if plate_reading.crop is not None:
                                st.image(
                                    plate_reading.crop,
                                    caption="Plate crop",
                                    use_container_width=True,
                                )

                        with pcol2:
                            st.metric("OCR Text", plate_reading.text or "—")

                            if plate_reading.ocr_confidence is not None:
                                conf_pct = plate_reading.ocr_confidence * 100
                                st.metric("OCR Confidence", f"{conf_pct:.1f}%")
                                if conf_pct < 50:
                                    st.warning(
                                        "Low OCR confidence — the text may be incorrect."
                                    )
                            else:
                                st.caption("OCR confidence: not available")

                            if plate_reading.detector_confidence is not None:
                                st.metric(
                                    "Detector Confidence",
                                    f"{plate_reading.detector_confidence * 100:.1f}%",
                                )

                            if plate_reading.plausible_indian:
                                st.caption("✅ Matches Indian plate format (heuristic only)")
                            elif plate_reading.text:
                                st.caption("ℹ️ Does not match typical Indian format")

                        # ── Database lookup ──
                        if plate_reading.text:
                            provider = get_vehicle_provider()
                            lookup_result: VehicleLookupResult = provider.lookup(
                                plate_reading.text
                            )

                            st.markdown("**Vehicle Database Lookup**")

                            if lookup_result.status == "found":
                                st.success("✅ " + lookup_result.message)
                                if lookup_result.vehicle:
                                    display_data = {
                                        k: v for k, v in lookup_result.vehicle.items()
                                        if not k.startswith("_")
                                    }
                                    st.dataframe(
                                        pd.DataFrame([display_data]),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                if lookup_result.is_demo_data:
                                    st.caption("📌 This is SYNTHETIC DEMO DATA.")

                            elif lookup_result.status == "not_found":
                                st.info("ℹ️ " + lookup_result.message)

                            elif lookup_result.status == "provider_unavailable":
                                st.warning("⚠️ " + lookup_result.message)

                            else:
                                st.error("❌ " + lookup_result.message)
                        else:
                            st.info("No readable text — cannot perform database lookup.")
            else:
                st.markdown("---")
                st.warning(
                    "No readable plate text was returned. Suggestions:\n"
                    "- Use a sharper, closer, well-lit image\n"
                    "- Ensure the plate is clearly visible\n"
                    "- Try adjusting the confidence threshold\n"
                    "- Install a plate-detection model for better accuracy"
                )


# ═══════════════════════════════════════════════════════
#  VIDEO TAB
# ═══════════════════════════════════════════════════════
with tab_video:
    st.subheader("🎥 Video Processing (Experimental)")
    st.caption(
        "Process a video file frame-by-frame. Performance depends on video length, "
        "resolution, frame rate, and hardware."
    )

    video_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        frame_skip = st.number_input(
            "Process every Nth frame",
            min_value=1,
            max_value=60,
            value=10,
            help="Higher = faster execution, but brief appearances may be missed",
        )
    with vcol2:
        max_frames = st.number_input(
            "Max frames to process",
            min_value=10,
            max_value=2000,
            value=200,
        )

    if video_file is not None:
        if st.button("▶️ Process video", type="primary"):
            # 1. Validate original file extension
            v_name = getattr(video_file, "name", "video.mp4")
            ext_ok, ext_msg = validate_video_extension(v_name)
            if not ext_ok:
                st.error(f"❌ {ext_msg}")
                st.stop()

            v_suffix = Path(v_name).suffix.lower()

            # 2. Validate video size limit
            v_bytes = video_file.getvalue()
            if len(v_bytes) > MAX_VIDEO_SIZE_BYTES:
                st.error(
                    f"❌ Video file size ({len(v_bytes) / (1024*1024):.1f} MB) "
                    f"exceeds the limit of {MAX_VIDEO_SIZE_BYTES / (1024*1024):.0f} MB."
                )
                st.stop()

            # 3. Create temporary file preserving original validated extension
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=v_suffix
            ) as tmp:
                tmp.write(v_bytes)
                tmp_path = tmp.name

            try:
                pipeline = get_pipeline(model_path, confidence, use_ocr_fallback)
                config = VideoProcessingConfig(
                    frame_skip=frame_skip,
                    max_frames=max_frames,
                )
                processor = VideoProcessor(pipeline, config)

                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = []
                for frame_num, total, frame_result in processor.process_frames_generator(tmp_path):
                    if total > 0:
                        progress_bar.progress(min(frame_num / total, 1.0))
                    status_text.text(f"Processing frame {frame_num}/{total}...")

                    if frame_result and frame_result.plates:
                        all_results.append(frame_result)

                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")

                # Display results
                if all_results:
                    st.success(f"Found plates in {len(all_results)} frame(s).")

                    results_data = []
                    for fr in all_results:
                        for j, plate in enumerate(fr.plates):
                            results_data.append({
                                "Frame": fr.frame_number,
                                "Time (s)": fr.timestamp_sec,
                                "Plate": plate,
                                "Confidence": (
                                    f"{fr.confidences[j] * 100:.1f}%"
                                    if j < len(fr.confidences) and fr.confidences[j]
                                    else "N/A"
                                ),
                            })

                    st.dataframe(
                        pd.DataFrame(results_data),
                        use_container_width=True,
                        hide_index=True,
                    )

                    with st.expander("📸 Annotated frames"):
                        for fr in all_results[:10]:
                            if fr.annotated_frame is not None:
                                st.image(
                                    fr.annotated_frame,
                                    caption=f"Frame {fr.frame_number} "
                                            f"(t={fr.timestamp_sec}s): "
                                            f"{', '.join(fr.plates)}",
                                    use_container_width=True,
                                )
                else:
                    st.info("No plates detected in the processed video frames.")

            except Exception as exc:
                logger.exception("Video processing failed: %s", exc)
                st.error("An error occurred while processing the video. Please verify the file format and codec.")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ═══════════════════════════════════════════════════════
#  HELP TAB
# ═══════════════════════════════════════════════════════
with tab_help:
    st.subheader("❓ How This Prototype Works")
    st.markdown(
        """
        ### Pipeline

        1. **Image validation** — checks file extension, upload size, and OpenCV decodability.
        2. **Plate detection** — if a YOLO plate-detection model is configured,
           it extracts bounding boxes specifically around license plate regions.
        3. **Preprocessing** — plate crops are upscaled, converted to grayscale, and enhanced via CLAHE.
        4. **OCR** — EasyOCR reads text from each detected crop.
        5. **Normalization** — spaces and symbols are removed, and a heuristic checks
           Indian registration structure (not proof of registration).
        6. **Database lookup** — the recognized string is matched against `data/vehicles.csv`,
           which contains **fictional demo records only**.

        ### Important Limitations & Licensing

        - **No plate detector fallback:** If no YOLO model weights are available, the app
          falls back to running OCR on the full image.
        - **OCR accuracy:** Recognized characters may contain errors. Confidence scores
          are not calibrated probabilities of correctness.
        - **Demo data only:** The CSV database is synthetic. A lookup miss means
          the plate is not in the demo database — it does NOT mean the vehicle
          is unregistered.
        - **Model licensing:** Ultralytics YOLOv8 code is AGPL-3.0. Model weights obtained from
          community repositories or fine-tuned on third-party datasets must be reviewed
          for specific redistribution and commercial usage terms.
        - **Privacy protection:** Do not process unauthorized images or publish vehicle sightings.
        """
    )

st.markdown("---")
st.caption(
    "Vehicle ANPR Prototype • Educational use only • "
    "Bundled data is SYNTHETIC DEMO DATA • "
    "See README.md for full documentation"
)
