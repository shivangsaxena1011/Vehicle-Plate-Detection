# Vehicle ANPR — Final Verification & Test Report

**Date & Time**: 2026-10-13 14:15 IST  
**Environment**: Windows 11 (build 26100), x86_64  
**Tested Python Environments**: Python 3.12.10 (verified clean environment) & Python 3.14.7  
**Project Directory**: `c:\Users\Project\Vehicle ANPR\vehicle_anpr_project`  
**Final Deliverable Archive**: `c:\Users\Project\Vehicle ANPR\vehicle_anpr_final.zip`  
**Safety Backup Preserved**: `c:\Users\Project\Vehicle ANPR\vehicle_anpr_project_backup/`

---

## 1. Summary of Final Corrections

1. **Evaluation Methodology & Metrics Overhaul (`evaluate.py`)**:
   - **Comprehensive Reporting**: The script now tracks and separately reports `manifest_records`, `processed_images`, `missing_images`, `unreadable_images`, `positive_images`, `positive_processed`, `negative_images`, `negative_processed`, `positive_recognized`, and `exact_matches`.
   - **Dual Recognition Metrics**:
     - `recognition_rate_all_positive` = `positive_recognized / positive_images`: Measures overall yield against all expected positive ground-truth manifest records without silently omitting failed or missing files.
     - `recognition_rate_processed_positive` = `positive_recognized / positive_processed`: Measures OCR/detection efficacy strictly among images successfully read and processed.
   - **Zero-Safe Denominators**: Fully protected against empty datasets, zero-positive manifests, and zero-negative subsets.
   - **Documentation Integrity**: Explicitly clarified that this measures end-to-end string extraction, not spatial bounding-box detection recall (IoU).
   - **Automated Tests**: Added 11 dedicated evaluation tests in `tests/test_evaluate.py` verifying all edge cases.

2. **Model Licensing Resolution**:
   - **Ultralytics Framework License**: GNU AGPL-3.0 (with commercial enterprise licensing available directly from Ultralytics).
   - **Checkpoint Licensing & Redistribution Status**: The community reference weights (`license_plate.pt`) originate from `Shrestha-Swami/automatic-number-plate-recognition` (trained on Kaggle `kedarsai/indian-license-plates-with-labels`). Because the source repository lacks an explicit open-source license grant, public redistribution rights cannot be formally established.
   - **Clean Redistribution Packaging**: To guarantee compliance, the redistributable package (`vehicle_anpr_final.zip`) excludes unverified third-party weights by default while preserving full modularity. Users configure weights locally via `setup_model.py` (which includes automated reference downloading and SHA-256 verification) or train custom checkpoints on an explicitly licensed dataset (e.g. CC BY 4.0).
   - Removed all unsupported claims of "open-source licensed model weights".

3. **Application Verification**:
   - **Test Suite**: All 106 unit tests passed in Python 3.12 and Python 3.14 (`106 passed in 9.40s`).
   - **Syntax Checks**: All 16 Python files compiled with zero syntax errors.
   - **Startup Verification**: Verified Streamlit web application startup in headless mode on port 8598 (`Uvicorn server started on :::8598`).
   - **No Fabricated Benchmarks**: Synthetic demo records are strictly labeled `SYNTHETIC DEMO DATA`. No real-world recognition accuracy is fabricated.

4. **Archive Integrity**:
   - Packaged `vehicle_anpr_final.zip` using standard POSIX forward-slash paths (`/`) without Windows backslash artifacts.
   - Preserved original safety backup in `c:\Users\Project\Vehicle ANPR\vehicle_anpr_project_backup/`.

---

## 2. Test Execution & Verification

### Test Results (106 Passed / 0 Failed)
```
tests/test_anpr.py ......................... [10 passed]
tests/test_database.py ..................... [21 passed]
tests/test_evaluate.py ..................... [11 passed]
tests/test_image_utils.py .................. [29 passed]
tests/test_ocr_engine.py ................... [27 passed]
tests/test_vehicle_provider.py ............. [ 8 passed]

============================= 106 passed in 9.40s =============================
```

### Commands Actually Run
| Command | Result |
|---|---|
| `python -m py_compile app.py evaluate.py setup_model.py src\*.py tests\*.py` | Exit 0 (`SYNTAX_CHECK_PASSED`) |
| `python -m pytest tests/ -v` | Exit 0 (`106 passed in 9.40s`) |
| `py -3.12 -m pytest tests/ -v` | Exit 0 (`106 passed in 12.96s`) |
| `py -3.12 -m streamlit run app.py --server.headless true --server.port 8598` | Exit 0 (Streamlit started successfully on port 8598) |

---

## 3. Genuine Blockers & Unresolved Limitations

1. **Third-Party Model Weights License**: The community checkpoint lacks a formal MIT/Apache license. For commercial deployments, users must train their own model or obtain an explicitly licensed commercial checkpoint.
2. **Real Registry Access**: Authorized vehicle data lookup requires government authorization (MoRTH/Parivahan API) with cryptographic tokens. The system uses synthetic demo data by default and does not scrape or simulate unauthorized registries.
3. **Real-World Empirical Accuracy**: End-to-end recognition accuracy in the wild requires testing against a held-out, labeled Indian number plate dataset using `evaluate.py`.
