# Google Antigravity handoff — audit and complete this ANPR project

Please inspect the entire repository before editing. Treat this as a Python ANPR starter, not as a claim of production readiness.

## Goals
1. Make the app reliably install and run on Windows with Python 3.11/3.12.
2. Fix all reproducible bugs and add tests before changing architecture.
3. Keep image capture, detection, preprocessing, OCR, validation, database access, and UI modular.
4. Add a real license-plate detection model only after verifying its source, license, class labels, and performance on Indian plates. Do not silently substitute a car detector.
5. Improve OCR preprocessing (deskew/perspective correction, adaptive thresholding, glare/blur handling) and keep confidence/uncertainty explicit.
6. Build an evaluation script for a held-out dataset and report exact plate accuracy, character error rate, detection precision/recall, latency, and failure cases.
7. Add input limits, error handling, structured logs without unnecessary personal data, dependency checks, and safe file handling.
8. Keep the included CSV clearly labeled fictional demo data. Never fabricate official vehicle records or add mock data that appears real.
9. Do not scrape or bypass access controls on VAHAN/Parivahan or third-party services. Any real lookup integration must use documented, authorized API access and only expose permitted fields.
10. Add authentication, authorization, audit logging, secret management, and retention/deletion controls before any deployment involving real images or records.
11. Update README with exact verified commands, model acquisition instructions, licenses, known limitations, and test results.
12. Run tests and report each command, pass/fail output, remaining issue, and what was not tested. Do not claim production-ready without evidence.

## Acceptance criteria
- Fresh environment installation instructions are accurate.
- Unit tests pass.
- Missing weights or missing OCR dependencies produce clear errors rather than crashes.
- A valid image with a supplied compatible plate model produces a plate crop and OCR output.
- An invalid image and a no-plate image are handled safely.
- Database misses are not described as unregistered vehicles.
- No real personal data, credentials, or unverified API claims are committed.
