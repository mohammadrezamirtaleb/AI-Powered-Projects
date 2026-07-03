# Changelog

All notable changes to **FaceVision Pro** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [1.0.0] — 2026-07-02

### Added
- **Face Detection** — DNN-based SSD ResNet detector with Haar Cascade fallback
- **Facial Landmarks** — MediaPipe Face Mesh with 478-point detection
- **Blink Detection** — Eye Aspect Ratio (EAR) algorithm with blink counter
- **Emotion Analysis** — DeepFace-powered emotion classification (7 classes)
- **Age Estimation** — Real-time age prediction via DeepFace
- **Gender Estimation** — Gender classification with confidence scores
- **Face Recognition** — dlib 128-d embedding-based identification with enrollment system
- **Head Pose Estimation** — Pitch / Yaw / Roll via OpenCV solvePnP
- **Real-Time Pipeline** — Full webcam pipeline with keyboard controls
- **Batch Pipeline** — Image directory and video file processing
- **Streamlit Dashboard** — Interactive web UI with dark theme
- **CLI Scripts** — `run_webcam.py`, `run_image.py`, `enroll_face.py`, `download_models.py`
- **Configuration** — YAML-based config system with dot-key access
- **Unit Tests** — pytest suite for detector, landmarks, and utilities
- **GitHub Actions CI** — Lint + multi-platform test matrix + package build
- **Pre-commit hooks** — black, isort, flake8 automated formatting
- **MIT License**

[Unreleased]: https://github.com/yourusername/FaceVisionPro/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/FaceVisionPro/releases/tag/v1.0.0
