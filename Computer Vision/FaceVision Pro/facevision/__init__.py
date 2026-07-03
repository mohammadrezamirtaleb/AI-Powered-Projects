"""
FaceVision Pro
==============
A professional, modular Computer Vision library for real-time face analysis.

Features:
    - Face Detection (DNN + Haar Cascades)
    - Facial Landmark Detection (MediaPipe 478-point mesh)
    - Face Recognition (dlib embeddings)
    - Emotion, Age & Gender Estimation (DeepFace)
    - Head Pose Estimation
    - Blink Detection & Eye Tracking
    - Real-time webcam pipeline
    - Streamlit dashboard

Author: FaceVision Pro Contributors
License: MIT
"""

__version__ = "1.0.0"
__author__ = "FaceVision Pro"
__license__ = "MIT"

__version__ = "1.0.0"
__author__ = "FaceVision Pro"
__license__ = "MIT"

try:
    from facevision.core.detector import FaceDetector
    from facevision.core.landmarks import LandmarkDetector
    from facevision.core.analyzer import FaceAnalyzer
    from facevision.core.recognizer import FaceRecognizer
    from facevision.pipeline.realtime import RealtimePipeline
except Exception as _e:  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Some FaceVision Pro modules could not be imported: %s", _e
    )

__all__ = [
    "FaceDetector",
    "LandmarkDetector",
    "FaceAnalyzer",
    "FaceRecognizer",
    "RealtimePipeline",
]
