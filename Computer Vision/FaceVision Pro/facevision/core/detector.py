"""
Face Detector Module
====================
Provides DNN-based (SSD ResNet) and Haar Cascade face detection.

The DNN detector is more accurate for varied lighting/angles.
The Haar detector is faster and works offline without model downloads.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Pre-trained DNN model files (Caffe — supported in OpenCV < 5)
DNN_PROTO = MODELS_DIR / "deploy.prototxt"
DNN_MODEL = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Haar Cascade XML — bundled in OpenCV < 5, downloaded to models/ for OpenCV 5+
_HAAR_XML_NAME = "haarcascade_frontalface_default.xml"
_HAAR_BUNDLED = Path(cv2.__file__).parent / "data" / _HAAR_XML_NAME
_HAAR_LOCAL   = MODELS_DIR / _HAAR_XML_NAME
_HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


def _get_haar_path() -> Optional[Path]:
    """Return a valid path to the Haar cascade XML, downloading if needed."""
    if _HAAR_BUNDLED.exists():
        return _HAAR_BUNDLED
    if _HAAR_LOCAL.exists():
        return _HAAR_LOCAL
    # Download from GitHub
    logger.info("Downloading Haar Cascade XML from GitHub ...")
    try:
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_HAAR_URL, str(_HAAR_LOCAL))
        logger.info("Haar Cascade saved to %s", _HAAR_LOCAL)
        return _HAAR_LOCAL
    except Exception as exc:
        logger.error("Failed to download Haar Cascade: %s", exc)
        return None


@dataclass
class FaceBox:
    """Represents a detected face bounding box with confidence score."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    face_id: Optional[int] = None

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) tuple."""
        return (self.x, self.y, self.w, self.h)

    def scale(self, factor: float) -> "FaceBox":
        """Return a scaled version of this bounding box."""
        cx, cy = self.center
        new_w = int(self.w * factor)
        new_h = int(self.h * factor)
        return FaceBox(
            x=max(0, cx - new_w // 2),
            y=max(0, cy - new_h // 2),
            w=new_w,
            h=new_h,
            confidence=self.confidence,
            face_id=self.face_id,
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the face region from an image."""
        h, w = image.shape[:2]
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(w, self.x2)
        y2 = min(h, self.y2)
        return image[y1:y2, x1:x2]


class FaceDetector:
    """
    Robust face detector supporting DNN (SSD ResNet) and Haar Cascade methods.

    Args:
        method: Detection method — "dnn" (default) or "haar".
        confidence_threshold: Minimum confidence for DNN detections (0–1).
        min_face_size: Minimum face size in pixels for Haar detection.

    Example:
        >>> detector = FaceDetector(method="dnn")
        >>> faces = detector.detect(frame)
        >>> for face in faces:
        ...     print(face.x, face.y, face.w, face.h, face.confidence)
    """

    def __init__(
        self,
        method: str = "dnn",
        confidence_threshold: float = 0.6,
        min_face_size: Tuple[int, int] = (30, 30),
    ) -> None:
        self.method = method.lower()
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self._net: Optional[cv2.dnn.Net] = None
        self._haar: Optional[cv2.CascadeClassifier] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the selected detection model."""
        if self.method == "dnn":
            if DNN_PROTO.exists() and DNN_MODEL.exists():
                # readNetFromCaffe was removed in OpenCV 5.x — try it, fall back on error
                if hasattr(cv2.dnn, "readNetFromCaffe"):
                    try:
                        logger.info("Loading DNN face detector model (Caffe) ...")
                        self._net = cv2.dnn.readNetFromCaffe(
                            str(DNN_PROTO), str(DNN_MODEL)
                        )
                        logger.info("DNN face detector loaded successfully.")
                    except Exception as exc:
                        logger.warning("Caffe model load failed (%s). Falling back to Haar.", exc)
                        self.method = "haar"
                else:
                    logger.warning(
                        "cv2.dnn.readNetFromCaffe not available in OpenCV %s. "
                        "Falling back to Haar Cascade.", cv2.__version__
                    )
                    self.method = "haar"
            else:
                logger.warning(
                    "DNN model files not found. "
                    "Run `python scripts/download_models.py` to download them. "
                    "Falling back to Haar Cascade."
                )
                self.method = "haar"

        if self.method == "haar":
            logger.info("Loading Haar Cascade face detector …")
            haar_path = _get_haar_path()
            if haar_path is None:
                 raise RuntimeError("Failed to download or locate Haar Cascade XML.")
            
            self._haar = cv2.CascadeClassifier(str(haar_path))
            if self._haar.empty():
                raise RuntimeError(
                    f"Failed to load Haar Cascade from {haar_path}. Check your OpenCV installation."
                )
            logger.info("Haar Cascade loaded successfully.")

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """
        Detect all faces in the given image.

        Args:
            image: BGR image as a NumPy array.

        Returns:
            List of FaceBox objects, sorted by area descending.
        """
        if image is None or image.size == 0:
            return []

        if self.method == "dnn" and self._net is not None:
            faces = self._detect_dnn(image)
        else:
            faces = self._detect_haar(image)

        # Assign IDs and sort by area
        for i, face in enumerate(sorted(faces, key=lambda f: f.area, reverse=True)):
            face.face_id = i
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def _detect_dnn(self, image: np.ndarray) -> List[FaceBox]:
        """DNN-based face detection using SSD ResNet."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        faces: List[FaceBox] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append(
                FaceBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=confidence)
            )
        return faces

    def _detect_haar(self, image: np.ndarray) -> List[FaceBox]:
        """Haar Cascade-based face detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if not isinstance(rects, np.ndarray):
            return []
        return [
            FaceBox(x=int(x), y=int(y), w=int(w), h=int(h), confidence=1.0)
            for x, y, w, h in rects
        ]

    def detect_largest(self, image: np.ndarray) -> Optional[FaceBox]:
        """Detect and return only the largest face in the image."""
        faces = self.detect(image)
        return faces[0] if faces else None

    @property
    def active_method(self) -> str:
        """Return the currently active detection method."""
        return self.method
