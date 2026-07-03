"""
Facial Landmark Detector
========================
Detects 478-point facial landmarks using MediaPipe FaceLandmarker (Tasks API).

This module supports MediaPipe 0.10.x+ which uses the Tasks API instead of
the deprecated `solutions` API. It auto-downloads the required model file
on first use.

Key landmark indices (MediaPipe 478-point model):
    LEFT_EYE  : [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, ...]
    RIGHT_EYE : [362, 382, 381, 380, 374, 373, 390, 249, 263, ...]
    LIPS      : [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, ...]
    NOSE_TIP  : 1
    CHIN      : 199
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe Tasks model asset ────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_MP_AVAILABLE = False
_mediapipe = None


def _try_load_mediapipe() -> bool:
    """Attempt to load MediaPipe. Returns True if successful."""
    global _MP_AVAILABLE, _mediapipe
    try:
        import mediapipe as mp
        _mediapipe = mp
        _MP_AVAILABLE = True
        logger.info("MediaPipe %s loaded successfully.", mp.__version__)
        return True
    except ImportError:
        logger.warning("MediaPipe not installed. Install with: pip install mediapipe")
        return False


def _download_model() -> bool:
    """Download the FaceLandmarker .task model file if not present (with retries)."""
    if FACE_LANDMARKER_MODEL.exists():
        return True
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face_landmarker.task model (~3 MB) ...")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            import sys
            import time
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 / total_size)
                    sys.stdout.write(f"\r  Downloading face_landmarker.task: {pct:.0f}%   ")
                    sys.stdout.flush()
            urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(FACE_LANDMARKER_MODEL), reporthook)
            print()
            logger.info("face_landmarker.task downloaded to %s", FACE_LANDMARKER_MODEL)
            return True
        except Exception as exc:
            logger.warning("Attempt %d/%d failed to download face_landmarker.task: %s", attempt, max_retries, exc)
            if FACE_LANDMARKER_MODEL.exists():
                FACE_LANDMARKER_MODEL.unlink()
            if attempt < max_retries:
                time.sleep(2)  # Wait before retry
            else:
                logger.error("Failed to download face_landmarker.task after %d attempts.", max_retries)
                return False
    return False


# ── Landmark group indices (MediaPipe 478-point model) ─────────────────────────
LANDMARK_GROUPS = {
    "left_eye":     [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "right_eye":    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "lips_outer":   [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
                     14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409],
    "nose":         [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275, 220, 440],
    "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "right_eyebrow":[46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "face_oval":    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
                     378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                     162, 21, 54, 103, 67, 109],
}

NOSE_TIP              = 1
CHIN                  = 199
LEFT_EYE_LEFT_CORNER  = 33
LEFT_EYE_RIGHT_CORNER = 133
RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER= 263
MOUTH_LEFT            = 61
MOUTH_RIGHT           = 291

# EAR indices for blink detection
LEFT_EYE_EAR  = [159, 145, 158, 153, 160, 144]
RIGHT_EYE_EAR = [386, 374, 387, 380, 385, 373]
EAR_BLINK_THRESHOLD = 0.25


@dataclass
class LandmarkResult:
    """Stores detected facial landmarks for one face."""

    points: np.ndarray          # shape (N, 2) — pixel coordinates
    points_3d: Optional[np.ndarray] = None   # shape (N, 3) — normalized coords
    left_ear: float = 0.0
    right_ear: float = 0.0
    is_blinking: bool = False
    blink_count: int = 0

    def get_group(self, group_name: str) -> np.ndarray:
        """Return landmark points for a named group."""
        indices = LANDMARK_GROUPS.get(group_name, [])
        if not indices or self.points is None or len(self.points) == 0:
            return np.array([])
        valid = [i for i in indices if i < len(self.points)]
        return self.points[valid]

    def get_point(self, index: int) -> Optional[Tuple[int, int]]:
        """Return a single landmark point by index."""
        if self.points is None or index >= len(self.points):
            return None
        return tuple(self.points[index].astype(int))


def _eye_aspect_ratio(landmarks: np.ndarray, eye_indices: List[int]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(landmarks) == 0:
        return 0.0
    p = [landmarks[i] for i in eye_indices if i < len(landmarks)]
    if len(p) < 6:
        return 0.0
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class LandmarkDetector:
    """
    Detects 478 facial landmarks using MediaPipe FaceLandmarker (Tasks API).

    Automatically downloads the required model file on first use.
    Compatible with MediaPipe 0.10.x+ (Tasks API).

    Args:
        max_faces: Maximum number of faces to detect landmarks for.
        min_detection_confidence: Minimum detection confidence (0-1).
        min_tracking_confidence: Minimum tracking confidence (0-1).
        refine_landmarks: Enable iris landmark refinement.

    Example:
        >>> detector = LandmarkDetector()
        >>> results = detector.detect(frame)
        >>> for result in results:
        ...     nose = result.get_point(1)
    """

    def __init__(
        self,
        max_faces: int = 5,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks
        self._detector = None
        self._blink_counts: Dict[int, int] = {}
        self._prev_blink_state: Dict[int, bool] = {}

        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize the MediaPipe FaceLandmarker."""
        if not _try_load_mediapipe():
            return
        if not _download_model():
            logger.error("Cannot initialize LandmarkDetector: model file unavailable.")
            return

        try:
            mp = _mediapipe
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._detector = FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker (Tasks API) initialized with model: %s", FACE_LANDMARKER_MODEL.name)
        except Exception as exc:
            logger.error("Failed to initialize FaceLandmarker: %s", exc)
            self._detector = None

    def detect(self, image: np.ndarray, face_id: int = 0) -> List[LandmarkResult]:
        """
        Detect landmarks for all faces in the image.

        Args:
            image: BGR image as NumPy array.
            face_id: Starting face ID for blink tracking.

        Returns:
            List of LandmarkResult objects (one per detected face).
        """
        if self._detector is None or image is None or image.size == 0:
            return []

        mp = _mediapipe
        h, w = image.shape[:2]

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = self._detector.detect(mp_image)
        except Exception as exc:
            logger.debug("FaceLandmarker detection error: %s", exc)
            return []

        if not detection_result.face_landmarks:
            return []

        outputs: List[LandmarkResult] = []
        for idx, face_lm_list in enumerate(detection_result.face_landmarks):
            pts = np.array(
                [[lm.x * w, lm.y * h] for lm in face_lm_list],
                dtype=np.float32,
            )
            pts_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_lm_list],
                dtype=np.float32,
            )

            left_ear  = _eye_aspect_ratio(pts, LEFT_EYE_EAR)
            right_ear = _eye_aspect_ratio(pts, RIGHT_EYE_EAR)
            is_blinking = left_ear < EAR_BLINK_THRESHOLD and right_ear < EAR_BLINK_THRESHOLD

            fid = face_id + idx
            was_blinking = self._prev_blink_state.get(fid, False)
            if is_blinking and not was_blinking:
                self._blink_counts[fid] = self._blink_counts.get(fid, 0) + 1
            self._prev_blink_state[fid] = is_blinking

            outputs.append(LandmarkResult(
                points=pts,
                points_3d=pts_3d,
                left_ear=left_ear,
                right_ear=right_ear,
                is_blinking=is_blinking,
                blink_count=self._blink_counts.get(fid, 0),
            ))

        return outputs

    def get_key_points(self, result: LandmarkResult, image_shape: Tuple) -> Dict[str, Tuple[int, int]]:
        """Extract named key landmark points."""
        names_to_idx = {
            "nose_tip":        NOSE_TIP,
            "chin":            CHIN,
            "left_eye_left":   LEFT_EYE_LEFT_CORNER,
            "left_eye_right":  LEFT_EYE_RIGHT_CORNER,
            "right_eye_left":  RIGHT_EYE_LEFT_CORNER,
            "right_eye_right": RIGHT_EYE_RIGHT_CORNER,
            "mouth_left":      MOUTH_LEFT,
            "mouth_right":     MOUTH_RIGHT,
        }
        return {
            name: result.get_point(idx)
            for name, idx in names_to_idx.items()
            if result.get_point(idx) is not None
        }

    def reset_blink_count(self, face_id: int = 0) -> None:
        """Reset the blink counter for a given face ID."""
        self._blink_counts[face_id] = 0

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector:
            self._detector.close()
            self._detector = None
