"""
Face Recognizer Module
======================
Identifies faces by comparing dlib 128-d embeddings against a database
of known (enrolled) faces.

Usage:
    1. Enroll faces using `enroll()` with a name and face crop.
    2. Save/load the database using `save_database()` / `load_database()`.
    3. Recognize faces using `recognize()`.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import face_recognition

    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False
    logger.warning(
        "face_recognition library not available. "
        "Install with: pip install face-recognition"
    )

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
DATABASE_PATH = DATA_DIR / "face_database.pkl"


@dataclass
class RecognitionResult:
    """Result of a face recognition attempt."""

    name: str
    confidence: float  # similarity score 0–1 (higher = more similar)
    is_known: bool
    face_id: int = 0

    @property
    def display_name(self) -> str:
        if not self.is_known:
            return "Unknown"
        return self.name

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.85:
            return "High"
        elif self.confidence >= 0.65:
            return "Medium"
        return "Low"


class FaceRecognizer:
    """
    Face recognition using 128-d dlib face embeddings.

    Args:
        tolerance: Maximum face distance for a match (lower = stricter, default 0.5).
        model: Embedding model — "hog" (fast) or "cnn" (accurate, GPU-recommended).
        database_path: Path to save/load the face database pickle file.

    Example:
        >>> recognizer = FaceRecognizer()
        >>> recognizer.load_database()
        >>> result = recognizer.recognize(face_crop)
        >>> print(result.name, result.confidence)
    """

    def __init__(
        self,
        tolerance: float = 0.50,
        model: str = "hog",
        database_path: Optional[Path] = None,
    ) -> None:
        self.tolerance = tolerance
        self.model = model
        self.database_path = database_path or DATABASE_PATH
        self._known_names: List[str] = []
        self._known_encodings: List[np.ndarray] = []

        if not _FR_AVAILABLE:
            logger.error("face_recognition library not installed. Recognition disabled.")

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------

    def enroll(self, name: str, image: np.ndarray) -> bool:
        """
        Enroll a new face into the recognition database.

        Args:
            name: Person's name.
            image: BGR image containing exactly one face.

        Returns:
            True if enrollment succeeded, False otherwise.
        """
        if not _FR_AVAILABLE:
            return False

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, model=self.model)
        if not encodings:
            logger.warning("No face found in enrollment image for '%s'.", name)
            return False

        self._known_names.append(name)
        self._known_encodings.append(encodings[0])
        logger.info("Enrolled face for '%s' (total: %d).", name, len(self._known_names))
        return True

    def enroll_from_file(self, name: str, image_path: Path) -> bool:
        """Enroll a face from an image file."""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error("Could not read image: %s", image_path)
            return False
        return self.enroll(name, image)

    def enroll_from_directory(self, directory: Optional[Path] = None) -> int:
        """
        Auto-enroll all images from a directory.
        Expected structure: <directory>/<person_name>/<image.jpg>

        Returns:
            Number of successfully enrolled faces.
        """
        directory = directory or KNOWN_FACES_DIR
        enrolled = 0
        for person_dir in sorted(directory.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for img_path in person_dir.glob("*.[jp][pn]g"):
                if self.enroll_from_file(name, img_path):
                    enrolled += 1
        logger.info("Enrolled %d faces from %s.", enrolled, directory)
        return enrolled

    def save_database(self, path: Optional[Path] = None) -> None:
        """Persist the face database to disk."""
        path = path or self.database_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "names": self._known_names,
                    "encodings": self._known_encodings,
                },
                f,
            )
        logger.info("Saved face database (%d entries) to %s.", len(self._known_names), path)

    def load_database(self, path: Optional[Path] = None) -> int:
        """
        Load the face database from disk.

        Returns:
            Number of loaded face entries.
        """
        path = path or self.database_path
        if not path.exists():
            logger.info("No face database found at %s. Starting fresh.", path)
            return 0
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._known_names = data.get("names", [])
        self._known_encodings = data.get("encodings", [])
        logger.info("Loaded %d faces from database.", len(self._known_names))
        return len(self._known_names)

    def remove(self, name: str) -> int:
        """Remove all entries for a given name. Returns count removed."""
        before = len(self._known_names)
        pairs = [(n, e) for n, e in zip(self._known_names, self._known_encodings) if n != name]
        if pairs:
            self._known_names, self._known_encodings = zip(*pairs)
            self._known_names = list(self._known_names)
            self._known_encodings = list(self._known_encodings)
        else:
            self._known_names = []
            self._known_encodings = []
        removed = before - len(self._known_names)
        logger.info("Removed %d entries for '%s'.", removed, name)
        return removed

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self, face_image: np.ndarray, face_id: int = 0
    ) -> RecognitionResult:
        """
        Identify a face against the enrolled database.

        Args:
            face_image: Cropped BGR face image.
            face_id: Face ID for logging.

        Returns:
            RecognitionResult with matched name and confidence.
        """
        if not _FR_AVAILABLE or not self._known_encodings:
            return RecognitionResult(name="Unknown", confidence=0.0, is_known=False, face_id=face_id)

        if face_image is None or face_image.size == 0:
            return RecognitionResult(name="Unknown", confidence=0.0, is_known=False, face_id=face_id)

        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, model=self.model)
        if not encodings:
            return RecognitionResult(name="Unknown", confidence=0.0, is_known=False, face_id=face_id)

        query_enc = encodings[0]
        distances = face_recognition.face_distance(self._known_encodings, query_enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = max(0.0, 1.0 - best_dist)
        is_known = best_dist <= self.tolerance

        name = self._known_names[best_idx] if is_known else "Unknown"
        return RecognitionResult(name=name, confidence=confidence, is_known=is_known, face_id=face_id)

    @property
    def enrolled_names(self) -> List[str]:
        """Return list of unique enrolled names."""
        return sorted(set(self._known_names))

    @property
    def database_size(self) -> int:
        """Return total number of enrolled face embeddings."""
        return len(self._known_names)
