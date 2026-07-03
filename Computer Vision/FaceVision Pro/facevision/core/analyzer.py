"""
Face Analyzer Module
====================
Estimates age, gender, emotion, and race using DeepFace.

Supports:
    - Emotion: angry, disgust, fear, happy, sad, surprise, neutral
    - Gender: Man / Woman
    - Age: integer estimate
    - Race: asian, indian, black, white, middle eastern, latino hispanic
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace

    _DEEPFACE_AVAILABLE = True
except ImportError:
    _DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not installed. Install with: pip install deepface")

# Analysis cooldown — run analysis only every N seconds to avoid lag
_ANALYSIS_COOLDOWN_SECONDS = 1.5


@dataclass
class AnalysisResult:
    """Stores DeepFace analysis results for one face."""

    emotion: str = "unknown"
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "unknown"
    age: int = 0
    gender: str = "unknown"
    gender_confidence: float = 0.0
    race: str = "unknown"
    race_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = False

    @property
    def age_range(self) -> str:
        """Return a human-readable age range string."""
        age = self.age
        if age < 13:
            return "Child (<13)"
        elif age < 20:
            return "Teen (13-19)"
        elif age < 30:
            return "Young Adult (20s)"
        elif age < 40:
            return "Adult (30s)"
        elif age < 50:
            return "Adult (40s)"
        elif age < 60:
            return "Middle-aged (50s)"
        else:
            return f"Senior ({age}+)"

    @property
    def emotion_emoji(self) -> str:
        """Return emoji corresponding to dominant emotion."""
        emoji_map = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "surprise": "😲",
            "fear": "😨",
            "disgust": "🤢",
            "neutral": "😐",
            "unknown": "❓",
        }
        return emoji_map.get(self.dominant_emotion.lower(), "❓")


class FaceAnalyzer:
    """
    Analyzes face attributes: emotion, age, gender, and race using DeepFace.

    Uses a cooldown mechanism to prevent running expensive inference on every frame.
    Results are cached between cooldown periods.

    Args:
        actions: List of analysis tasks to run. Options: "emotion", "age", "gender", "race".
        cooldown_seconds: Minimum seconds between full analyses per face ID.
        enforce_detection: Whether DeepFace should raise errors if no face found.

    Example:
        >>> analyzer = FaceAnalyzer(actions=["emotion", "age", "gender"])
        >>> result = analyzer.analyze(face_crop, face_id=0)
        >>> print(result.dominant_emotion, result.age, result.gender)
    """

    def __init__(
        self,
        actions: Optional[List[str]] = None,
        cooldown_seconds: float = _ANALYSIS_COOLDOWN_SECONDS,
        enforce_detection: bool = False,
    ) -> None:
        self.actions = actions or ["emotion", "age", "gender"]
        self.cooldown_seconds = cooldown_seconds
        self.enforce_detection = enforce_detection
        self._cache: Dict[int, AnalysisResult] = {}
        self._last_run: Dict[int, float] = {}

        if not _DEEPFACE_AVAILABLE:
            logger.error("DeepFace is not installed. Analysis disabled.")

    def analyze(
        self, face_image: np.ndarray, face_id: int = 0
    ) -> AnalysisResult:
        """
        Analyze a cropped face image.

        Args:
            face_image: BGR face crop as NumPy array.
            face_id: Unique ID to associate with this face for caching.

        Returns:
            AnalysisResult with predicted attributes.
        """
        if not _DEEPFACE_AVAILABLE:
            return AnalysisResult()

        if face_image is None or face_image.size == 0:
            return AnalysisResult()

        now = time.time()
        last = self._last_run.get(face_id, 0)

        # Return cached result within cooldown window
        if (now - last) < self.cooldown_seconds and face_id in self._cache:
            return self._cache[face_id]

        try:
            # DeepFace expects RGB
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            raw: List[Dict[str, Any]] = DeepFace.analyze(
                img_path=rgb,
                actions=self.actions,
                enforce_detection=self.enforce_detection,
                silent=True,
            )
            data: Dict[str, Any] = raw[0] if isinstance(raw, list) else raw

            result = AnalysisResult(
                emotion=data.get("dominant_emotion", "unknown"),
                emotion_scores=data.get("emotion", {}),
                dominant_emotion=data.get("dominant_emotion", "unknown"),
                age=int(data.get("age", 0)),
                gender=data.get("dominant_gender", "unknown"),
                gender_confidence=float(
                    max(data.get("gender", {}).values()) if data.get("gender") else 0.0
                ),
                race=data.get("dominant_race", "unknown"),
                race_scores=data.get("race", {}),
                timestamp=now,
                success=True,
            )
        except Exception as exc:
            logger.debug("DeepFace analysis error for face %d: %s", face_id, exc)
            result = self._cache.get(face_id, AnalysisResult(success=False))

        self._cache[face_id] = result
        self._last_run[face_id] = now
        return result

    def clear_cache(self) -> None:
        """Clear all cached analysis results."""
        self._cache.clear()
        self._last_run.clear()

    def get_cached(self, face_id: int) -> Optional[AnalysisResult]:
        """Return cached result for a face without triggering new analysis."""
        return self._cache.get(face_id)
