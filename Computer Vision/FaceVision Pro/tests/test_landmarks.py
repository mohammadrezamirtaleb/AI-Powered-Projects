"""
Tests for LandmarkDetector
"""

from __future__ import annotations

import numpy as np
import pytest


class TestLandmarkResult:
    def test_get_group_empty_on_none_points(self):
        from facevision.core.landmarks import LandmarkResult
        result = LandmarkResult(points=np.array([]))
        group = result.get_group("left_eye")
        assert len(group) == 0

    def test_get_point_none_on_out_of_range(self):
        from facevision.core.landmarks import LandmarkResult
        pts = np.zeros((10, 2), dtype=np.float32)
        result = LandmarkResult(points=pts)
        assert result.get_point(999) is None

    def test_get_point_valid(self):
        from facevision.core.landmarks import LandmarkResult
        pts = np.array([[100.0, 200.0]] * 478, dtype=np.float32)
        result = LandmarkResult(points=pts)
        pt = result.get_point(1)
        assert pt == (100, 200)

    def test_blink_count_default_zero(self):
        from facevision.core.landmarks import LandmarkResult
        result = LandmarkResult(points=np.zeros((10, 2)))
        assert result.blink_count == 0
        assert result.is_blinking is False


class TestEAR:
    def test_ear_returns_float(self):
        from facevision.core.landmarks import _eye_aspect_ratio, LEFT_EYE_EAR
        pts = np.random.rand(478, 2).astype(np.float32) * 100
        ear = _eye_aspect_ratio(pts, LEFT_EYE_EAR)
        assert isinstance(ear, float)
        assert ear >= 0.0

    def test_ear_empty_returns_zero(self):
        from facevision.core.landmarks import _eye_aspect_ratio, LEFT_EYE_EAR
        ear = _eye_aspect_ratio(np.array([]), LEFT_EYE_EAR)
        assert ear == 0.0


class TestLandmarkDetector:
    def test_instantiation(self):
        from facevision.core.landmarks import LandmarkDetector
        detector = LandmarkDetector(max_faces=3)
        assert detector.max_faces == 3

    def test_detect_blank_image_returns_empty(self):
        from facevision.core.landmarks import LandmarkDetector
        detector = LandmarkDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(img)
        assert isinstance(results, list)

    def test_detect_empty_returns_empty(self):
        from facevision.core.landmarks import LandmarkDetector
        detector = LandmarkDetector()
        result = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result == []

    def test_reset_blink_count(self):
        from facevision.core.landmarks import LandmarkDetector
        detector = LandmarkDetector()
        detector._blink_counts[0] = 5
        detector.reset_blink_count(0)
        assert detector._blink_counts[0] == 0

    def test_close_does_not_raise(self):
        from facevision.core.landmarks import LandmarkDetector
        detector = LandmarkDetector()
        detector.close()  # Should not raise
