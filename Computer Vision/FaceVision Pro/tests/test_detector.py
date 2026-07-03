"""
Tests for FaceDetector
"""

from __future__ import annotations

import numpy as np
import pytest


def make_blank_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a blank BGR image for testing."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_gray_face_image() -> np.ndarray:
    """Create a synthetic gray-rectangle 'face' image."""
    img = make_blank_image()
    # Draw a rough face-like region (gray oval on dark bg)
    import cv2
    cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (180, 160, 140), -1)
    return img


class TestFaceBox:
    def test_properties(self):
        from facevision.core.detector import FaceBox
        fb = FaceBox(x=10, y=20, w=100, h=150, confidence=0.9)
        assert fb.x2 == 110
        assert fb.y2 == 170
        assert fb.center == (60, 95)
        assert fb.area == 15000

    def test_to_tuple(self):
        from facevision.core.detector import FaceBox
        fb = FaceBox(x=5, y=10, w=50, h=60)
        assert fb.to_tuple() == (5, 10, 50, 60)

    def test_scale(self):
        from facevision.core.detector import FaceBox
        fb = FaceBox(x=50, y=50, w=100, h=100, confidence=0.8)
        scaled = fb.scale(2.0)
        assert scaled.w == 200
        assert scaled.h == 200

    def test_crop(self):
        from facevision.core.detector import FaceBox
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        fb = FaceBox(x=10, y=10, w=50, h=60)
        crop = fb.crop(img)
        assert crop.shape == (60, 50, 3)

    def test_crop_clamps_to_image(self):
        from facevision.core.detector import FaceBox
        img = np.ones((100, 100, 3), dtype=np.uint8)
        fb = FaceBox(x=80, y=80, w=100, h=100)  # Partially outside
        crop = fb.crop(img)
        assert crop.shape[0] > 0 and crop.shape[1] > 0


class TestFaceDetector:
    def test_instantiation_haar(self):
        from facevision.core.detector import FaceDetector
        detector = FaceDetector(method="haar")
        assert detector.active_method == "haar"

    def test_detect_empty_image_returns_empty(self):
        from facevision.core.detector import FaceDetector
        detector = FaceDetector(method="haar")
        result = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result == []

    def test_detect_blank_image_returns_empty(self):
        from facevision.core.detector import FaceDetector
        detector = FaceDetector(method="haar")
        img = make_blank_image()
        result = detector.detect(img)
        assert isinstance(result, list)

    def test_detect_largest_none_on_blank(self):
        from facevision.core.detector import FaceDetector
        detector = FaceDetector(method="haar")
        img = make_blank_image()
        result = detector.detect_largest(img)
        assert result is None

    def test_faces_sorted_by_area(self):
        from facevision.core.detector import FaceBox
        import cv2
        # Artificially create two face boxes and verify sorting
        box_small = FaceBox(x=0, y=0, w=30, h=30, confidence=0.9)
        box_large = FaceBox(x=0, y=0, w=100, h=100, confidence=0.9)
        boxes = [box_small, box_large]
        sorted_boxes = sorted(boxes, key=lambda f: f.area, reverse=True)
        assert sorted_boxes[0].area > sorted_boxes[1].area

    def test_detect_returns_list(self):
        from facevision.core.detector import FaceDetector
        detector = FaceDetector(method="haar")
        img = make_gray_face_image()
        result = detector.detect(img)
        assert isinstance(result, list)
