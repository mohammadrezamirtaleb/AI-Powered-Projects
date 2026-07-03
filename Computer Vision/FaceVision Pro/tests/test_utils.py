"""
Tests for utility modules (Config, ImageIO, FPSCounter, FaceOverlay)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


# ── Config tests ────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_values(self):
        from facevision.utils.config import Config
        cfg = Config()
        assert cfg.get("detector.method") == "dnn"
        assert cfg.get("detector.confidence_threshold") == 0.60

    def test_set_and_get(self):
        from facevision.utils.config import Config
        cfg = Config()
        cfg.set("detector.method", "haar")
        assert cfg.get("detector.method") == "haar"

    def test_get_missing_returns_default(self):
        from facevision.utils.config import Config
        cfg = Config()
        result = cfg.get("nonexistent.key", default="fallback")
        assert result == "fallback"

    def test_get_section(self):
        from facevision.utils.config import Config
        cfg = Config()
        section = cfg.section("detector")
        assert isinstance(section, dict)
        assert "method" in section

    def test_deep_merge(self):
        from facevision.utils.config import _deep_merge
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 99}, "e": 5}
        _deep_merge(base, override)
        assert base["a"]["b"] == 99
        assert base["a"]["c"] == 2
        assert base["d"] == 3
        assert base["e"] == 5


# ── ImageIO tests ───────────────────────────────────────────────────────────────

class TestImageIO:
    def test_resize_to_width(self):
        from facevision.utils.io import ImageIO
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = ImageIO.resize_to_width(img, 320)
        assert resized.shape[1] == 320
        assert resized.shape[0] == 240  # Aspect preserved

    def test_resize_to_fit(self):
        from facevision.utils.io import ImageIO
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = ImageIO.resize_to_fit(img, 640, 480)
        assert resized.shape[1] <= 640
        assert resized.shape[0] <= 480

    def test_add_padding(self):
        from facevision.utils.io import ImageIO
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        padded = ImageIO.add_padding(img, 10, 10, 5, 5)
        assert padded.shape == (120, 110, 3)

    def test_read_nonexistent_returns_none(self):
        from facevision.utils.io import ImageIO
        result = ImageIO.read_image("nonexistent_file_xyz.jpg")
        assert result is None

    def test_list_images_empty_dir(self, tmp_path):
        from facevision.utils.io import ImageIO
        images = ImageIO.list_images(tmp_path)
        assert images == []


# ── FPSCounter tests ────────────────────────────────────────────────────────────

class TestFPSCounter:
    def test_returns_positive_fps(self):
        from facevision.utils.io import FPSCounter
        counter = FPSCounter()
        time.sleep(0.01)
        fps = counter.tick()
        assert fps > 0

    def test_window_capped(self):
        from facevision.utils.io import FPSCounter
        counter = FPSCounter(window=5)
        for _ in range(20):
            counter.tick()
        assert len(counter._times) <= 5


# ── FaceOverlay tests ───────────────────────────────────────────────────────────

class TestFaceOverlay:
    def make_image(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def make_face_box(self):
        from facevision.core.detector import FaceBox
        return FaceBox(x=100, y=100, w=200, h=200, confidence=0.9)

    def test_draw_hud_does_not_raise(self):
        from facevision.utils.drawing import FaceOverlay
        overlay = FaceOverlay()
        img = self.make_image()
        result = overlay.draw_hud(img, fps=30.0, face_count=1, detector_method="dnn", frame_id=1)
        assert result is img  # Modified in-place

    def test_draw_face_box_does_not_raise(self):
        from facevision.utils.drawing import FaceOverlay
        overlay = FaceOverlay()
        img = self.make_image()
        face = self.make_face_box()
        result = overlay.draw_face_box(img, face, label="Test")
        assert result is img

    def test_draw_watermark_does_not_raise(self):
        from facevision.utils.drawing import FaceOverlay
        overlay = FaceOverlay()
        img = self.make_image()
        result = overlay.draw_watermark(img)
        assert result is img

    def test_draw_emotion_bars_empty_does_not_raise(self):
        from facevision.utils.drawing import FaceOverlay
        overlay = FaceOverlay()
        img = self.make_image()
        result = overlay.draw_emotion_bars(img, {}, x=10, y=50)
        assert result is img
