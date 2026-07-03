"""
Custom OpenCV Drawing Utilities
================================
Provides a rich, HUD-style overlay for face analysis results.

Color Palette (BGR):
    NEON_CYAN   = (255, 220, 0)
    NEON_GREEN  = (0, 255, 128)
    NEON_PINK   = (180, 0, 255)
    NEON_ORANGE = (0, 165, 255)
    WHITE       = (255, 255, 255)
    DARK_BG     = (20, 20, 20)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Color palette (BGR) ────────────────────────────────────────────────────────
CLR_CYAN = (230, 210, 0)
CLR_GREEN = (50, 220, 100)
CLR_PINK = (200, 50, 255)
CLR_ORANGE = (30, 150, 255)
CLR_YELLOW = (0, 220, 220)
CLR_WHITE = (240, 240, 240)
CLR_DARK = (20, 20, 30)
CLR_RED = (50, 50, 220)
CLR_BLUE = (220, 120, 50)

EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "happy": CLR_YELLOW,
    "sad": CLR_BLUE,
    "angry": CLR_RED,
    "surprise": CLR_ORANGE,
    "fear": CLR_PINK,
    "disgust": CLR_GREEN,
    "neutral": CLR_WHITE,
    "unknown": CLR_CYAN,
}


class FaceOverlay:
    """
    Rich, HUD-style OpenCV drawing utilities for face analysis results.

    All methods modify the image in place and return it for chaining.

    Example:
        >>> overlay = FaceOverlay()
        >>> frame = overlay.draw_face_box(frame, face_box)
        >>> frame = overlay.draw_analysis_panel(frame, face_box, analysis, recognition, pose)
    """

    def __init__(self, alpha: float = 0.75) -> None:
        """
        Args:
            alpha: Transparency of overlay panels (0=fully transparent, 1=opaque).
        """
        self.alpha = alpha

    # ── Bounding box ────────────────────────────────────────────────────────────

    def draw_face_box(
        self,
        image: np.ndarray,
        face_box,
        color: Optional[Tuple[int, int, int]] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw a stylised corner-bracket bounding box around a face."""
        x, y, w, h = face_box.x, face_box.y, face_box.w, face_box.h
        c = color or CLR_CYAN
        thickness = 2
        corner = min(w, h) // 5

        # Corner brackets
        pts = [
            # Top-left
            ((x, y + corner), (x, y), (x + corner, y)),
            # Top-right
            ((x + w - corner, y), (x + w, y), (x + w, y + corner)),
            # Bottom-right
            ((x + w, y + h - corner), (x + w, y + h), (x + w - corner, y + h)),
            # Bottom-left
            ((x + corner, y + h), (x, y + h), (x, y + h - corner)),
        ]
        for p1, p2, p3 in pts:
            cv2.line(image, p1, p2, c, thickness, cv2.LINE_AA)
            cv2.line(image, p2, p3, c, thickness, cv2.LINE_AA)

        # Center crosshair
        cx, cy = face_box.center
        cv2.drawMarker(image, (cx, cy), c, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

        # Confidence badge
        conf_text = f"{face_box.confidence:.0%}"
        cv2.putText(
            image, conf_text, (x + 4, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA,
        )

        if label:
            self._draw_label_banner(image, label, x, y, w, c)

        return image

    def _draw_label_banner(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        w: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a semi-transparent label banner above a bounding box."""
        font = cv2.FONT_HERSHEY_DUPLEX
        scale, thickness = 0.55, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        bx1, by1 = x, max(0, y - th - 12)
        bx2, by2 = x + max(tw + 12, w), y
        overlay = image.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), CLR_DARK, -1)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
        cv2.putText(
            image, text, (bx1 + 6, by2 - 5),
            font, scale, color, thickness, cv2.LINE_AA,
        )

    # ── Landmarks ───────────────────────────────────────────────────────────────

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmark_result,
        groups: Optional[List[str]] = None,
        draw_dots: bool = True,
    ) -> np.ndarray:
        """Draw facial landmark points and contours."""
        from facevision.core.landmarks import LANDMARK_GROUPS

        groups = groups or list(LANDMARK_GROUPS.keys())
        group_colors = {
            "left_eye": CLR_CYAN,
            "right_eye": CLR_CYAN,
            "lips_outer": CLR_PINK,
            "nose": CLR_ORANGE,
            "left_eyebrow": CLR_GREEN,
            "right_eyebrow": CLR_GREEN,
            "face_oval": (80, 80, 80),
        }

        for group_name in groups:
            pts = landmark_result.get_group(group_name)
            if len(pts) == 0:
                continue
            color = group_colors.get(group_name, CLR_WHITE)
            ipts = pts.astype(int)

            if group_name in ("face_oval", "lips_outer"):
                # Draw polyline for contour groups
                cv2.polylines(image, [ipts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
            elif group_name in ("left_eye", "right_eye"):
                cv2.polylines(image, [ipts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

            if draw_dots:
                for pt in ipts:
                    cv2.circle(image, tuple(pt), 1, color, -1, cv2.LINE_AA)

        return image

    # ── Analysis info panel ─────────────────────────────────────────────────────

    def draw_analysis_panel(
        self,
        image: np.ndarray,
        face_box,
        analysis=None,
        recognition=None,
        pose=None,
        landmark_result=None,
    ) -> np.ndarray:
        """
        Draw a rich info panel to the right of the face bounding box.
        """
        x2 = face_box.x2
        y = face_box.y
        panel_x = min(x2 + 10, image.shape[1] - 200)
        panel_y = y
        line_h = 22
        lines: List[Tuple[str, Tuple[int, int, int]]] = []

        # ── Recognition
        if recognition:
            name_color = CLR_GREEN if recognition.is_known else CLR_ORANGE
            lines.append((f"ID: {recognition.display_name}", name_color))
            lines.append((f"Conf: {recognition.confidence:.0%} ({recognition.confidence_label})", CLR_WHITE))

        # ── Analysis
        if analysis and analysis.success:
            emo_color = EMOTION_COLORS.get(analysis.dominant_emotion.lower(), CLR_WHITE)
            lines.append((f"Emotion: {analysis.emotion_emoji} {analysis.dominant_emotion.title()}", emo_color))
            lines.append((f"Age: ~{analysis.age} ({analysis.age_range})", CLR_YELLOW))
            lines.append((f"Gender: {analysis.gender}", CLR_CYAN))

        # ── Pose
        if pose and pose.success:
            lines.append((f"Yaw: {pose.yaw:+.1f}°  Pitch: {pose.pitch:+.1f}°", CLR_ORANGE))
            lines.append((f"Roll: {pose.roll:+.1f}°", CLR_ORANGE))
            lines.append((pose.direction, CLR_WHITE))

        # ── Blink
        if landmark_result:
            ear_avg = (landmark_result.left_ear + landmark_result.right_ear) / 2
            blink_label = "👁 Blinking" if landmark_result.is_blinking else "👁 Open"
            lines.append((f"{blink_label}  EAR:{ear_avg:.2f}", CLR_CYAN))
            lines.append((f"Blinks: {landmark_result.blink_count}", CLR_WHITE))

        if not lines:
            return image

        # Panel background
        panel_w = 220
        panel_h = len(lines) * line_h + 14
        x1p = max(0, panel_x)
        y1p = max(0, panel_y)
        x2p = min(image.shape[1], x1p + panel_w)
        y2p = min(image.shape[0], y1p + panel_h)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1p, y1p), (x2p, y2p), CLR_DARK, -1)
        cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0, image)
        cv2.rectangle(image, (x1p, y1p), (x2p, y2p), CLR_CYAN, 1, cv2.LINE_AA)

        for i, (text, color) in enumerate(lines):
            ty = y1p + 14 + i * line_h
            cv2.putText(
                image, text, (x1p + 6, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

        return image

    # ── Pose axes ───────────────────────────────────────────────────────────────

    def draw_pose_axes(
        self,
        image: np.ndarray,
        axis_points: Optional[Tuple],
    ) -> np.ndarray:
        """Draw 3D head pose axes on the image."""
        if axis_points is None:
            return image
        origin, x_pt, y_pt, z_pt = axis_points
        cv2.arrowedLine(image, tuple(origin), tuple(x_pt), CLR_RED, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(image, tuple(origin), tuple(y_pt), CLR_GREEN, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(image, tuple(origin), tuple(z_pt), CLR_BLUE, 2, cv2.LINE_AA, tipLength=0.2)
        return image

    # ── HUD overlay ─────────────────────────────────────────────────────────────

    def draw_hud(
        self,
        image: np.ndarray,
        fps: float,
        face_count: int,
        detector_method: str,
        frame_id: int = 0,
    ) -> np.ndarray:
        """Draw a global HUD bar at the top of the frame."""
        h, w = image.shape[:2]
        bar_h = 38

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), CLR_DARK, -1)
        cv2.addWeighted(overlay, 0.80, image, 0.20, 0, image)
        cv2.line(image, (0, bar_h), (w, bar_h), CLR_CYAN, 1, cv2.LINE_AA)

        left_text = f"  FaceVision Pro  |  {detector_method.upper()} Detector  |  Faces: {face_count}"
        right_text = f"FPS: {fps:.1f}  |  Frame #{frame_id}"

        cv2.putText(image, left_text, (8, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, CLR_CYAN, 1, cv2.LINE_AA)
        (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.putText(image, right_text, (w - rw - 10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, CLR_GREEN, 1, cv2.LINE_AA)

        return image

    # ── Emotion bar ─────────────────────────────────────────────────────────────

    def draw_emotion_bars(
        self,
        image: np.ndarray,
        emotion_scores: Dict[str, float],
        x: int,
        y: int,
        width: int = 160,
    ) -> np.ndarray:
        """Draw horizontal emotion probability bars."""
        if not emotion_scores:
            return image
        line_h = 18
        for i, (emotion, score) in enumerate(sorted(emotion_scores.items(), key=lambda kv: kv[1], reverse=True)):
            ty = y + i * line_h
            bar_w = int(score / 100.0 * width) if score > 1 else int(score * width)
            color = EMOTION_COLORS.get(emotion.lower(), CLR_WHITE)
            cv2.rectangle(image, (x, ty - 10), (x + bar_w, ty - 2), color, -1)
            cv2.rectangle(image, (x, ty - 10), (x + width, ty - 2), CLR_WHITE, 1)
            cv2.putText(image, f"{emotion[:7]}", (x + width + 4, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return image

    # ── Watermark ───────────────────────────────────────────────────────────────

    def draw_watermark(self, image: np.ndarray) -> np.ndarray:
        """Draw a subtle watermark in the bottom-right corner."""
        h, w = image.shape[:2]
        text = "FaceVision Pro v1.0"
        cv2.putText(
            image, text, (w - 175, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA,
        )
        return image
