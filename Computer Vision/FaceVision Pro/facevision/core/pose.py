"""
Head Pose Estimator
===================
Estimates head pose (pitch, yaw, roll) using OpenCV's solvePnP with
facial landmarks from MediaPipe.

Output:
    - Pitch: Nodding up/down (positive = looking down)
    - Yaw:   Turning left/right (positive = turning right)
    - Roll:  Tilting head (positive = tilting right)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# 3D reference face model points (canonical face geometry in mm)
# Matches MediaPipe landmarks: nose_tip(1), chin(199), left_eye_left(33),
# right_eye_right(263), mouth_left(61), mouth_right(291)
FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],     # Chin
        [-225.0, 170.0, -135.0],  # Left eye left corner
        [225.0, 170.0, -135.0],   # Right eye right corner
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0],  # Right mouth corner
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe landmark indices
POSE_LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]


@dataclass
class PoseResult:
    """Head pose estimation result."""

    pitch: float  # Up/down angle in degrees
    yaw: float    # Left/right angle in degrees
    roll: float   # Tilt angle in degrees
    success: bool = True

    @property
    def direction(self) -> str:
        """Return a text description of gaze direction."""
        if not self.success:
            return "Unknown"
        directions = []
        if self.pitch < -15:
            directions.append("Looking Up")
        elif self.pitch > 15:
            directions.append("Looking Down")
        if self.yaw < -20:
            directions.append("Turned Left")
        elif self.yaw > 20:
            directions.append("Turned Right")
        if abs(self.roll) > 20:
            directions.append("Tilted")
        return " / ".join(directions) if directions else "Facing Forward"

    @property
    def is_frontal(self) -> bool:
        """Return True if the face is roughly frontal."""
        return (
            abs(self.pitch) < 20
            and abs(self.yaw) < 25
            and abs(self.roll) < 20
        )


class HeadPoseEstimator:
    """
    Estimates 3D head pose (pitch, yaw, roll) from 2D facial landmarks.

    Uses OpenCV's solvePnP with a canonical 3D face model.

    Args:
        smoothing: Number of frames to smooth pose over (temporal filter).

    Example:
        >>> estimator = HeadPoseEstimator()
        >>> pose = estimator.estimate(landmark_result, image.shape)
        >>> print(pose.yaw, pose.pitch, pose.roll, pose.direction)
    """

    def __init__(self, smoothing: int = 5) -> None:
        self.smoothing = smoothing
        self._history: list = []

    def _get_camera_matrix(self, image_shape: Tuple) -> np.ndarray:
        """Estimate camera intrinsic matrix from image dimensions."""
        h, w = image_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        return np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def estimate(
        self,
        landmark_result,
        image_shape: Tuple,
    ) -> PoseResult:
        """
        Estimate head pose from landmark points.

        Args:
            landmark_result: LandmarkResult from LandmarkDetector.
            image_shape: (height, width, channels) of the source image.

        Returns:
            PoseResult with pitch, yaw, roll in degrees.
        """
        if landmark_result is None or landmark_result.points is None:
            return PoseResult(pitch=0.0, yaw=0.0, roll=0.0, success=False)

        # Extract the 6 key 2D points
        pts = landmark_result.points
        try:
            face_2d = np.array(
                [pts[i] for i in POSE_LANDMARK_INDICES],
                dtype=np.float64,
            )
        except (IndexError, TypeError):
            return PoseResult(pitch=0.0, yaw=0.0, roll=0.0, success=False)

        camera_matrix = self._get_camera_matrix(image_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            FACE_3D_MODEL,
            face_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return PoseResult(pitch=0.0, yaw=0.0, roll=0.0, success=False)

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Simpler Euler angles via atan2
        pitch = float(np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2])))
        yaw = float(np.degrees(np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1] ** 2 + rot_mat[2, 2] ** 2))))
        roll = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))

        # Temporal smoothing
        self._history.append((pitch, yaw, roll))
        if len(self._history) > self.smoothing:
            self._history.pop(0)
        avg = np.mean(self._history, axis=0)

        return PoseResult(pitch=float(avg[0]), yaw=float(avg[1]), roll=float(avg[2]), success=True)

    def reset(self) -> None:
        """Clear pose smoothing history."""
        self._history.clear()

    def get_axis_points(
        self,
        landmark_result,
        image_shape: Tuple,
        axis_length: float = 100.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Project 3D axes onto image for visualization.

        Returns:
            Tuple of (nose_tip_2d, x_axis_pt, y_axis_pt, z_axis_pt) or None.
        """
        if landmark_result is None or landmark_result.points is None:
            return None

        pts = landmark_result.points
        try:
            face_2d = np.array(
                [pts[i] for i in POSE_LANDMARK_INDICES], dtype=np.float64
            )
        except (IndexError, TypeError):
            return None

        camera_matrix = self._get_camera_matrix(image_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        axes_3d = np.float32(
            [
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, -axis_length, 0],
                [0, 0, -axis_length],
            ]
        )
        img_pts, _ = cv2.projectPoints(
            axes_3d, rot_vec, trans_vec, camera_matrix, dist_coeffs
        )
        img_pts = img_pts.reshape(-1, 2).astype(int)
        return img_pts[0], img_pts[1], img_pts[2], img_pts[3]
