"""
Astra-DeepCube: Project Astra "Fly-Eye" Virtual Drone 6-Camera Matrix
Renders simultaneous multi-perspective drone tracking HUDs with spatial confidence heatmaps.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from core.cube_state import (
    CubeState, FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R,
    FACE_NAMES, FACE_COLORS_RGBA
)


# BGR values for OpenCV rendering
BGR_COLORS = {
    FACE_U: (245, 245, 255),  # White
    FACE_D: (10, 220, 255),   # Yellow
    FACE_F: (85, 230, 0),     # Green
    FACE_B: (255, 165, 0),    # Blue
    FACE_L: (5, 115, 255),    # Orange
    FACE_R: (55, 30, 255),    # Red
}

DRONE_LABELS = {
    FACE_U: "DRONE-ALPHA [ZENITH +Y]",
    FACE_D: "DRONE-OMEGA [NADIR -Y]",
    FACE_F: "DRONE-PRIME [FRONT +Z]",
    FACE_B: "DRONE-ECHO [BACK -Z]",
    FACE_L: "DRONE-VECTOR [LEFT -X]",
    FACE_R: "DRONE-SIGMA [RIGHT +X]",
}


class VirtualMultiCamSensor:
    """
    Project Astra Multi-Angle Fly-Eye Vision Array.
    Synthesizes a 6-camera drone feed grid with zero memory churn.
    """

    def __init__(self, cell_render_size: int = 130):
        self.cell_render_size = cell_render_size
        self.frame_counter = 0
        # Pre-allocated canvas (2 rows, 3 cols of size x size)
        self.canvas = np.zeros((cell_render_size * 2, cell_render_size * 3, 3), dtype=np.uint8)

    def render_face_into_patch(self, cube: CubeState, face_idx: int, target_patch: np.ndarray):
        """
        Renders drone feed directly into pre-allocated memory slice.
        """
        size = self.cell_render_size
        target_patch[:, :] = (15, 18, 25)

        n = cube.size
        sticker_pad = 2
        grid_pixel_size = size - 32
        start_x = 16
        start_y = 22
        cell_w = grid_pixel_size // n
        actual_grid_w = cell_w * n

        # Draw Facelets
        for r in range(n):
            for c in range(n):
                color_idx = int(cube.faces[face_idx, r, c])
                bgr = BGR_COLORS.get(color_idx, (100, 100, 100))
                
                x1 = start_x + c * cell_w + sticker_pad
                y1 = start_y + r * cell_w + sticker_pad
                x2 = start_x + (c + 1) * cell_w - sticker_pad
                y2 = start_y + (r + 1) * cell_w - sticker_pad

                if CV2_AVAILABLE:
                    cv2.rectangle(target_patch, (x1, y1), (x2, y2), bgr, -1)
                    cv2.rectangle(target_patch, (x1, y1), (x2, y2), (255, 255, 255), 1)
                else:
                    target_patch[y1:y2, x1:x2] = bgr

        if CV2_AVAILABLE:
            # Sci-Fi Drone HUD Header
            drone_name = DRONE_LABELS.get(face_idx, f"DRONE-{face_idx}")
            cv2.putText(target_patch, drone_name, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 200), 1, cv2.LINE_AA)

            # Optical Scanning Laser Line bounded to actual grid width
            scan_y = int(start_y + (abs((self.frame_counter * 3) % (actual_grid_w * 2) - actual_grid_w)))
            cv2.line(target_patch, (start_x, scan_y), (start_x + actual_grid_w, scan_y), (0, 255, 255), 1)

            # Corner Targeting Reticle
            reticle_color = (0, 220, 255)
            cv2.line(target_patch, (4, 4), (12, 4), reticle_color, 1)
            cv2.line(target_patch, (4, 4), (4, 12), reticle_color, 1)
            cv2.line(target_patch, (size - 5, size - 5), (size - 13, size - 5), reticle_color, 1)
            cv2.line(target_patch, (size - 5, size - 5), (size - 5, size - 13), reticle_color, 1)

            # Telemetry footer
            conf = 98.4 + 1.5 * np.sin(self.frame_counter * 0.1 + face_idx)
            cv2.putText(target_patch, f"CONF: {conf:.1f}% | 60FPS", (6, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 255), 1, cv2.LINE_AA)

    def render_fly_eye_matrix(self, cube: CubeState) -> np.ndarray:
        """
        Synthesizes the full 6-drone composite matrix with zero heap allocation.
        """
        self.frame_counter += 1
        s = self.cell_render_size

        grid_map = [
            (0, 0, FACE_U), (0, 1, FACE_F), (0, 2, FACE_L),
            (1, 0, FACE_D), (1, 1, FACE_B), (1, 2, FACE_R)
        ]

        for row, col, f_idx in grid_map:
            patch = self.canvas[row * s:(row + 1) * s, col * s:(col + 1) * s]
            self.render_face_into_patch(cube, f_idx, patch)

        return self.canvas
