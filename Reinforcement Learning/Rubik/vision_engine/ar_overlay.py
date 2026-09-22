"""
Astra-DeepCube: Augmented Reality Move Visualizer
Projects dynamic 3D trajectory arrows, directional chevrons, and move annotations onto camera feeds.
"""

from typing import Tuple, List, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class AROverlayEngine:
    """
    Renders dynamic AR move trajectories with directional chevrons and layer offsets.
    """

    def __init__(self):
        self.pulse_phase = 0.0

    def draw_move_arrow(self, frame: np.ndarray, move: str, center: Tuple[int, int], radius: int = 60) -> np.ndarray:
        """
        Draws layer-localized AR rotation arcs and chevrons without mutating input buffer.
        """
        if not CV2_AVAILABLE or not move:
            return frame

        cx, cy = center
        self.pulse_phase += 0.12
        pulse = 1.0 + 0.12 * np.sin(self.pulse_phase)
        r = int(radius * pulse)

        output = frame.copy()
        base_char = move[0].upper()
        is_counter = "'" in move
        is_double = "2" in move

        # Layer-specific center offsets
        target_cx, target_cy = cx, cy
        if base_char == "U":
            target_cy -= r // 2
        elif base_char == "D":
            target_cy += r // 2
        elif base_char == "L":
            target_cx -= r // 2
        elif base_char == "R":
            target_cx += r // 2

        color_arrow = (0, 255, 255)  # Cyan
        color_glow = (0, 180, 255)

        # Pulsing target circle
        cv2.circle(output, (target_cx, target_cy), r, color_glow, 2, cv2.LINE_AA)

        # Move Hologram Text Banner
        text_str = f"AR TARGET: {move}"
        cv2.putText(output, text_str, (target_cx - 60, target_cy - r - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Arc Angles
        sweep = 300 if is_double else 180
        start_angle = 30 if is_counter else (30 + sweep)
        end_angle = (30 + sweep) if is_counter else 30

        cv2.ellipse(output, (target_cx, target_cy), (r, r), 0, min(start_angle, end_angle), max(start_angle, end_angle), color_arrow, 3, cv2.LINE_AA)

        # Directional Arrowhead
        tip_angle = np.radians(end_angle if is_counter else start_angle)
        tip_x = int(target_cx + r * np.cos(tip_angle))
        tip_y = int(target_cy + r * np.sin(tip_angle))

        # Chevron wings
        wing_len = 12
        tangent_angle = tip_angle + (np.pi / 2 if is_counter else -np.pi / 2)
        w1_x = int(tip_x - wing_len * np.cos(tangent_angle - 0.5))
        w1_y = int(tip_y - wing_len * np.sin(tangent_angle - 0.5))
        w2_x = int(tip_x - wing_len * np.cos(tangent_angle + 0.5))
        w2_y = int(tip_y - wing_len * np.sin(tangent_angle + 0.5))

        cv2.line(output, (tip_x, tip_y), (w1_x, w1_y), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(output, (tip_x, tip_y), (w2_x, w2_y), (0, 255, 0), 3, cv2.LINE_AA)

        return output
