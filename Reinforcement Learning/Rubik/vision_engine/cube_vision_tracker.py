"""
Astra-DeepCube: Computer Vision Tracker & Real-Time Color Segmentation
Processes camera frames, classifies facelet colors via CIELAB Delta-E / HSV, and reconstructs cube state.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from core.cube_state import (
    FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R,
    FACE_NAMES, FACE_COLORS_RGBA
)


# Target BGR Reference Colors for Standard Lighting
REFERENCE_BGR = {
    FACE_U: np.array([240, 240, 245], dtype=np.float32),  # White
    FACE_D: np.array([20, 220, 240], dtype=np.float32),   # Yellow
    FACE_F: np.array([50, 200, 0], dtype=np.float32),     # Green
    FACE_B: np.array([220, 120, 0], dtype=np.float32),    # Blue
    FACE_L: np.array([0, 120, 240], dtype=np.float32),    # Orange
    FACE_R: np.array([30, 30, 220], dtype=np.float32),    # Red
}

# Pre-computed Reference LAB Colors for Perceptual Uniformity
REFERENCE_LAB = {}
if CV2_AVAILABLE:
    for f_idx, bgr_val in REFERENCE_BGR.items():
        pixel = np.uint8([[bgr_val.astype(np.uint8)]])
        lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
        REFERENCE_LAB[f_idx] = lab_pixel[0, 0].astype(np.float32)


class CubeVisionTracker:
    """
    Real-time Computer Vision engine for cube detection and state extraction.
    """

    def __init__(self):
        self.camera_index = 0
        self.cap = None
        self.is_camera_active = False

    def open_camera(self, cam_idx: int = 0) -> bool:
        """Attempts to open physical webcam stream with low-latency settings."""
        if not CV2_AVAILABLE:
            return False
        
        self.close_camera()  # Ensure previous device handle is released

        try:
            # Try DirectShow on Windows for fast device open
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)

            if self.cap.isOpened():
                # Set low latency single-frame buffer
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_camera_active = True
                self.camera_index = cam_idx
                return True
        except Exception:
            pass
            
        self.close_camera()
        return False

    def close_camera(self):
        """Releases the camera device safely."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_camera_active = False

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a single fresh frame from the camera."""
        if not self.is_camera_active or self.cap is None:
            return None
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        except Exception:
            pass
        
        # Read failed / device disconnected
        self.close_camera()
        return None

    def classify_bgr_color(self, bgr_pixel: np.ndarray) -> Tuple[int, float]:
        """
        Classifies BGR patch using perceptual CIELAB Delta-E distance + HSV saturation.
        Returns: (face_idx, confidence)
        """
        bgr = bgr_pixel.astype(np.uint8)

        if CV2_AVAILABLE and REFERENCE_LAB:
            # Convert sample to CIELAB and HSV
            pixel_mat = np.uint8([[bgr]])
            lab_val = cv2.cvtColor(pixel_mat, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            hsv_val = cv2.cvtColor(pixel_mat, cv2.COLOR_BGR2HSV)[0, 0]

            h, s, v = int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])

            # Special case for Low Saturation White
            if s < 45 and v > 150:
                return FACE_U, 0.96

            # CIELAB Delta-E (Euclidean distance in L*a*b* space)
            best_face = FACE_U
            min_dist = 1e9

            for face_idx, ref_lab in REFERENCE_LAB.items():
                # Weighted Delta-E giving more weight to chromatic channels a*, b*
                dl = lab_val[0] - ref_lab[0]
                da = lab_val[1] - ref_lab[1]
                db = lab_val[2] - ref_lab[2]
                dist = np.sqrt(0.5 * dl * dl + da * da + db * db)

                if dist < min_dist:
                    min_dist = dist
                    best_face = face_idx

            confidence = max(0.2, min(1.0, float(np.exp(-min_dist / 60.0))))
            return best_face, confidence
        else:
            # Fallback distance
            min_dist = 1e9
            best_face = FACE_U
            for face_idx, ref_bgr in REFERENCE_BGR.items():
                d = float(np.linalg.norm(bgr.astype(np.float32) - ref_bgr))
                if d < min_dist:
                    min_dist = d
                    best_face = face_idx
            return best_face, 0.85

    def process_face_grid(self, frame: np.ndarray, grid_size: int = 3) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Processes a camera frame with a centered NxN alignment grid.
        """
        h, w = frame.shape[:2]
        size_box = min(h, w) // 2
        start_x = (w - size_box) // 2
        start_y = (h - size_box) // 2
        cell_size = size_box // grid_size

        annotated = frame.copy()
        facelet_matrix = np.zeros((grid_size, grid_size), dtype=np.int8)
        boxes_info = []

        # Draw Futuristic Cyber Outer Reticle
        if CV2_AVAILABLE:
            cv2.rectangle(
                annotated,
                (start_x - 6, start_y - 6),
                (start_x + size_box + 6, start_y + size_box + 6),
                (0, 255, 255), 2
            )

        sample_radius = max(3, cell_size // 6)

        for r in range(grid_size):
            for c in range(grid_size):
                cx = start_x + c * cell_size + cell_size // 2
                cy = start_y + r * cell_size + cell_size // 2

                # Sample ROI avoiding glare on sticker boundaries
                roi = frame[cy - sample_radius:cy + sample_radius, cx - sample_radius:cx + sample_radius]
                if roi.size > 0:
                    avg_bgr = np.mean(roi, axis=(0, 1))
                else:
                    avg_bgr = np.array([128, 128, 128], dtype=np.float32)

                color_idx, conf = self.classify_bgr_color(avg_bgr)
                facelet_matrix[r, c] = color_idx

                boxes_info.append({
                    "row": r, "col": c,
                    "center": (cx, cy),
                    "color_idx": color_idx,
                    "color_name": FACE_NAMES[color_idx],
                    "confidence": conf
                })

                if CV2_AVAILABLE:
                    pad = 4
                    bx1, by1 = cx - cell_size // 2 + pad, cy - cell_size // 2 + pad
                    bx2, by2 = cx + cell_size // 2 - pad, cy + cell_size // 2 - pad
                    
                    # HUD color (fixed integer tuple in 0..255)
                    ref_c = REFERENCE_BGR[color_idx]
                    hud_color = (int(ref_c[0]), int(ref_c[1]), int(ref_c[2]))
                    
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), hud_color, 2)
                    cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

                    cv2.putText(
                        annotated,
                        f"{FACE_NAMES[color_idx]} ({int(conf*100)}%)",
                        (bx1 + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

        return annotated, facelet_matrix, boxes_info
