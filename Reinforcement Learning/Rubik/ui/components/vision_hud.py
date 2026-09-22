"""
Astra-DeepCube: Project Astra Multi-Angle Drone Matrix & AR Webcam HUD
Displays real-time multi-drone tracking feeds, edge heatmaps, and physical webcam scanning.
"""

from typing import Optional
import numpy as np

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QSizePolicy
    )
    from PySide6.QtCore import Qt, QTimer, QSize
    from PySide6.QtGui import QImage, QPixmap, QFont
    import cv2
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


from core.cube_state import CubeState
from vision_engine.virtual_multicam_sensor import VirtualMultiCamSensor
from vision_engine.cube_vision_tracker import CubeVisionTracker
from vision_engine.ar_overlay import AROverlayEngine


class VisionHUDWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Holographic HUD combining 6-Drone Fly-Eye sensor feeds and Webcam AR tracking.
    """

    def __init__(self, cube: CubeState, parent=None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._init_ui()

        self.cube = cube
        self.multicam_sensor = VirtualMultiCamSensor(cell_render_size=130)
        self.vision_tracker = CubeVisionTracker()
        self.ar_engine = AROverlayEngine()
        self.active_mode = "DRONES"  # "DRONES" or "WEBCAM"
        self.current_ar_move = ""

        # Feed update timer (30 FPS)
        if PYSIDE_AVAILABLE:
            self.stream_timer = QTimer(self)
            self.stream_timer.timeout.connect(self._update_feed)
            self.stream_timer.start(33)

    def minimumSizeHint(self):
        return QSize(160, 100) if PYSIDE_AVAILABLE else None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header Bar
        header = QHBoxLayout()
        title = QLabel("PROJECT ASTRA // SPATIAL VISION HUD")
        title.setFont(QFont("Consolas", 8, QFont.Bold))
        title.setStyleSheet("color: #00E5FF; letter-spacing: 0.5px;")
        title.setMinimumWidth(0)
        header.addWidget(title)
        header.addStretch()

        # Switcher Mode Button
        self.btn_switch_mode = QPushButton("WEBCAM SCAN")
        self.btn_switch_mode.setFont(QFont("Consolas", 7, QFont.Bold))
        self.btn_switch_mode.setStyleSheet("""
            QPushButton {
                background-color: #142038;
                color: #00E5FF;
                border: 1px solid #00E5FF;
                border-radius: 3px;
                padding: 2px 6px;
            }
            QPushButton:hover {
                background-color: #1E335C;
            }
        """)
        self.btn_switch_mode.clicked.connect(self._toggle_mode)
        header.addWidget(self.btn_switch_mode)

        layout.addLayout(header)

        # Video/Image Display Label
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("""
            background-color: #0B0E17;
            border: 1px solid #1E2D4A;
            border-radius: 6px;
        """)
        self.video_display.setMinimumSize(80, 80)
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setScaledContents(False)
        layout.addWidget(self.video_display)

    def set_cube(self, cube: CubeState):
        self.cube = cube

    def set_ar_move(self, move: str):
        self.current_ar_move = move

    def _toggle_mode(self):
        if self.active_mode == "DRONES":
            success = self.vision_tracker.open_camera(0)
            if success:
                self.active_mode = "WEBCAM"
                self.btn_switch_mode.setText("SWITCH: 6-DRONE MATRIX")
            else:
                self.active_mode = "DRONES"
        else:
            self.vision_tracker.close_camera()
            self.active_mode = "DRONES"
            self.btn_switch_mode.setText("SWITCH: WEBCAM SCAN")

    def _update_feed(self):
        if not PYSIDE_AVAILABLE:
            return

        if self.active_mode == "DRONES":
            composite_bgr = self.multicam_sensor.render_fly_eye_matrix(self.cube)
            self._display_bgr_frame(composite_bgr)
        else:
            frame = self.vision_tracker.read_frame()
            if frame is not None:
                annotated, face_matrix, _ = self.vision_tracker.process_face_grid(frame, grid_size=self.cube.size)
                if self.current_ar_move:
                    h, w = frame.shape[:2]
                    annotated = self.ar_engine.draw_move_arrow(annotated, self.current_ar_move, (w // 2, h // 2))
                self._display_bgr_frame(annotated)

    def _display_bgr_frame(self, bgr_img: np.ndarray):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        
        # Deep copy to ensure buffer memory safety
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        target_w = max(10, self.video_display.width() - 8)
        target_h = max(10, self.video_display.height() - 8)

        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_display.setPixmap(scaled_pixmap)
