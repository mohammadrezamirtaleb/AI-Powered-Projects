"""
Astra-DeepCube: High-Performance 3D Holographic Viewport & OpenGL Renderer
Provides real-time 3D interactive rendering, smooth slice rotation animations,
cyberpunk holographic shading, exploded view mode, and 360-degree orbit controls.
"""

import math
import time
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, Signal, QPointF
    from PySide6.QtGui import (
        QPainter, QColor, QPolygonF, QPen, QBrush, QLinearGradient,
        QFont, QRadialGradient, QPainterPath
    )
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


from core.cube_state import (
    CubeState, FACE_U, FACE_D, FACE_F, FACE_B, FACE_L, FACE_R,
    FACE_COLORS_RGBA, FACE_NAMES
)


# QColor equivalents with vibrant Cyberpunk Neon saturation
Q_FACE_COLORS = {
    FACE_U: QColor(245, 248, 255),       # Cyber White
    FACE_D: QColor(255, 215, 0),         # Solar Gold / Yellow
    FACE_F: QColor(0, 230, 118),         # Emerald Matrix Green
    FACE_B: QColor(0, 176, 255),         # Cyber Cobalt Blue
    FACE_L: QColor(255, 109, 0),         # Plasma Neon Orange
    FACE_R: QColor(255, 23, 68),         # Laser Crimson Red
}

CUBIE_BASE_COLOR = QColor(20, 26, 38)    # Cyber Dark Carbon
CUBIE_BORDER_COLOR = QColor(10, 14, 22)


class Cube3DViewport(QWidget if PYSIDE_AVAILABLE else object):
    """
    Hardware-accelerated 3D interactive viewport for the Rubik's Cube.
    """
    move_completed = Signal() if PYSIDE_AVAILABLE else None

    def __init__(self, parent=None, cube: Optional[CubeState] = None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self.setMinimumSize(220, 220)
            self.setMouseTracking(True)

        self.cube = cube if cube is not None else CubeState(3)
        
        # Camera Orbit Angles (Degrees)
        self.rot_x = 25.0   # Pitch
        self.rot_y = -45.0  # Yaw
        self.rot_z = 0.0
        self.zoom = 1.0
        self.exploded_factor = 0.0  # 0.0 = normal, 1.0 = fully exploded
        self.auto_spin = False

        # Mouse interaction
        self.last_mouse_pos = None
        self.is_dragging = False

        # Animation State
        self.animating_move: Optional[str] = None
        self.anim_progress = 1.0  # 0.0 to 1.0
        self.anim_speed = 0.16    # progress delta per frame
        self.anim_queue: List[str] = []

        # Hologram & Particle Effects
        self.frame_tick = 0
        self.particles: List[Dict] = []
        self._init_particles(50)

        # 60 FPS Render Timer
        if PYSIDE_AVAILABLE:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._on_render_tick)
            self.timer.start(16)  # ~60 FPS

    def _init_particles(self, count: int):
        for _ in range(count):
            self.particles.append({
                "x": np.random.uniform(-300, 300),
                "y": np.random.uniform(-300, 300),
                "z": np.random.uniform(-300, 300),
                "speed": np.random.uniform(0.5, 2.0),
                "size": np.random.uniform(1.5, 3.5),
                "alpha": np.random.uniform(40, 160)
            })

    def set_cube(self, cube: CubeState):
        """Sets a new cube state."""
        self.cube = cube
        self.animating_move = None
        self.anim_progress = 1.0
        self.anim_queue.clear()
        if PYSIDE_AVAILABLE:
            self.update()

    def set_exploded_factor(self, factor: float):
        """Sets the exploded view factor (0.0 to 1.5)."""
        self.exploded_factor = max(0.0, min(1.5, factor))
        if PYSIDE_AVAILABLE:
            self.update()

    def queue_moves(self, moves: List[str]):
        """Queues a sequence of moves for smooth animated playback."""
        self.anim_queue.extend(moves)

    def trigger_move_animation(self, move: str):
        """Starts animating a single face rotation."""
        if not move:
            return
        if self.animating_move is None:
            self.animating_move = move
            self.anim_progress = 0.0
        else:
            self.anim_queue.append(move)

    def _on_render_tick(self):
        self.frame_tick += 1

        if self.auto_spin:
            self.rot_y = (self.rot_y + 0.5) % 360.0

        # Update animation progress
        if self.animating_move is not None:
            self.anim_progress += self.anim_speed
            if self.anim_progress >= 1.0:
                self.anim_progress = 1.0
                # Apply move to internal state
                self.cube.apply_move(self.animating_move)
                self.animating_move = None
                if self.move_completed:
                    self.move_completed.emit()
        elif self.anim_queue:
            nxt = self.anim_queue.pop(0)
            self.animating_move = nxt
            self.anim_progress = 0.0

        self.update()

    # -------------------------------------------------------------------------
    # Mouse Orbit & Zoom Controls
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            self.is_dragging = True
            self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.rot_y += delta.x() * 0.6
            self.rot_x = max(-89.0, min(89.0, self.rot_x + delta.y() * 0.6))
            self.last_mouse_pos = event.position()
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(2.5, self.zoom * 1.1)
        else:
            self.zoom = max(0.4, self.zoom / 1.1)
        self.update()

    # -------------------------------------------------------------------------
    # 3D Math & Projection Pipeline
    # -------------------------------------------------------------------------
    def _project_point_3d(self, x: float, y: float, z: float, cx: float, cy: float, scale: float) -> Tuple[float, float, float]:
        """
        Rotates point (x,y,z) by camera pitch & yaw and projects to screen coordinates.
        Returns: (screen_x, screen_y, depth_z)
        """
        rad_x = math.radians(self.rot_x)
        rad_y = math.radians(self.rot_y)

        # 1. Rotate around Y axis (Yaw)
        x1 = x * math.cos(rad_y) + z * math.sin(rad_y)
        y1 = y
        z1 = -x * math.sin(rad_y) + z * math.cos(rad_y)

        # 2. Rotate around X axis (Pitch)
        x2 = x1
        y2 = y1 * math.cos(rad_x) - z1 * math.sin(rad_x)
        z2 = y1 * math.sin(rad_x) + z1 * math.cos(rad_x)

        # Perspective projection: closer points (larger z2) appear larger
        distance = 1200.0
        persp = distance / max(100.0, (distance - z2))
        
        sx = cx + x2 * scale * self.zoom * persp
        sy = cy - y2 * scale * self.zoom * persp

        return sx, sy, z2

    def _rotate_point_3d(self, x: float, y: float, z: float, axis: str, angle_deg: float) -> Tuple[float, float, float]:
        """Rotates a single 3D point around X, Y, or Z axis."""
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        if axis == "X":
            return (x, y * cos_a - z * sin_a, y * sin_a + z * cos_a)
        elif axis == "Y":
            return (x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a)
        elif axis == "Z":
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)
        return (x, y, z)

    def _cubie_belongs_to_move(self, grid_pos: Tuple[int, int, int], move: str) -> Tuple[bool, str, float]:
        """
        Determines if a cubie is affected by active move and returns (is_affected, axis, target_angle_deg).
        """
        if not move:
            return False, "", 0.0

        n = self.cube.size
        x, y, z = grid_pos

        s = move.strip()
        layer_idx = 0
        if s and s[0].isdigit():
            layer_idx = int(s[0]) - 1
            s = s[1:]

        if not s:
            return False, "", 0.0

        base_char = s[0]
        modifier = s[1:]
        base_upper = base_char.upper()

        is_counter = "'" in modifier
        is_double = "2" in modifier
        is_wide = ("w" in modifier) or base_char.islower()

        # Rotation angle for clockwise / counter-clockwise / double
        angle = 180.0 if is_double else 90.0
        if is_counter:
            angle = -angle

        # Check whole cube moves
        if base_char == "x":
            return True, "X", -angle
        elif base_char == "y":
            return True, "Y", -angle
        elif base_char == "z":
            return True, "Z", -angle

        # Face/Slice checks
        if base_upper == "U":
            if (y == n - 1 - layer_idx) or (is_wide and y >= n - 1 - (n // 2)):
                return True, "Y", -angle
        elif base_upper == "D":
            if (y == layer_idx) or (is_wide and y <= (n // 2)):
                return True, "Y", angle
        elif base_upper == "R":
            if (x == n - 1 - layer_idx) or (is_wide and x >= n - 1 - (n // 2)):
                return True, "X", -angle
        elif base_upper == "L":
            if (x == layer_idx) or (is_wide and x <= (n // 2)):
                return True, "X", angle
        elif base_upper == "F":
            if (z == n - 1 - layer_idx) or (is_wide and z >= n - 1 - (n // 2)):
                return True, "Z", -angle
        elif base_upper == "B":
            if (z == layer_idx) or (is_wide and z <= (n // 2)):
                return True, "Z", angle
        elif base_upper == "M":
            if x == n // 2:
                return True, "X", angle
        elif base_upper == "E":
            if y == n // 2:
                return True, "Y", angle
        elif base_upper == "S":
            if z == n // 2:
                return True, "Z", -angle

        return False, "", 0.0

    # -------------------------------------------------------------------------
    # Paint Pipeline
    # -------------------------------------------------------------------------
    def paintEvent(self, event):
        if not PYSIDE_AVAILABLE:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        # 1. Background Cybernetic Radial Gradient
        bg_grad = QRadialGradient(cx, cy, max(w, h) * 0.7)
        bg_grad.setColorAt(0.0, QColor(16, 22, 36))
        bg_grad.setColorAt(0.7, QColor(8, 10, 18))
        bg_grad.setColorAt(1.0, QColor(4, 5, 10))
        painter.fillRect(0, 0, w, h, bg_grad)

        # 2. Draw Floating Cyber Particles & Hologram Grid Floor
        self._draw_hologram_grid(painter, cx, cy, w, h)
        self._draw_particles(painter, cx, cy)

        # 3. Compute 3D Polygons for all Cubies & Sort by Depth
        polygons_to_draw = []
        n = self.cube.size
        cubie_size = 64.0 / n
        cubie_spacing = cubie_size * (1.0 + self.exploded_factor * 0.9)
        half_s = (cubie_size * 0.95) / 2.0

        cubies_data = self.cube.get_cubie_stickers_3d()

        # Ease-out cubic animation factor
        ease = 1.0 - math.pow(1.0 - self.anim_progress, 3)

        for cdata in cubies_data:
            grid_pos = cdata["grid_pos"]
            orig_pos = cdata["pos_3d"]
            
            # Check if this cubie is currently rotating
            is_affected, rot_axis, total_angle = self._cubie_belongs_to_move(grid_pos, self.animating_move)
            current_angle = total_angle * ease if is_affected else 0.0

            px, py, pz = orig_pos[0] * cubie_spacing, orig_pos[1] * cubie_spacing, orig_pos[2] * cubie_spacing

            # Define 6 facelets of mini cubie with consistent outward CCW winding
            # Looking at face from outside: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left
            raw_faces = [
                # U face (+Y normal)
                ("U", [(px-half_s, py+half_s, pz-half_s), (px+half_s, py+half_s, pz-half_s), (px+half_s, py+half_s, pz+half_s), (px-half_s, py+half_s, pz+half_s)]),
                # D face (-Y normal)
                ("D", [(px-half_s, py-half_s, pz+half_s), (px+half_s, py-half_s, pz+half_s), (px+half_s, py-half_s, pz-half_s), (px-half_s, py-half_s, pz-half_s)]),
                # F face (+Z normal)
                ("F", [(px-half_s, py+half_s, pz+half_s), (px+half_s, py+half_s, pz+half_s), (px+half_s, py-half_s, pz+half_s), (px-half_s, py-half_s, pz+half_s)]),
                # B face (-Z normal)
                ("B", [(px+half_s, py+half_s, pz-half_s), (px-half_s, py+half_s, pz-half_s), (px-half_s, py-half_s, pz-half_s), (px+half_s, py-half_s, pz-half_s)]),
                # L face (-X normal)
                ("L", [(px-half_s, py+half_s, pz-half_s), (px-half_s, py+half_s, pz+half_s), (px-half_s, py-half_s, pz+half_s), (px-half_s, py-half_s, pz-half_s)]),
                # R face (+X normal)
                ("R", [(px+half_s, py+half_s, pz+half_s), (px+half_s, py+half_s, pz-half_s), (px+half_s, py-half_s, pz-half_s), (px+half_s, py-half_s, pz+half_s)]),
            ]

            for face_name, verts in raw_faces:
                color_idx = cdata["colors"].get(face_name)
                is_sticker = (color_idx is not None)

                # Skip hidden internal faces when cube is solid
                if not is_sticker and self.exploded_factor < 0.05:
                    continue

                # Apply slice rotation to all 4 vertices if cubie is rotating
                if is_affected and abs(current_angle) > 0.001:
                    rotated_verts = [self._rotate_point_3d(v[0], v[1], v[2], rot_axis, current_angle) for v in verts]
                else:
                    rotated_verts = verts

                # Project 4 vertices to 2D
                proj_verts = [self._project_point_3d(v[0], v[1], v[2], cx, cy, 3.0) for v in rotated_verts]
                avg_depth = sum(p[2] for p in proj_verts) / 4.0

                # 2D cross product for backface culling in Qt screen coords (Y points down)
                v0, v1, v2 = proj_verts[0], proj_verts[1], proj_verts[2]
                cross = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
                if cross >= 0:
                    continue  # Back-facing polygon culled

                # Shading & Lighting
                if is_sticker:
                    base_color = Q_FACE_COLORS.get(color_idx, QColor(240, 240, 240))
                else:
                    base_color = CUBIE_BASE_COLOR
                
                # Directional lighting
                light_factor = 0.84 + 0.16 * math.sin(math.radians(self.rot_x) + (0.4 if face_name in ['U', 'F', 'R'] else -0.3))
                shaded_color = QColor(
                    int(max(0, min(255, base_color.red() * light_factor))),
                    int(max(0, min(255, base_color.green() * light_factor))),
                    int(max(0, min(255, base_color.blue() * light_factor))),
                    base_color.alpha()
                )

                # Painter's Algorithm: Furthest points (smaller z2) drawn first
                depth_key = avg_depth + (0.5 if is_sticker else 0.0)
                polygons_to_draw.append((depth_key, proj_verts, shaded_color, is_sticker))

        # Sort all faces in ascending order of depth (furthest to nearest)
        polygons_to_draw.sort(key=lambda item: item[0], reverse=False)

        # 4. Render Sorted Polygons
        for depth_key, verts, color, is_sticker in polygons_to_draw:
            poly = QPolygonF([QPointF(v[0], v[1]) for v in verts])
            
            painter.setBrush(QBrush(color))
            if is_sticker:
                pen = QPen(QColor(15, 20, 30, 220), 1.8)
            else:
                pen = QPen(QColor(10, 14, 20, 240), 1.0)
            
            painter.setPen(pen)
            painter.drawPolygon(poly)

        # 5. Draw Sci-Fi HUD Watermark & Controls Overlay
        self._draw_viewport_hud(painter, w, h)

    def _draw_hologram_grid(self, painter: QPainter, cx: float, cy: float, w: int, h: int):
        """Draws a subtle perspective holographic floor grid."""
        grid_pen = QPen(QColor(0, 180, 255, 18), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        
        floor_y = cy + 180
        for i in range(-5, 6):
            x1 = cx + i * 80
            painter.drawLine(int(x1), int(floor_y), int(cx + i * 200), int(h))

    def _draw_particles(self, painter: QPainter, cx: float, cy: float):
        """Draws floating neural dust / cyber particles."""
        for p in self.particles:
            p["z"] = (p["z"] + p["speed"])
            if p["z"] > 300:
                p["z"] = -300

            sx, sy, sz = self._project_point_3d(p["x"], p["y"], p["z"], cx, cy, 3.0)
            if sz > -500:
                col = QColor(0, 240, 255, int(p["alpha"]))
                painter.setBrush(QBrush(col))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(sx, sy), p["size"], p["size"])

    def _draw_viewport_hud(self, painter: QPainter, w: int, h: int):
        """Draws futuristic status overlay on top of 3D viewport."""
        font = QFont("Consolas", 9)
        font.setBold(True)
        painter.setFont(font)
        
        # Top-Left Viewport Metadata
        painter.setPen(QColor(0, 255, 200, 220))
        dim_str = f"CORE: {self.cube.size}x{self.cube.size}x{self.cube.size} MATRIX"
        painter.drawText(15, 25, dim_str)
        
        painter.setPen(QColor(180, 200, 220, 160))
        status_str = "SOLVED [100%]" if self.cube.is_solved() else f"ENTROPY: {int((1.0 - self.cube.get_solved_fraction()) * 100)}%"
        painter.drawText(15, 42, status_str)

        # Top-Right FPS & Camera stats
        cam_str = f"PITCH: {int(self.rot_x)}° | YAW: {int(self.rot_y)}° | ZOOM: {self.zoom:.1f}x"
        painter.drawText(w - 240, 25, cam_str)

        # Active Animation Indicator
        if self.animating_move:
            painter.setPen(QColor(255, 215, 0, 240))
            painter.drawText(w // 2 - 40, h - 25, f"EXECUTING: {self.animating_move}")
