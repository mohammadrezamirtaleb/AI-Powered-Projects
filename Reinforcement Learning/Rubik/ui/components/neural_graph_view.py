"""
Astra-DeepCube: Neural A* Graph & Live Telemetry Visualizer
Visualizes real-time heuristic search metrics, action probabilities, solve progress, and state entropy.
"""

from typing import Dict, List, Optional
import numpy as np

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGridLayout, QSizePolicy
    )
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


ACTION_NAMES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]


class PolicyBarsCanvas(QWidget if PYSIDE_AVAILABLE else object):
    """Clean isolated canvas for rendering policy probability distribution bars."""
    def __init__(self, parent=None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.setMinimumHeight(65)
        self.q_values = np.ones(12) / 12.0
        self.active_move = ""

    def set_q_values(self, q_values: np.ndarray, active_move: str = ""):
        self.q_values = np.array(q_values, dtype=float)
        self.active_move = active_move
        if PYSIDE_AVAILABLE:
            self.update()

    def paintEvent(self, event):
        if not PYSIDE_AVAILABLE:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()
        if h < 20 or w < 30:
            return

        # Background frame
        painter.fillRect(0, 0, w, h, QColor(11, 14, 23))
        painter.setPen(QPen(QColor(26, 38, 61), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Softmax normalization for clean probability distribution
        q = self.q_values
        if len(q) == 12:
            shift_q = q - np.max(q)
            exp_q = np.exp(np.clip(shift_q, -20.0, 20.0))
            probs = exp_q / np.sum(exp_q)
        else:
            probs = np.ones(12) / 12.0

        num_actions = len(ACTION_NAMES)
        pad = 3
        bar_w = (w - pad * (num_actions + 1)) / max(1, num_actions)
        max_prob = max(0.01, float(np.max(probs)))
        label_h = 13
        chart_h = h - label_h - 6

        for i, name in enumerate(ACTION_NAMES):
            bx = pad + i * (bar_w + pad)
            prob = float(probs[i]) if i < len(probs) else 0.0
            bh = max(3.0, (prob / max_prob) * (chart_h - 4))
            by = chart_h - bh + 2

            is_active = (name == self.active_move)
            is_best = (prob == max_prob and prob > 0.12)

            grad = QLinearGradient(bx, by, bx, by + bh)
            if is_active:
                # Active rotating action: Solar Gold to Cyber Green
                grad.setColorAt(0.0, QColor(255, 215, 0))
                grad.setColorAt(0.5, QColor(0, 255, 200))
                grad.setColorAt(1.0, QColor(0, 230, 118))
            elif is_best:
                # Highest neural probability: Neon Cyan
                grad.setColorAt(0.0, QColor(0, 255, 255))
                grad.setColorAt(1.0, QColor(0, 130, 240))
            else:
                # Regular candidate: Cobalt Blue
                grad.setColorAt(0.0, QColor(70, 110, 190))
                grad.setColorAt(1.0, QColor(24, 38, 70))

            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bx, by, bar_w, bh, 2, 2)

            # Move label
            painter.setFont(QFont("Consolas", 7, QFont.Bold if (is_active or is_best) else QFont.Normal))
            label_col = QColor(255, 215, 0) if is_active else (QColor(0, 255, 220) if is_best else QColor(160, 185, 210))
            painter.setPen(label_col)
            painter.drawText(int(bx), int(h - label_h), int(bar_w), label_h, Qt.AlignCenter, name)


class NeuralGraphWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Real-time RL Telemetry & Search Sensor Dashboard.
    Tracks live solve progress, neural cost h(s), tree expansion, and action probabilities.
    """

    def __init__(self, parent=None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._init_ui()

        self.stats = {
            "nodes_expanded": 0,
            "queue_size": 0,
            "current_g": 0,
            "current_h": 0.0,
            "entropy": "0.0%",
            "progress_pct": 0.0,
            "progress_text": "READY / IDLE",
            "active_move": "",
            "confidence": 0.0
        }

    def minimumSizeHint(self):
        return QSize(160, 100) if PYSIDE_AVAILABLE else None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # 1. Header
        header = QLabel("NEURAL SEARCH // REAL-TIME SENSORS")
        header.setFont(QFont("Consolas", 8, QFont.Bold))
        header.setStyleSheet("color: #00E5FF; letter-spacing: 0.5px;")
        header.setMinimumWidth(0)
        layout.addWidget(header)

        # 2. Live Solve Progress Bar & HUD
        self.progress_container = QWidget()
        prog_layout = QVBoxLayout(self.progress_container)
        prog_layout.setContentsMargins(0, 0, 0, 0)
        prog_layout.setSpacing(2)

        self.lbl_progress_info = QLabel("SOLVE PROGRESS: 0% [IDLE]")
        self.lbl_progress_info.setFont(QFont("Consolas", 7, QFont.Bold))
        self.lbl_progress_info.setStyleSheet("color: #00E676;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #0F1422;
                border: 1px solid #1E2D4A;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00B0FF, stop:1 #00E676);
                border-radius: 2px;
            }
        """)

        prog_layout.addWidget(self.lbl_progress_info)
        prog_layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_container)

        # 3. 4-Sensor Metric Grid
        self.metrics_container = QWidget()
        grid = QGridLayout(self.metrics_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)

        self.lbl_nodes = self._create_metric_label("NODES", "0")
        self.lbl_depth = self._create_metric_label("DEPTH g", "0")
        self.lbl_heuristic = self._create_metric_label("COST h", "0.00")
        self.lbl_entropy = self._create_metric_label("ENTROPY", "0.0%")

        grid.addWidget(self.lbl_nodes, 0, 0)
        grid.addWidget(self.lbl_depth, 0, 1)
        grid.addWidget(self.lbl_heuristic, 1, 0)
        grid.addWidget(self.lbl_entropy, 1, 1)

        layout.addWidget(self.metrics_container)

        # 4. Action & Confidence Status Bar
        self.lbl_action_hud = QLabel("NEXT ACTION: -- | CONFIDENCE: --")
        self.lbl_action_hud.setFont(QFont("Consolas", 7, QFont.Bold))
        self.lbl_action_hud.setStyleSheet("color: #FFD700; margin-top: 1px;")
        self.lbl_action_hud.setMinimumWidth(0)
        layout.addWidget(self.lbl_action_hud)

        # 5. Policy Bars Canvas
        self.bars_canvas = PolicyBarsCanvas()
        layout.addWidget(self.bars_canvas, stretch=1)

    def _create_metric_label(self, title: str, init_val: str) -> QLabel:
        lbl = QLabel(f"<b>{title}:</b> <span style='color:#00E676'>{init_val}</span>")
        lbl.setFont(QFont("Consolas", 7))
        lbl.setStyleSheet("""
            background-color: #0F1422;
            border: 1px solid #1E2D4A;
            border-radius: 3px;
            padding: 2px 4px;
            color: #A0C0E0;
        """)
        return lbl

    def update_telemetry(self, stats: Dict):
        """Updates live search statistics and triggers repaint of policy bars."""
        self.stats.update(stats)

        # 1. Update Progress
        pct = float(self.stats.get("progress_pct", 0.0))
        prog_text = self.stats.get("progress_text", f"PROGRESS: {pct:.1f}%")
        
        if PYSIDE_AVAILABLE:
            self.progress_bar.setValue(int(pct))
            self.lbl_progress_info.setText(f"SOLVE PROGRESS: {prog_text}")

            # 2. Update Metrics
            nodes = self.stats.get("nodes_expanded", 0)
            self.lbl_nodes.setText(f"<b>NODES:</b> <span style='color:#00E676'>{nodes:,}</span>")

            depth = self.stats.get("current_g", 0)
            self.lbl_depth.setText(f"<b>DEPTH g:</b> <span style='color:#00E5FF'>{depth}</span>")

            h_val = float(self.stats.get("current_h", 0.0))
            h_color = "#00E676" if h_val < 0.01 else ("#FFD700" if h_val <= 6.0 else "#FF4081")
            self.lbl_heuristic.setText(f"<b>COST h:</b> <span style='color:{h_color}'>{h_val:.2f}</span>")

            ent = self.stats.get("entropy", "0.0%")
            self.lbl_entropy.setText(f"<b>ENTROPY:</b> <span style='color:#FF9100'>{ent}</span>")

            # 3. Update Action & Confidence
            act = self.stats.get("active_move", "")
            conf = float(self.stats.get("confidence", 0.0))
            if act:
                self.lbl_action_hud.setText(f"ACTIVE MOVE: <b style='color:#00E676'>{act}</b> | CONF: <b style='color:#00E5FF'>{conf:.1f}%</b>")
            else:
                self.lbl_action_hud.setText("POLICY DISTRIBUTION P(a|s)")

            # 4. Update Policy Bars
            if "q_values" in stats:
                self.bars_canvas.set_q_values(stats["q_values"], active_move=act)

            self.update()

