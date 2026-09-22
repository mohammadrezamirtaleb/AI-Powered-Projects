"""
Astra-DeepCube: Live RL Training & Loss Curve Dashboard
Enables real-time Autodidactic Iteration (ADI) training with live loss plotting and hyperparameter control.
Optimized with lazy background initialization for instantaneous application startup (Zero Emojis).
"""

from typing import Optional, List
import threading
import time

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QSlider, QSpinBox, QGroupBox, QSizePolicy
    )
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QPointF, QSize
    from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPolygonF
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


from rl_engine.autodidactic_iteration import AutodidacticTrainer
from rl_engine.deepcube_model import DeepCubeNetwork


class LiveTrainingWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Live interactive dashboard for training the Deep Neural Heuristic via ADI.
    Thread-safe signal-driven UI updates with lazy initialization.
    """
    metrics_updated = Signal(dict) if PYSIDE_AVAILABLE else None

    def __init__(self, model: DeepCubeNetwork, parent=None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._init_ui()

        self.model = model
        self.trainer: Optional[AutodidacticTrainer] = None  # Lazy init to eliminate startup lag
        self.is_training_active = False
        self.loss_history: List[float] = []

        if PYSIDE_AVAILABLE and self.metrics_updated:
            self.metrics_updated.connect(self._on_metrics_received)

    def minimumSizeHint(self):
        return QSize(160, 100) if PYSIDE_AVAILABLE else None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header
        header = QLabel("RL ADI // LIVE LOSS & VALUE")
        header.setFont(QFont("Consolas", 8, QFont.Bold))
        header.setStyleSheet("color: #00E5FF; letter-spacing: 0.5px;")
        header.setMinimumWidth(0)
        layout.addWidget(header)

        # Controls Row
        ctrl_layout = QHBoxLayout()
        
        self.btn_toggle_train = QPushButton("START RL TRAINING")
        self.btn_toggle_train.setFont(QFont("Consolas", 8, QFont.Bold))
        self.btn_toggle_train.setStyleSheet("""
            QPushButton {
                background-color: #00E676;
                color: #051A10;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #69F0AE;
            }
        """)
        self.btn_toggle_train.clicked.connect(self._toggle_training)
        ctrl_layout.addWidget(self.btn_toggle_train)

        self.lbl_iter_stats = QLabel("ITERS: 0 | LOSS: --")
        self.lbl_iter_stats.setFont(QFont("Consolas", 7))
        self.lbl_iter_stats.setStyleSheet("color: #FFD700; margin-left: 4px;")
        ctrl_layout.addWidget(self.lbl_iter_stats)
        ctrl_layout.addStretch()

        layout.addLayout(ctrl_layout)

    def _toggle_training(self):
        if not self.is_training_active:
            self.is_training_active = True
            self.btn_toggle_train.setText("PAUSE TRAINING")
            self.btn_toggle_train.setStyleSheet("""
                QPushButton {
                    background-color: #FF1744;
                    color: #FFFFFF;
                    border-radius: 4px;
                    padding: 6px 14px;
                }
            """)
            threading.Thread(target=self._training_loop, daemon=True).start()
        else:
            self.is_training_active = False
            self.btn_toggle_train.setText("RESUME TRAINING")
            self.btn_toggle_train.setStyleSheet("""
                QPushButton {
                    background-color: #00E676;
                    color: #051A10;
                    border-radius: 4px;
                    padding: 6px 14px;
                }
            """)

    def _training_loop(self):
        # Initialize trainer lazily on background thread
        if self.trainer is None:
            self.trainer = AutodidacticTrainer(self.model, lr=1e-3)

        iteration = 0
        while self.is_training_active:
            iteration += 1
            loss = self.trainer.train_step(batch_size=32, max_scramble_depth=10)
            if self.metrics_updated:
                self.metrics_updated.emit({
                    "iteration": iteration,
                    "loss": loss
                })
            time.sleep(0.05)

    @Slot(dict)
    def _on_metrics_received(self, data: dict):
        iteration = data.get("iteration", 0)
        loss = data.get("loss", 0.0)
        self.loss_history.append(loss)
        if len(self.loss_history) > 60:
            self.loss_history.pop(0)

        self.lbl_iter_stats.setText(f"ITERS: {iteration} | LOSS: {loss:.4f}")
        self.update()

    def paintEvent(self, event):
        if not PYSIDE_AVAILABLE:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()
        chart_top = 80
        chart_h = max(50, h - chart_top - 15)
        chart_w = w - 24
        chart_left = 12

        # Background grid
        painter.fillRect(chart_left, chart_top, chart_w, chart_h, QColor(11, 15, 25))
        painter.setPen(QPen(QColor(26, 38, 61), 1))
        painter.drawRect(chart_left, chart_top, chart_w, chart_h)

        if len(self.loss_history) < 2:
            painter.setPen(QColor(138, 155, 184))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(chart_left + 20, chart_top + chart_h // 2, "Awaiting training iterations...")
            return

        # Plot loss curve
        max_loss = max(self.loss_history) if self.loss_history else 1.0
        min_loss = min(self.loss_history) if self.loss_history else 0.0
        loss_range = max(1e-5, max_loss - min_loss)

        points = []
        dx = chart_w / max(1, len(self.loss_history) - 1)
        for i, val in enumerate(self.loss_history):
            norm_y = (val - min_loss) / loss_range
            px = chart_left + i * dx
            py = chart_top + chart_h - (norm_y * (chart_h - 10) + 5)
            points.append(QPointF(px, py))

        # Fill under curve
        fill_poly = [QPointF(chart_left, chart_top + chart_h)] + points + [QPointF(points[-1].x(), chart_top + chart_h)]
        painter.setBrush(QBrush(QColor(0, 229, 255, 40)))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF(fill_poly))

        # Draw line
        painter.setPen(QPen(QColor(0, 229, 255), 2))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])
