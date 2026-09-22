"""
Astra-DeepCube: Project Astra Spatial Neural Stream Widget
Displays real-time stream-of-consciousness AI reasoning, group theory analysis, and spatial perception.
"""

from typing import Optional
import time
import random

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel, QHBoxLayout, QSizePolicy
    from PySide6.QtCore import Qt, QTimer, QSize
    from PySide6.QtGui import QFont, QColor, QTextOption
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


SAMPLE_THOUGHTS = [
    "Analyzing Cayley graph orbits across subgroup <U, D, L2, R2, F2, B2>...",
    "Computing Schreier-Sims stabilizer chain for 3x3 permutation group...",
    "Neural heuristic h(s) detected 4 misplaced corner orientations on Down face.",
    "Evaluating Bellman optimality operator: min_a (1 + V(s_a))...",
    "Spatial vision confidence at 98.6%. Facelet parity invariant validated.",
    "Applying commutators to reduce center entropy...",
    "Pruning suboptimal branches in priority queue (beam width = 5000)...",
    "God's algorithm vicinity reached: distance to target state <= 16 moves.",
    "Project Astra spatial alignment confirmed. Executing optimal rotation sequence."
]


class AstraStreamWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Project Astra Spatial Intelligence Thought Stream.
    """

    def __init__(self, parent=None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._init_ui()

    def minimumSizeHint(self):
        return QSize(160, 80) if PYSIDE_AVAILABLE else None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("PROJECT ASTRA // REASONING STREAM")
        title.setFont(QFont("Consolas", 8, QFont.Bold))
        title.setStyleSheet("color: #00E5FF; letter-spacing: 0.5px;")
        title.setMinimumWidth(0)
        
        self.live_indicator = QLabel("◈ LIVE")
        self.live_indicator.setFont(QFont("Consolas", 7, QFont.Bold))
        self.live_indicator.setStyleSheet("color: #00E676;")

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.live_indicator)
        layout.addLayout(header_layout)

        # Text stream
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Consolas", 8))
        self.text_area.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_area.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.text_area.setMinimumSize(80, 60)
        self.text_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0B0E17;
                color: #A0C0E0;
                border: 1px solid #1E2D4A;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.text_area)

        # Periodic background thoughts timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_idle_thought)
        self.timer.start(3500)

        self.log_thought("Astra Neural Spatial Perception Engine online. Awaiting cube input.")

    def log_thought(self, message: str, tag: str = "ASTRA"):
        """Appends a new thought to the stream."""
        if not PYSIDE_AVAILABLE:
            return
        timestamp = time.strftime("%H:%M:%S")
        colored_tag = f"<span style='color: #00E5FF;'>[{timestamp}]</span> <b style='color: #FFD700;'>[{tag}]</b>"
        formatted = f"{colored_tag} <span style='color: #E0E8F5;'>{message}</span>"
        self.text_area.append(formatted)
        
        # Auto scroll to bottom
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_idle_thought(self):
        """Generates realistic autonomous perception thoughts."""
        if random.random() < 0.4:
            thought = random.choice(SAMPLE_THOUGHTS)
            self.log_thought(thought, tag="SPATIAL-AI")
