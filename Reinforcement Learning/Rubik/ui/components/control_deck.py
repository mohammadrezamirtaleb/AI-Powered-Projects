"""
Astra-DeepCube: Sci-Fi Interactive Control Deck
Houses Scramble, Solve Modes (DeepCubeA RL / Kociemba / Reduction), Playback, and Manual Controls.
All action triggers are equipped with high-resolution vector cyber icons (Zero Emojis).
"""

from typing import Callable, Optional

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QSlider, QComboBox, QGroupBox, QGridLayout
    )
    from PySide6.QtCore import Qt, Signal, QSize
    from PySide6.QtGui import QFont, QColor, QIcon
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

from ui.components.cyber_icons import create_cyber_icon


class ControlDeckWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Main interactive control console with vector cyber icons.
    """
    scramble_requested = Signal(int) if PYSIDE_AVAILABLE else None
    solve_deepcube_requested = Signal() if PYSIDE_AVAILABLE else None
    solve_kociemba_requested = Signal() if PYSIDE_AVAILABLE else None
    manual_move_requested = Signal(str) if PYSIDE_AVAILABLE else None
    dimension_changed = Signal(int) if PYSIDE_AVAILABLE else None
    exploded_changed = Signal(float) if PYSIDE_AVAILABLE else None
    play_pause_toggled = Signal(bool) if PYSIDE_AVAILABLE else None
    step_forward_requested = Signal() if PYSIDE_AVAILABLE else None
    export_linkedin_requested = Signal() if PYSIDE_AVAILABLE else None

    def __init__(self, parent=None):
        if PYSIDE_AVAILABLE:
            super().__init__(parent)
            self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(5)

        # ---------------- Row 1: Dimensions & Core Actions ----------------
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # Dimension selector
        lbl_dim = QLabel("CUBE MATRIX:")
        lbl_dim.setFont(QFont("Consolas", 8, QFont.Bold))
        lbl_dim.setStyleSheet("color: #00E5FF;")
        row1.addWidget(lbl_dim)

        self.combo_dim = QComboBox()
        self.combo_dim.addItems(["3x3x3 Standard", "4x4x4 Master", "5x5x5 Professor", "7x7x7 MegaCube"])
        self.combo_dim.setStyleSheet("""
            QComboBox {
                background-color: #141B2D;
                color: #FFFFFF;
                border: 1px solid #1E2D4A;
                border-radius: 3px;
                padding: 3px 6px;
                font-family: Consolas;
                font-size: 11px;
            }
            QComboBox QAbstractItemView {
                background-color: #141B2D;
                color: #FFFFFF;
                selection-background-color: #00E5FF;
                selection-color: #000000;
            }
        """)
        self.combo_dim.currentIndexChanged.connect(self._on_dim_changed)
        row1.addWidget(self.combo_dim)

        # Scramble Depth & Button with vector icon
        self.btn_scramble = self._create_btn("SCRAMBLE", "#FF9100", "#1A1005", icon_type="scramble")
        self.btn_scramble.clicked.connect(lambda: self.scramble_requested.emit(20))
        row1.addWidget(self.btn_scramble)

        # Solvers with vector icons
        self.btn_solve_rl = self._create_btn("DEEPCUBE-A RL", "#00E5FF", "#041525", icon_type="neural")
        self.btn_solve_rl.clicked.connect(lambda: self.solve_deepcube_requested.emit())
        row1.addWidget(self.btn_solve_rl)

        self.btn_solve_kociemba = self._create_btn("KOCIEMBA OPTIMAL", "#00E676", "#042010", icon_type="optimal")
        self.btn_solve_kociemba.clicked.connect(lambda: self.solve_kociemba_requested.emit())
        row1.addWidget(self.btn_solve_kociemba)

        self.btn_export = self._create_btn("LINKEDIN EXPORT", "#E040FB", "#200425", icon_type="export")
        self.btn_export.clicked.connect(lambda: self.export_linkedin_requested.emit())
        row1.addWidget(self.btn_export)

        row1.addStretch()
        main_layout.addLayout(row1)

        # ---------------- Row 2: Sliders & Playback ----------------
        row2 = QHBoxLayout()
        row2.setSpacing(5)

        # Exploded View Slider
        lbl_explode = QLabel("EXPLODE:")
        lbl_explode.setFont(QFont("Consolas", 8, QFont.Bold))
        lbl_explode.setStyleSheet("color: #FFD700;")
        row2.addWidget(lbl_explode)

        self.slider_explode = QSlider(Qt.Horizontal)
        self.slider_explode.setRange(0, 100)
        self.slider_explode.setValue(0)
        self.slider_explode.setFixedWidth(80)
        self.slider_explode.valueChanged.connect(lambda v: self.exploded_changed.emit(v / 100.0))
        row2.addWidget(self.slider_explode)

        # Playback Controls with vector icons
        self.btn_play_pause = self._create_btn("PLAY", "#00E5FF", "#051525", compact=True, icon_type="play")
        self.is_playing = False
        self.btn_play_pause.clicked.connect(self._toggle_playback)
        row2.addWidget(self.btn_play_pause)

        self.btn_step = self._create_btn("STEP", "#00E676", "#052010", compact=True, icon_type="step")
        self.btn_step.clicked.connect(lambda: self.step_forward_requested.emit())
        row2.addWidget(self.btn_step)

        # Manual Move Buttons
        lbl_manual = QLabel("MANUAL:")
        lbl_manual.setFont(QFont("Consolas", 8, QFont.Bold))
        lbl_manual.setStyleSheet("color: #A0C0E0; margin-left: 6px;")
        row2.addWidget(lbl_manual)

        for m in ["U", "D", "L", "R", "F", "B"]:
            btn_m = self._create_btn(m, "#448AFF", "#051020", compact=True)
            btn_m.clicked.connect(lambda _, move=m: self.manual_move_requested.emit(move))
            row2.addWidget(btn_m)

        for m in ["U'", "D'", "L'", "R'", "F'", "B'"]:
            btn_m = self._create_btn(m, "#2979FF", "#051020", compact=True)
            btn_m.clicked.connect(lambda _, move=m: self.manual_move_requested.emit(move))
            row2.addWidget(btn_m)

        row2.addStretch()
        main_layout.addLayout(row2)

    def _create_btn(
        self,
        text: str,
        bg_color: str,
        text_color: str,
        compact: bool = False,
        icon_type: Optional[str] = None
    ) -> QPushButton:
        btn = QPushButton(f" {text}" if icon_type else text)
        padding = "3px 6px" if compact else "4px 10px"
        btn.setFont(QFont("Consolas", 8, QFont.Bold))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border-radius: 3px;
                padding: {padding};
                text-align: center;
            }}
            QPushButton:hover {{
                opacity: 0.88;
                border: 1px solid #FFFFFF;
            }}
        """)
        if icon_type and PYSIDE_AVAILABLE:
            icon = create_cyber_icon(icon_type, color_hex=text_color, size=15)
            if icon:
                btn.setIcon(icon)
                btn.setIconSize(QSize(14, 14))
        return btn

    def _on_dim_changed(self, idx: int):
        sizes = [3, 4, 5, 7]
        chosen = sizes[idx] if idx < len(sizes) else 3
        self.dimension_changed.emit(chosen)

    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        self._update_play_button_state()
        if self.play_pause_toggled:
            self.play_pause_toggled.emit(self.is_playing)

    def set_playing(self, playing: bool):
        """Programmatically update playing state without triggering signals."""
        self.is_playing = playing
        if PYSIDE_AVAILABLE:
            self._update_play_button_state()

    def _update_play_button_state(self):
        if not PYSIDE_AVAILABLE:
            return
        if self.is_playing:
            self.btn_play_pause.setText(" PAUSE")
            icon = create_cyber_icon("pause", color_hex="#051525", size=18)
            if icon:
                self.btn_play_pause.setIcon(icon)
        else:
            self.btn_play_pause.setText(" PLAY")
            icon = create_cyber_icon("play", color_hex="#051525", size=18)
            if icon:
                self.btn_play_pause.setIcon(icon)
