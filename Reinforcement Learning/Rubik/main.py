"""
================================================================================
  ASTRA-DEEPCUBE // Autonomous Neural RL & Spatial Vision Rubik's Cube Engine
================================================================================
A Native Desktop AI System powered by:
 - DeepCubeA (Autodidactic Iteration + Neural Weighted A* Search)
 - Project Astra "Fly-Eye" Multi-Perspective Spatial Vision & Webcam AR
 - Hardware-Accelerated 3D Holographic Viewport & Cyberpunk Glassmorphism UI
 - Multi-Dimensional NxNxN MegaCube Reduction Solver (3x3, 4x4, 5x5, 7x7)
 - 1-Click Viral LinkedIn Showcase Media Exporter
================================================================================
"""

import sys
import os

# Ensure local packages are on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cube_state import CubeState
from rl_engine.deepcube_model import TORCH_AVAILABLE


def main():
    print("=" * 75)
    print("  [INITIALIZING] ASTRA-DEEPCUBE NEURAL DESKTOP ENGINE...")
    print("=" * 75)
    print(f"  * PyTorch Neural Engine: {'ACTIVE (GPU/CPU Acceleration)' if TORCH_AVAILABLE else 'NUMPY FALLBACK'}")
    print("  * Project Astra Spatial Vision: 6-Drone Multi-Angle Array + Webcam AR")
    print("  * 3D Hardware Renderer: 60 FPS Cybernetic Viewport")
    print("  * Solvers: DeepCube-A Neural A*, Kociemba 2-Phase, NxNxN Reduction")
    print("=" * 75)

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from ui.main_window import AstraDeepCubeWindow

        # Enable high-DPI scaling for ultra-crisp display on modern monitors
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        window = AstraDeepCubeWindow(initial_size=3)
        window.showMaximized()

        sys.exit(app.exec())

    except ImportError as e:
        print(f"[ERROR] PySide6 or graphics dependency missing: {e}")
        print("Please ensure PySide6 is installed: pip install PySide6")
        sys.exit(1)


if __name__ == "__main__":
    main()
