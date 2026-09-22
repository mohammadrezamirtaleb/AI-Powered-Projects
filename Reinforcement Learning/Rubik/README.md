# 🚀 ASTRA-DEEPCUBE
### Autonomous Neural Reinforcement Learning & Spatial Vision Rubik's Cube Engine (Native Desktop)

---

```
   █████╗ ███████╗████████╗██████╗  █████╗     ██████╗ ███████╗███████╗██████╗  ██████╗██╗   ██╗██████╗ ███████╗
  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗    ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██║   ██║██╔══██╗██╔════╝
  ███████║███████╗   ██║   ██████╔╝███████║    ██║  ██║█████╗  █████╗  ██████╔╝██║     ██║   ██║██████╔╝█████╗  
  ██╔══██║╚════██║   ██║   ██╔══██╗██╔══██║    ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║     ██║   ██║██╔══██╗██╔══╝  
  ██║  ██║███████║   ██║   ██║  ██║██║  ██║    ██████╔╝███████╗███████╗██║     ╚██████╗╚██████╔╝██████╔╝███████╗
  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚══════╝╚═╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

An ultra-futuristic, high-performance native desktop AI application combining **Deep Reinforcement Learning (DeepCubeA / ADI)**, **Project Astra "Fly-Eye" Spatial Computer Vision**, and **Hardware-Accelerated 3D Holographic Rendering** for solving Rubik's Cubes (from standard 3x3 up to NxNxN mega-cubes).

---

## 🌟 Key Architecture & Highlights

### 1. 🧠 Deep Reinforcement Learning (DeepCubeA + ADI)
- **Neural Architecture**: Deep Residual MLP with Layer Normalization, ELU activations, dual Value Head $v(s)$ (cost-to-go) and Policy Head $p(a|s)$.
- **Autodidactic Iteration (ADI)**: Self-supervised Bellman updates generated from backward random walks from the solved state.
- **Weighted Neural $A^*$ Search**: Real-time graph search $f(s) = g(s) + \lambda \cdot h_\theta(s)$ with live telemetry of expanded nodes, depth, and action probability radars.
- **Live Training Dashboard**: Real-time loss plotting and on-the-fly model fine-tuning directly in the desktop app.

### 2. 👁️ Project Astra "Fly-Eye" Spatial Vision & Webcam AR
- **6-Drone Multi-Angle Virtual Camera Matrix**: 6 autonomous virtual drone viewpoints orbiting the cube (Top, Bottom, Front, Back, Left, Right) with confidence heatmaps, edge detection, and laser scan lines.
- **Physical Webcam Scanner**: Real-time OpenCV color segmentation (HSV/CIELAB) and 3D state reconstruction from your physical Rubik's cube.
- **Augmented Reality (AR) Overlay**: Dynamic 3D move indicator arrows and target rotation cues projected on the video feed.

### 3. 🎮 High-Performance 3D Holographic Viewport (Native Desktop)
- **100% Native Desktop Window (PySide6)**: Zero browser dependency, runs directly on Windows with GPU acceleration.
- **Physics-Based Smooth Rotations**: 60 FPS face slice animations with ease-out cubic interpolation.
- **Holographic Exploded View**: Smooth slider to inspect internal mechanics by exploding cubies outwards.
- **Multi-Dimensional Support**: Seamless switching between 3x3x3, 4x4x4 Master, 5x5x5 Professor, and 7x7x7 MegaCube!

### 4. 🚀 1-Click LinkedIn Showcase Exporter
- Automatically generates high-resolution social preview cards and animated solution benchmarks ready to share on LinkedIn, GitHub, and Twitter.

---

## 📂 Project Structure

```
c:\Users\Apple\Desktop\Rubik/
├── core/
│   ├── cube_state.py          # NxNxN Rubik's cube state & physics model
│   ├── scrambler.py           # Anti-cancellation scramble generator
│   ├── kociemba_solver.py     # 2-Phase optimal / admissible IDA* solver
│   └── large_cube_solver.py   # NxNxN Reduction solver (4x4, 5x5, 7x7)
├── rl_engine/
│   ├── deepcube_model.py      # PyTorch Deep Residual Value/Policy Network
│   ├── autodidactic_iteration.py # ADI self-supervised Bellman trainer
│   ├── neural_astar_search.py # Weighted Neural A* heuristic search engine
│   └── pretrained_weights.py  # Model checkpoint cache & rapid warm-up
├── vision_engine/
│   ├── cube_vision_tracker.py # Real-time OpenCV webcam scanner & color calibration
│   ├── virtual_multicam_sensor.py # 6-Drone Project Astra Fly-Eye holographic matrix
│   └── ar_overlay.py          # Augmented Reality solver overlay
├── renderer_3d/
│   └── cube_renderer.py       # Hardware-accelerated 3D viewport & particle engine
├── ui/
│   ├── main_window.py         # Sci-Fi Cyberpunk Native Glassmorphism HUD Window
│   └── components/
│       ├── astra_stream.py    # Spatial reasoning stream-of-consciousness log
│       ├── neural_graph_view.py # Live A* Search tree & Q-value radar widget
│       ├── training_widget.py # Live RL training & Loss curve dashboard
│       ├── vision_hud.py      # Multi-camera drone matrix & webcam AR HUD
│       └── control_deck.py    # Sci-Fi interactive button decks & sliders
├── audio/
│   └── sound_synthesizer.py   # Sci-Fi synthesized audio & voice commentary
├── exporter/
│   └── media_exporter.py      # 1-Click LinkedIn Showcase generator
├── tests/
│   └── test_cube_engine.py    # Comprehensive automated test suite
└── main.py                    # Main Desktop Application Entry Point
```

---

## 🚀 How to Run

### 1. Launch the Desktop Application
```bash
python main.py
```

### 2. Run the Verification Tests
```bash
python -m unittest tests/test_cube_engine.py
```

---

## 💡 How to Use for a Viral LinkedIn Post

1. **Launch the App**: Run `python main.py`.
2. **Hit "SCRAMBLE"**: Observe the 3D cube scramble with synthesized audio whooshes.
3. **Click "DEEPCUBE-A RL SOLVE"**:
   - Watch the live **Neural Search Telemetry** update in real-time (nodes expanded, depth $g(s)$, heuristic cost $h(s)$, and $Q(s, a)$ action probability radar).
   - Watch the **Project Astra Neural Reasoning Stream** analyze Cayley graph orbits and group theory symmetries.
   - Watch the **Project Astra Fly-Eye 6-Drone Matrix** track the faces simultaneously.
4. **Slide "EXPLODED VIEW"**: Show the exploded 3D CAD holographic inspection view.
5. **Click "LINKEDIN SHOWCASE EXPORT"**: Instantly generates an image ready to post.
