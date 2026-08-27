<div align="center">
  <img src="assets/banner_animated.gif" alt="Cross Road Autonomous AI Banner" width="100%">
  
  <h1>🚦 Cross Road: Autonomous AI Simulation</h1>
  <p><b>Advanced Deep Reinforcement Learning for Autonomous Traffic Management & Navigation</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
  [![Pygame](https://img.shields.io/badge/Pygame-Simulation-yellow.svg)](https://www.pygame.org/)
</div>

---

## 🌟 About the Project

This project is an advanced **Deep Reinforcement Learning (Deep RL)** simulator that models the behavior of autonomous vehicles at a busy intersection using the **Dueling Double DQN** architecture.

In this environment, autonomous agents learn how to navigate through a heavily congested crossroad safely and efficiently without any collisions. The agents perceive weather conditions, traffic lights, pedestrians, and other vehicles through their simulated LiDAR and radar sensors to make real-time intelligent driving decisions.

---

## 🚀 Key Features

- 🧠 **Advanced Deep RL:** Utilizes `Prioritized Experience Replay (PER)` and a `Dueling DQN` neural network architecture for faster and more stable learning.
- 🌦️ **Dynamic Weather System:** Rain, snow, and varying road grip (friction) are fed into the neural network, teaching the agent to brake earlier on slippery roads.
- 🚶 **Pedestrians & Crosswalks:** The AI learns to yield to pedestrians crossing the street on zebra crossings.
- 🚑 **Smart Emergency Vehicles:** Ambulances can actuate and override traffic signals (Emergency Preemption) to clear the intersection.
- 🌙 **Day & Night Rendering Engine:** Features a dynamic 2D lighting engine with realistic headlights, bloom, and shadows.
- 📊 **Cyberpunk Live Telemetry HUD:** Displays real-time neural network layer activations, Q-Values, decision probabilities, and success metrics.

---

## 📁 Project Structure & Architecture

```text
📁 Cross Road/
├── 📄 run.bat                     # Quick launch script for GUI simulation
├── 📄 train.bat                   # Quick launch script for fast headless training
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # The file you are currently reading!
├── 📁 assets/                     # Media and images
│   └── 🖼️ banner.png
├── 📁 src/                        # Main source code
│   ├── 📄 config.py               # Global configuration and RL hyperparameters
│   ├── 📄 main.py                 # Core simulation loop and Pygame rendering
│   ├── 📄 train_headless.py       # Background training loop (No GUI, maximum FPS)
│   ├── 📁 ai/                     # Artificial Intelligence modules
│   │   ├── 📄 dqn_agent.py        # DQN Agent, Memory Buffer, and RL logic
│   │   ├── 📄 network.py          # PyTorch Neural Network architecture
│   │   └── 📁 weights/            # Pre-trained model weights (.pt files)
│   ├── 📁 simulation/             # Physics and world simulation logic
│   │   ├── 📄 vehicle.py          # Vehicle physics, kinematics, and OBB collisions
│   │   ├── 📄 sensors.py          # Raycasting sensors, LiDAR, and TTC calculations
│   │   ├── 📄 intersection.py     # Map geometry and routing
│   │   ├── 📄 pedestrians.py      # Pedestrian behavior and spawning
│   │   ├── 📄 weather.py          # Weather engine and road friction
│   │   ├── 📄 traffic_controller.py # Smart and adaptive traffic light controller
│   │   └── 📄 particles.py        # Particle systems (smoke, sparks, skid marks)
│   └── 📁 render/                 # Graphics and UI Engine
│       ├── 📄 renderer.py         # Drawing the environment and entities
│       ├── 📄 lighting.py         # Night lighting and shadow casting
│       └── 📄 ui_hud.py           # Cyberpunk telemetry dashboard and NN Visualizer
└── 📁 tests/                      # Unit Tests
    └── 📄 test_collision.py       # Collision detection tests
```

---

## 🎮 Installation & Usage

### 1. Install Dependencies
Ensure you have Python 3 (preferably 3.10+) installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Simulation (GUI Mode)
To watch the AI drive in the fully rendered environment, run `run.bat` or use the terminal:
```bash
python src/main.py
```

### 3. Train the AI (Headless Mode)
To train the AI in the background without wasting resources on rendering graphics, run `train.bat` or use the terminal:
```bash
python src/train_headless.py
```

---

## ⌨️ Keybindings (GUI Mode)

- `Space`: Pause / Resume simulation
- `W`: Toggle Weather (Clear -> Rain -> Storm)
- `N`: Toggle Day / Night mode
- `T`: Manually switch traffic lights
- `A`: Spawn Ambulance
- `V`: Toggle AI Vision Rays (LiDAR view)
- `1` / `2` / `3`: Switch AI Mode between: Untrained Chaos / Live Training / Master AI

<br>
<div align="center">
  <i>Made with ❤️ for AI Enthusiasts</i>
</div>
