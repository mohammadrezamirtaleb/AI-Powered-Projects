# 🚦 Autonomous Crossroad Deep Reinforcement Learning Simulation

![Cross Road Simulation](assets/banner_animated.gif)

An advanced, 60 FPS, realistic 4-way intersection simulation featuring **Deep Reinforcement Learning (Dueling DQN with PyTorch and Prioritized Experience Replay)**, **Perception and LiDAR sensors (LiDAR Raycasting & Time-to-Collision)**, a **Dynamic Weather Engine (Rain, Storm, Wet Asphalt Physics)**, **Dynamic Day/Night Cycles with Conical Headlights**, **7 Diverse Vehicle Classes (including an Ambulance with Preemption)**, **Pedestrian Crossings**, Adaptive Traffic Lights, Realistic Collision Physics, Spark/Smoke/Fire Particles, and Skid Marks.

---

## 🌟 Key Highlights

1. **Deep Reinforcement Learning (Dueling DQN & PER):**
   - **Untrained Chaos Mode:** Cars run red lights, speed recklessly, and cause pile-ups.
   - **Live RL Training Mode:** Cars learn in real-time while the simulation runs, using a Prioritized Experience Replay (PER) buffer, 15 Hz decision rate (Action Repeat = 4), learning to stop at red lights and maintain safe following distances.
   - **Master AI Mode:** Fully trained, highly capable AI models driving safely and smoothly through the intersection.

2. **Machine Perception & Vision (LiDAR & TTC Sensors):**
   - 9-Ray LiDAR sensor detecting the distance and relative speed of vehicles and pedestrians.
   - **Time-to-Collision (TTC)** metric predicting time remaining until a potential impact.
   - Vision sensors to detect traffic light states (Red, Yellow, Green) and distance to the stop line.
   - Radar for detecting traffic congestion and interference inside the intersection.

3. **Dynamic Weather & Wet Road Physics:**
   - 3-Phase Cycle: ☀️ **Clear**, 🌧️ **Rain**, and ⛈️ **Storm**.
   - Wind-affected suspended rain particles and ground splash ripples on the asphalt.
   - Physical reduction of tire friction coefficients in rain ($\mu = 0.52 - 0.68$) resulting in increased stopping distances.

4. **Pedestrian Crossings & 7 Vehicle Classes:**
   - Animated pedestrians crossing the street during red traffic phases with swinging arm animations.
   - **7 Vehicle Classes:** Sedan, SUV, Heavy Truck, City Bus, Sports Car, Motorcycle, and Emergency Ambulance.
   - Ambulance equipped with flashing red/blue lights and an Emergency Preemption system.

5. **Smart & Adaptive Traffic Lights (Actuated / Adaptive Signals):**
   - Automatically adjusts green phase duration based on queue length and vehicle density in each lane to dissolve traffic jams.

6. **2D Lighting Engine & Day/Night Cycle:**
   - Pre-cached background rendering for a smooth 60+ FPS experience.
   - Smooth transitions between day, sunset, and midnight.
   - Realistic conical headlights cutting through the darkness.
   - Neon bloom glow for traffic lights reflecting on the asphalt.
   - Glowing red brake lights and blinking orange turn signals.
   - Physical impulse spin-outs during collisions accompanied by sparks, smoke, and fire particles.

7. **Cyber HUD & Live Neural Visualizer:**
   - Click on any vehicle to inspect its AI brain in real-time.
   - Live visualization of neural network layers and active neurons (Input -> Dense 1 -> Dense 2 -> Q-Outputs).
   - Real-time line chart tracking the Success Rate % Trend.
   - Bar chart displaying Q-Values and the vehicle's current decision probabilities.
   - Live telemetry stats: Success Rate, Total Crashes, FPS, Exploration Rate ($\epsilon$), Weather State, and Friction Coefficient ($\mu$).

---

## 🎮 Controls & Keybindings

| Key | Action |
| :--- | :--- |
| **`Space`** | Pause / Resume Simulation |
| **`1`** | Activate Untrained Chaos Mode |
| **`2`** | Activate Live Training Mode |
| **`3`** | Activate Master AI Mode |
| **`W`** | Toggle Weather (☀️ Clear $\leftrightarrow$ 🌧️ Rain $\leftrightarrow$ ⛈️ Storm) |
| **`N`** | Day / Night Toggle |
| **`A`** | Spawn Emergency Ambulance (with Preemption) |
| **`S`** | Spawn a Random Vehicle |
| **`T`** | Manually Switch Traffic Light Phase |
| **`V`** | Toggle Vision Rays (LiDAR/Sensors visibility) |
| **`R`** | Reset Stats and Crashes |
| **`Click on Car`** | Select and track the vehicle's brain and sensors in the HUD |

---

## 🚀 How to Run

### 1-Click Fast Method (Recommended on Windows):
Simply double-click the **`run.bat`** file.

### Manual Terminal Execution:
```powershell
# Run the graphical simulation
.\venv\Scripts\python.exe src\main.py
```

### Fast Headless Training:
Double-click **`train.bat`** or run the following command in your terminal:
```powershell
.\venv\Scripts\python.exe src\train_headless.py
```

---

## 📁 Project Structure
```
c:\Users\Apple\Desktop\Cross Road/
├── venv/                       # Python Virtual Environment
├── requirements.txt            # Dependencies (PyTorch, Pygame, NumPy, ...)
├── run.bat                     # Easy runner for the GUI simulation
├── train.bat                   # Easy runner for headless training
├── README.md                   # Project Documentation
└── src/
    ├── config.py               # Physics, colors, sensors, and RL hyperparameters
    ├── main.py                 # Main simulation loop and event handler
    ├── train_headless.py       # Fast headless training script using PER & DQfD
    ├── simulation/
    │   ├── intersection.py     # 4-way geometry, lanes, bezier curves
    │   ├── vehicle.py          # Kinematics, 7 classes, crash spin-outs, SAT OBB collisions
    │   ├── pedestrians.py      # Pedestrians and crosswalk logic
    │   ├── weather.py          # Dynamic weather engine, rain, ripples, wet asphalt
    │   ├── sensors.py          # LiDAR, radar, vision perception, TTC calculation
    │   ├── traffic_controller.py # Automatic phasing, adaptive logic, ambulance preemption
    │   └── particles.py        # Sparks, smoke, fire, and skid marks system
    ├── ai/
    │   ├── network.py          # Dueling DQN architecture (PyTorch)
    │   ├── dqn_agent.py        # PER buffer, action selection, reward calculation, optimizer
    │   └── weights/            # Pre-trained model weights (pretrained_master.pt)
    └── render/
        ├── renderer.py         # Background caching, asphalt textures, markings, lights
        ├── lighting.py         # Day/Night lighting engine, headlight cones
        └── ui_hud.py           # Telemetry dashboard, live charts, AI brain visualizer
```
