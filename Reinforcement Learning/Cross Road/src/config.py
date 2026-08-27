"""
Global configuration constants for the Autonomous Crossroad Simulation.
Includes screen dimensions, road geometry, physics parameters, RL settings, and color palettes.
"""
import math

# --- Screen & Display Settings ---
SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 800
FPS = 60
SIM_NAME = "DeepRL Autonomous Crossroad Simulation"

# --- World & Road Geometry ---
# Center of the intersection
CENTER_X = SCREEN_WIDTH // 2 - 120  # offset slightly to leave room for HUD on the right
CENTER_Y = SCREEN_HEIGHT // 2

ROAD_WIDTH = 140       # Total width of one 2-way road (2 lanes in, 2 lanes out)
LANE_WIDTH = ROAD_WIDTH / 4  # 35 px per lane
NUM_LANES_PER_DIR = 2  # 2 incoming lanes, 2 outgoing lanes

# Stop line distance from center
STOP_LINE_OFFSET = ROAD_WIDTH / 2 + 10 # Distance from center_x/y to stop line

# --- Traffic Lights Configuration ---
TRAFFIC_LIGHT_PHASES = {
    'NS_GREEN': 0,     # North & South Green, East & West Red
    'NS_YELLOW': 1,    # North & South Yellow, East & West Red
    'ALL_RED_1': 2,    # Clearance all-red buffer
    'EW_GREEN': 3,     # East & West Green, North & South Red
    'EW_YELLOW': 4,    # East & West Yellow, North & South Red
    'ALL_RED_2': 5,    # Clearance all-red buffer
}

PHASE_DURATIONS = {
    'NS_GREEN': 10.0,   # Seconds
    'NS_YELLOW': 2.5,
    'ALL_RED_1': 1.0,
    'EW_GREEN': 10.0,
    'EW_YELLOW': 2.5,
    'ALL_RED_2': 1.0,
}

# --- Vehicle Physical & Kinematics Settings ---
VEHICLE_LENGTH = 38.0  # pixels (~4.5 meters scaled)
VEHICLE_WIDTH = 18.0   # pixels (~2.0 meters scaled)
MAX_SPEED = 5.0        # pixels per frame (~50 km/h)
MIN_SPEED = 0.0        # No reversing allowed in lanes
MAX_ACCEL = 0.15       # Normal throttle acceleration
MAX_BRAKE = 0.35       # Normal braking deceleration
EMERGENCY_BRAKE = 0.70 # Emergency hard brake
FRICTION = 0.02        # Natural roll drag

# Turn Speeds
MAX_TURN_SPEED = 3.2

# Spawning parameters
SPAWN_INTERVAL_MIN = 1.2  # Seconds between cars in a lane
SPAWN_INTERVAL_MAX = 3.5
MAX_ACTIVE_CARS = 26

# --- Perception & Sensor Settings ---
LIDAR_NUM_RAYS = 9
LIDAR_MAX_DIST = 160.0  # Max vision distance
LIDAR_FOV = math.radians(70.0) # Field of view in front of vehicle

VISION_STATE_SIZE = (
    LIDAR_NUM_RAYS * 2 +  # Distance + Relative velocity for each ray (18)
    4 +                   # Traffic light state (Red, Yellow, Green, None) (one-hot) (4)
    1 +                   # Distance to stop line (normalized) (1)
    1 +                   # Vehicle current speed (normalized) (1)
    1 +                   # Target speed (normalized) (1)
    1 +                   # Distance to leading car on route (1)
    2 +                   # Intersection conflict zone clearance radar (2)
    1                     # Road friction/wetness (1)
) # Total: 29 features

NUM_ACTIONS = 5
ACTIONS_MAP = {
    0: "COAST",         # Maintain speed with slight drag
    1: "ACCEL_MILD",    # Mild throttle (+0.08)
    2: "ACCEL_FULL",    # Full throttle (+0.15)
    3: "BRAKE_MILD",    # Mild brake (-0.18)
    4: "BRAKE_HARD"     # Emergency hard brake (-0.45)
}

# --- Decision Frequency & Action Repeat ---
ACTION_REPEAT = 4  # Agent decides every 4 frames (15 Hz)

# --- Deep RL Hyperparameters ---
RL_GAMMA = 0.99
RL_LR = 0.00025
RL_BATCH_SIZE = 256
RL_BUFFER_CAPACITY = 100000
RL_TARGET_UPDATE_FREQ = 250
RL_EPSILON_START = 1.0
RL_EPSILON_MIN = 0.02
RL_EPSILON_DECAY = 0.99985

# --- Reward Weights ---
REWARD_CRASH = -50.0
REWARD_RED_LIGHT_RUN = -30.0
REWARD_SMOOTH_STOP_RED = +25.0
REWARD_PASS_EVENT = +80.0
REWARD_PROGRESS = +0.5
REWARD_TIME_PENALTY = -0.04
REWARD_JERK_PENALTY = -0.02

# --- Color Palette & Visuals ---
COLOR_BG = (28, 36, 43)              # Dark slate ambient
COLOR_GRASS_DAY = (72, 128, 72)      # Vibrant green grass
COLOR_GRASS_NIGHT = (24, 46, 28)     # Deep dark night grass
COLOR_ROAD_DAY = (50, 54, 60)        # Fresh dark asphalt
COLOR_ROAD_NIGHT = (22, 24, 28)
COLOR_ROAD_MARKING = (235, 238, 242) # Bright white lines
COLOR_ROAD_YELLOW = (245, 190, 40)   # Yellow divider lines
COLOR_SIDEWALK_DAY = (145, 150, 158)
COLOR_SIDEWALK_NIGHT = (45, 48, 52)
COLOR_STOP_LINE = (255, 255, 255)

# Traffic Light Colors
COLOR_TL_RED = (255, 45, 45)
COLOR_TL_RED_GLOW = (255, 70, 70, 80)
COLOR_TL_YELLOW = (255, 200, 30)
COLOR_TL_YELLOW_GLOW = (255, 210, 50, 80)
COLOR_TL_GREEN = (40, 235, 100)
COLOR_TL_GREEN_GLOW = (60, 255, 120, 80)
COLOR_TL_HOUSING = (25, 25, 25)

# Vehicle Visual Colors
CAR_COLORS = [
    (220, 48, 48),    # Crimson Red
    (38, 128, 235),   # Royal Blue
    (240, 240, 245),  # Pearl White
    (35, 35, 40),     # Obsidian Black
    (245, 180, 25),   # Amber Taxi Yellow
    (50, 185, 120),   # Emerald Green
    (155, 80, 220),   # Deep Purple
    (240, 100, 30),   # Sunset Orange
    (140, 150, 160),  # Metallic Silver
]

# UI / Cyber Theme Colors
UI_PANEL_BG = (18, 22, 30, 230)
UI_PANEL_BORDER = (45, 60, 85)
UI_ACCENT_CYAN = (0, 215, 255)
UI_ACCENT_GREEN = (46, 230, 138)
UI_ACCENT_ORANGE = (255, 155, 40)
UI_ACCENT_RED = (255, 65, 85)
UI_TEXT_WHITE = (240, 245, 250)
UI_TEXT_MUTED = (140, 155, 175)
