"""
Global configuration constants for the Autonomous City Crossroad Simulation.
Includes screen dimensions, multi-node road geometry, physics, RL, and palettes.
"""
import math

# --- Screen & Display Settings ---
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60
SIM_NAME = "DeepRL Autonomous City Network"

# --- World & Road Geometry (3 lanes per direction) ---
CENTER_X = SCREEN_WIDTH // 2 - 120
CENTER_Y = SCREEN_HEIGHT // 2

NUM_LANES_PER_DIR = 3
LANE_WIDTH = 26.0
ROAD_WIDTH = LANE_WIDTH * NUM_LANES_PER_DIR * 2  # 156 px two-way arterial

ROUNDABOUT_ISLAND_R = 30.0
ROUNDABOUT_INNER_R = 48.0   # inner circulating centerline
ROUNDABOUT_CIRC_R = 74.0    # outer circulating centerline
ROUNDABOUT_OUTER_R = 102.0  # curb

STOP_LINE_OFFSET = ROAD_WIDTH / 2 + 10

# --- Traffic Lights Configuration ---
TRAFFIC_LIGHT_PHASES = {
    'NS_GREEN': 0,
    'NS_YELLOW': 1,
    'ALL_RED_1': 2,
    'EW_GREEN': 3,
    'EW_YELLOW': 4,
    'ALL_RED_2': 5,
}

PHASE_DURATIONS = {
    'NS_GREEN': 10.0,
    'NS_YELLOW': 2.5,
    'ALL_RED_1': 1.8,
    'EW_GREEN': 10.0,
    'EW_YELLOW': 2.5,
    'ALL_RED_2': 1.8,
}

# --- Vehicle Physical & Kinematics Settings ---
VEHICLE_LENGTH = 38.0
VEHICLE_WIDTH = 18.0
MAX_SPEED = 5.0
MIN_SPEED = 0.0
MAX_ACCEL = 0.15
MAX_BRAKE = 0.35
EMERGENCY_BRAKE = 0.70
FRICTION = 0.02
MAX_TURN_SPEED = 3.2

SIREN_RADIUS = 160.0
EMERGENCY_PREEMPT_DIST = 240.0

# Roundabout give-way rule. Only traffic already on the ring and within this arc
# upstream of our entry has priority; anything further round will not reach the
# entry before we clear it. Yielding to the whole ring deadlocks every approach.
RING_CONFLICT_ARC = math.radians(105.0)
# A vehicle standing still for longer than this is an obstacle, not circulating
# traffic. Without the distinction one stalled car freezes every approach forever.
RING_STALLED_IGNORE_SEC = 1.5

SPAWN_INTERVAL_MIN = 1.8
SPAWN_INTERVAL_MAX = 4.2
MAX_ACTIVE_CARS = 28

# --- Perception & Sensor Settings ---
LIDAR_NUM_RAYS = 9
LIDAR_MAX_DIST = 180.0
LIDAR_FOV = math.radians(70.0)

LIDAR_FEATURE_SIZE = LIDAR_NUM_RAYS * 2  # range + relative velocity
SITUATION_FEATURE_SIZE = 14              # semantic multi-context slice

VISION_STATE_SIZE = (
    LIDAR_FEATURE_SIZE +  # Distance + Relative velocity (18)
    5 +                   # Traffic light one-hot: RED/YELLOW/GREEN/YIELD/NONE (5)
    1 +                   # Distance to stop line (1)
    1 +                   # Vehicle speed (1)
    1 +                   # Target speed (1)
    1 +                   # Distance to leading car (1)
    2 +                   # Conflict zone radar (2)
    1 +                   # Road friction (1)
    1 +                   # In roundabout (1)
    1 +                   # Circulating density (1)
    1 +                   # Emergency proximity (1)
    1 +                   # Lane index normalized (1)
    1 +                   # Yield required (1)
    1 +                   # Normalized distance to roundabout yield line (1)
    1 +                   # Inside conflict box flag (1)
    1 +                   # Time-to-collision, normalized & inverted (1)
    SITUATION_FEATURE_SIZE
)  # Total: 52 features

FRAME_STACK_SIZE = 3
STACKED_STATE_SIZE = VISION_STATE_SIZE * FRAME_STACK_SIZE  # 156

NUM_ACTIONS = 5
ACTIONS_MAP = {
    0: "COAST",
    1: "ACCEL_MILD",
    2: "ACCEL_FULL",
    3: "BRAKE_MILD",
    4: "BRAKE_HARD"
}

ACTION_REPEAT = 4

# --- Deep RL Hyperparameters ---
RL_GAMMA = 0.997
RL_LR = 0.0005
RL_BATCH_SIZE = 256
RL_BUFFER_CAPACITY = 100000
RL_EPSILON_START = 1.0
RL_EPSILON_MIN = 0.08
RL_EPSILON_DECAY = 0.9998
RL_HIDDEN_DIM = 320
RL_TRUNK_DIM = 192      # width of the shared trunk feeding both dueling streams
RL_STREAM_DIM = 128     # width of the value / advantage heads

# --- Reward & Penalty Weights ---
# Two distinct families. DENSE terms are expressed PER SECOND and are multiplied
# by dt inside the reward function, so they no longer depend on the frame rate.
# EVENT terms fire once per vehicle and are already in final units.
#
# Scale check for a clean ~10s trip: progress +8, junction pass +12,
# route completion +25  =>  ~+45. An at-fault crash is -60 and forfeits the
# remaining bonuses, so crashing is always worse than driving slowly.

# Dense: continuous shaping, units are reward-per-second
REWARD_PROGRESS = +1.0          # at full target speed
REWARD_IDLE_RED = +0.15         # small credit for waiting properly at a red
PENALTY_TIME = -0.25            # cost of existing, discourages loitering
PENALTY_TAILGATE = -4.0         # headway below the safe following distance
PENALTY_TTC = -12.0             # time-to-collision below the safe threshold
PENALTY_SPEEDING = -3.0         # above the vehicle's own speed limit
PENALTY_BLOCK_BOX = -6.0        # stalled inside the junction / roundabout box
PENALTY_STALL = -2.0            # frozen on a green light with a clear road
PENALTY_PED_PROXIMITY = -8.0    # moving fast close to a pedestrian
PENALTY_RED_CREEP = -5.0        # commanding throttle while held at a red light

# Event: one-shot, fired at most once per vehicle
REWARD_PASS_EVENT = +12.0       # cleared the signalized junction legally
REWARD_ROUTE_COMPLETE = +25.0   # reached the end of the route alive
REWARD_SMOOTH_STOP_RED = +6.0   # came to rest behind the stop line on red
REWARD_ROUNDABOUT_CLEAR = +8.0  # entered and exited the roundabout cleanly
REWARD_ROUNDABOUT_COURTESY = +3.0
REWARD_YIELD_EMERGENCY = +5.0

PENALTY_CRASH = -60.0           # at-fault collision with another vehicle
PENALTY_CRASH_PEDESTRIAN = -120.0
PENALTY_RED_LIGHT_RUN = -45.0   # crossed the junction on red or yellow
PENALTY_STOP_LINE_OVERRUN = -20.0
PENALTY_YIELD_VIOLATION = -30.0 # entered the roundabout into circulating traffic
PENALTY_BLOCK_EMERGENCY = -15.0
PENALTY_JERK = -0.3             # accel/brake reversal between macro-actions

# Thresholds the penalty function reads
SAFE_TTC_SECONDS = 2.2
SAFE_HEADWAY_FACTOR = 9.0       # safe gap in px = speed * this factor
MIN_SAFE_HEADWAY = 24.0
SPEED_LIMIT_TOLERANCE = 1.08    # fraction of target_speed treated as legal

# --- Color Palette & Visuals ---
COLOR_BG = (14, 18, 24)
COLOR_GRASS_DAY = (58, 108, 64)
COLOR_GRASS_NIGHT = (18, 36, 24)
COLOR_ROAD_DAY = (46, 48, 54)
COLOR_ROAD_NIGHT = (18, 20, 24)
COLOR_ROAD_MARKING = (236, 238, 242)
COLOR_ROAD_YELLOW = (236, 178, 42)
COLOR_SIDEWALK_DAY = (156, 158, 164)
COLOR_SIDEWALK_NIGHT = (48, 50, 56)
COLOR_STOP_LINE = (248, 250, 252)
COLOR_ISLAND_DAY = (70, 124, 74)
COLOR_ISLAND_NIGHT = (26, 48, 32)

COLOR_TL_RED = (255, 45, 45)
COLOR_TL_RED_GLOW = (255, 70, 70, 80)
COLOR_TL_YELLOW = (255, 200, 30)
COLOR_TL_YELLOW_GLOW = (255, 210, 50, 80)
COLOR_TL_GREEN = (40, 235, 100)
COLOR_TL_GREEN_GLOW = (60, 255, 120, 80)
COLOR_TL_HOUSING = (25, 25, 25)

CAR_COLORS = [
    (220, 48, 48),
    (38, 128, 235),
    (240, 240, 245),
    (35, 35, 40),
    (245, 180, 25),
    (50, 185, 120),
    (155, 80, 220),
    (240, 100, 30),
    (140, 150, 160),
]

UI_PANEL_BG = (18, 22, 30, 230)
UI_PANEL_BORDER = (45, 60, 85)
UI_ACCENT_CYAN = (0, 215, 255)
UI_ACCENT_GREEN = (46, 230, 138)
UI_ACCENT_ORANGE = (255, 155, 40)
UI_ACCENT_RED = (255, 65, 85)
UI_TEXT_WHITE = (240, 245, 250)
UI_TEXT_MUTED = (140, 155, 175)
