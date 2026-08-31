"""
Vehicle Physics, Kinematics, Bounding Box Collisions, Diverse Vehicle Models, and Visual Representation.
Implements OBB collision via SAT, crash impulse dynamics & spin-out,
turn indicator blinking, brake lighting, side mirrors, wheels, and diverse car classes
(Sedan, SUV, Heavy Truck, Transit Bus, Sports Car, Motorcycle, and Emergency Ambulance).
"""
import math
import random
import pygame
import numpy as np
from src.config import (
    VEHICLE_LENGTH, VEHICLE_WIDTH, MAX_SPEED, MIN_SPEED,
    MAX_ACCEL, MAX_BRAKE, EMERGENCY_BRAKE, FRICTION,
    ACTIONS_MAP, CAR_COLORS
)
from src.simulation.sensors import SensorSuite

VEHICLE_PROFILES = {
    'SEDAN': {
        'length': 38.0, 'width': 18.0, 'max_speed': 5.0, 'accel': 0.15, 'brake': 0.35,
        'weight': 1.0, 'is_emergency': False
    },
    'SUV': {
        'length': 42.0, 'width': 20.0, 'max_speed': 4.6, 'accel': 0.13, 'brake': 0.32,
        'weight': 1.3, 'is_emergency': False
    },
    'TRUCK': {
        'length': 58.0, 'width': 22.0, 'max_speed': 3.8, 'accel': 0.09, 'brake': 0.24,
        'weight': 2.5, 'is_emergency': False
    },
    'BUS': {
        'length': 64.0, 'width': 22.0, 'max_speed': 3.6, 'accel': 0.08, 'brake': 0.22,
        'weight': 2.8, 'is_emergency': False
    },
    'SPORTS': {
        'length': 36.0, 'width': 18.0, 'max_speed': 5.8, 'accel': 0.20, 'brake': 0.42,
        'weight': 0.85, 'is_emergency': False
    },
    'MOTORCYCLE': {
        'length': 24.0, 'width': 10.0, 'max_speed': 5.4, 'accel': 0.22, 'brake': 0.38,
        'weight': 0.35, 'is_emergency': False
    },
    'AMBULANCE': {
        'length': 44.0, 'width': 20.0, 'max_speed': 5.5, 'accel': 0.18, 'brake': 0.38,
        'weight': 1.4, 'is_emergency': True
    },
}

def get_rotated_rect_corners(cx, cy, length, width, angle):
    """
    Compute the 4 world-coordinate corners of a rotated rectangle.
    angle is in radians.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    hl = length / 2.0
    hw = width / 2.0

    local_corners = [
        (hl, hw),
        (hl, -hw),
        (-hl, -hw),
        (-hl, hw)
    ]

    world_corners = []
    for lx, ly in local_corners:
        wx = cx + (lx * cos_a - ly * sin_a)
        wy = cy + (lx * sin_a + ly * cos_a)
        world_corners.append((wx, wy))

    return world_corners

def check_sat_collision(corners_a, corners_b):
    """
    Separating Axis Theorem (SAT) collision check between two oriented convex quadrilaterals.
    Returns True if overlapping, False otherwise.
    """
    for corners in (corners_a, corners_b):
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            normal = (-(p2[1] - p1[1]), p2[0] - p1[0])
            norm_len = math.hypot(normal[0], normal[1])
            if norm_len < 1e-6:
                continue
            normal = (normal[0] / norm_len, normal[1] / norm_len)

            min_a = float('inf')
            max_a = float('-inf')
            for cp in corners_a:
                proj = cp[0] * normal[0] + cp[1] * normal[1]
                min_a = min(min_a, proj)
                max_a = max(max_a, proj)

            min_b = float('inf')
            max_b = float('-inf')
            for cp in corners_b:
                proj = cp[0] * normal[0] + cp[1] * normal[1]
                min_b = min(min_b, proj)
                max_b = max(max_b, proj)

            if max_a < min_b or max_b < min_a:
                return False

    return True


class Vehicle:
    _id_counter = 0

    def __init__(self, route, v_type=None, spawn_speed=2.5, color=None):
        Vehicle._id_counter += 1
        self.id = Vehicle._id_counter
        self.route = route

        # Vehicle Type Selection
        if v_type is None:
            # Weighted random choice: 50% Sedan, 18% SUV, 10% Sports, 8% Truck, 6% Bus, 5% Motorcycle, 3% Ambulance
            v_type = random.choices(
                ['SEDAN', 'SUV', 'SPORTS', 'TRUCK', 'BUS', 'MOTORCYCLE', 'AMBULANCE'],
                weights=[50, 18, 10, 8, 6, 5, 3]
            )[0]

        self.v_type = v_type
        prof = VEHICLE_PROFILES.get(v_type, VEHICLE_PROFILES['SEDAN'])

        self.length = prof['length']
        self.width = prof['width']
        self.max_accel = prof['accel']
        self.max_brake = prof['brake']
        self.weight = prof['weight']
        self.is_emergency = prof['is_emergency']

        if self.is_emergency:
            self.color = (245, 245, 250) # Pure Ambulance White
        elif color is not None:
            self.color = color
        elif v_type == 'BUS':
            self.color = (40, 130, 220) # Transit Blue
        elif v_type == 'TRUCK':
            self.color = (180, 75, 45) # Rusty / Orange Truck
        elif v_type == 'MOTORCYCLE':
            self.color = (235, 40, 40)
        else:
            self.color = random.choice(CAR_COLORS)

        # Path progression & Kinematics
        self.path_distance = 0.0
        self.speed = spawn_speed
        self.target_speed = prof['max_speed'] * random.uniform(0.88, 1.05)
        self.accel = 0.0
        self.is_braking = False

        # Orientation & Position
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        self.update_pose_from_path()

        # Crash dynamics & impulse spin
        self.angular_vel = 0.0
        self.crash_vx = 0.0
        self.crash_vy = 0.0

        # State lifecycle
        self.is_alive = True
        self.has_crashed = False
        self.is_at_fault = False
        self.has_passed_intersection = False
        self.tl_state_at_entry = None
        self.passed_reward_granted = False
        self.has_finished = False
        self.time_alive = 0.0
        self.total_reward = 0.0

        # RL Memory & action
        self.last_state = None
        self.last_action = 0
        self.action_name = "COAST"
        self.decision_step = 0

        # Blinkers & lighting
        self.turn_signal_timer = 0
        self.left_blinker = False
        self.right_blinker = False
        if route.turn_type == 'LEFT':
            self.left_blinker = True
        elif route.turn_type == 'RIGHT':
            self.right_blinker = True

        # Sensor suite
        self.sensors = SensorSuite(self)

    def update_pose_from_path(self):
        """Update x, y, and angle based on current path_distance along route."""
        pose = self.route.get_pose_at_distance(self.path_distance)
        if pose is not None:
            self.x, self.y, self.angle = pose
            
            # Apply pull over offset (shift right perpendicular to angle)
            offset = getattr(self, 'pull_over_offset', 0)
            if offset > 0:
                self.x += math.cos(self.angle + math.pi/2) * offset
                self.y += math.sin(self.angle + math.pi/2) * offset
        else:
            self.has_finished = True

    def get_corners(self):
        return get_rotated_rect_corners(self.x, self.y, self.length, self.width, self.angle)

    def apply_action(self, action_idx):
        """Apply discrete action from neural network scaled by vehicle class limits."""
        self.last_action = action_idx
        self.action_name = ACTIONS_MAP.get(action_idx, "COAST")

        if action_idx == 0: # COAST
            self.accel = -FRICTION
            self.is_braking = False
        elif action_idx == 1: # ACCEL_MILD
            self.accel = self.max_accel * 0.55
            self.is_braking = False
        elif action_idx == 2: # ACCEL_FULL
            self.accel = self.max_accel
            self.is_braking = False
        elif action_idx == 3: # BRAKE_MILD
            self.accel = -self.max_brake * 0.55
            self.is_braking = True
        elif action_idx == 4: # BRAKE_HARD
            self.accel = -self.max_brake * 1.5
            self.is_braking = True

    def update_physics(self, dt=1.0/60.0, friction_coeff=1.0, current_tl_state=None, all_vehicles=None, puddles=None, v2v_enabled=False):
        """Update speed, trajectory, or crash impulse spin-out with proper dt scaling and wet road grip."""
        dt_scale = dt * 60.0
        self.time_alive += dt
        self.turn_signal_timer += 1

        if not self.is_alive or self.has_crashed:
            self.x += self.crash_vx * dt_scale
            self.y += self.crash_vy * dt_scale
            self.angle += self.angular_vel * dt_scale
            self.crash_vx *= (0.92 ** dt_scale)
            self.crash_vy *= (0.92 ** dt_scale)
            self.angular_vel *= (0.90 ** dt_scale)
            self.speed = max(0.0, self.speed - 0.15 * dt_scale)
            return

        self.v2v_active = self.is_braking

        # V2V Sync & Ambulance Pull-over Logic
        if all_vehicles:
            dist_ahead, lead_car = self.get_leading_car_info(all_vehicles)
            if v2v_enabled and lead_car and dist_ahead < 60.0 and lead_car.v2v_active:
                self.is_braking = True
                self.accel = -self.max_brake * 1.5
                self.v2v_active = True
                self.v2v_triggered = True
            else:
                self.v2v_triggered = False

            # Ambulance check (check if ambulance is behind us within 80px)
            if not self.is_emergency:
                ambulance_behind = False
                for other in all_vehicles:
                    if other.is_alive and other.is_emergency and other.route.id == self.route.id:
                        dist_behind = self.path_distance - other.path_distance
                        if 0 < dist_behind < 80.0:
                            ambulance_behind = True
                            break
                if ambulance_behind:
                    self.is_braking = True
                    self.accel = -self.max_brake
                    # Slightly shift right (pull over) if not already shifted
                    self.pull_over_offset = getattr(self, 'pull_over_offset', 0)
                    if self.pull_over_offset < 8.0:
                        self.pull_over_offset += 0.5 * dt_scale
                else:
                    self.pull_over_offset = getattr(self, 'pull_over_offset', 0)
                    if self.pull_over_offset > 0:
                        self.pull_over_offset -= 0.5 * dt_scale

        # Puddle Hydroplaning Logic
        self.is_hydroplaning = False
        if puddles and self.speed > 2.0:
            for p in puddles:
                if math.hypot(self.x - p.x, self.y - p.y) < p.radius:
                    self.is_hydroplaning = True
                    # Random slip angle
                    self.angular_vel = random.uniform(-0.05, 0.05)
                    self.speed *= 0.95 # lose speed when hydroplaning
                    break

        if self.is_hydroplaning:
            self.angle += self.angular_vel * dt_scale

        # Integrate acceleration with road surface friction grip
        if self.is_braking:
            effective_accel = self.accel * friction_coeff # Longer stopping distance in rain!
        else:
            effective_accel = self.accel * min(1.0, friction_coeff + 0.1)

        self.speed += effective_accel * dt_scale
        self.speed = max(MIN_SPEED, min(self.target_speed, self.speed))

        # Move along trajectory
        self.path_distance += self.speed * dt_scale
        self.update_pose_from_path()

        # Check if crossed intersection stop line
        if not self.has_passed_intersection and self.path_distance >= self.route.stop_line_dist:
            self.has_passed_intersection = True
            self.tl_state_at_entry = current_tl_state

    def get_distance_to_stop_line(self):
        """Return distance in pixels along route to the stop line."""
        return self.route.stop_line_dist - self.path_distance

    def get_leading_car_info(self, all_vehicles):
        """Find leading vehicle on same route ahead of this vehicle."""
        closest_dist = float('inf')
        lead_car = None
        for other in all_vehicles:
            if other.id == self.id or not other.is_alive:
                continue
            if other.route.id == self.route.id:
                dist_ahead = other.path_distance - self.path_distance
                if 0 < dist_ahead < closest_dist:
                    closest_dist = dist_ahead
                    lead_car = other
        if lead_car is not None:
            return closest_dist, lead_car
        return None, None

    def crash(self, is_at_fault=True):
        """Trigger vehicle crash state with realistic spin-out impulse."""
        self.has_crashed = True
        self.is_at_fault = is_at_fault
        self.is_alive = False
        self.angular_vel = random.uniform(-0.12, 0.12)
        impact_dir = self.angle + random.uniform(-0.8, 0.8)
        impact_speed = max(0.8, self.speed * 0.6)
        self.crash_vx = math.cos(impact_dir) * impact_speed
        self.crash_vy = math.sin(impact_dir) * impact_speed
        self.speed = 0.0

    def draw(self, surface, is_night=False, is_selected=False):
        """Draw detailed vehicle top-down with specialized styling per vehicle type."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        hl = self.length / 2.0
        hw = self.width / 2.0

        # 1. Shadow beneath vehicle
        shadow_offset = (3, 4) if not is_night else (1, 2)
        corners = self.get_corners()
        shadow_pts = [(int(p[0] + shadow_offset[0]), int(p[1] + shadow_offset[1])) for p in corners]
        s_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(s_surf, (0, 0, 0, 85 if not is_night else 140), shadow_pts)
        surface.blit(s_surf, (0, 0))

        # 2. Wheels
        if self.v_type != 'MOTORCYCLE':
            wheel_w = 4.0
            wheel_l = 8.0
            wheel_offsets = [
                (hl * 0.65, hw + 0.5),
                (hl * 0.65, -hw - 0.5),
                (-hl * 0.65, hw + 0.5),
                (-hl * 0.65, -hw - 0.5)
            ]
            for w_lx, w_ly in wheel_offsets:
                wx = self.x + (w_lx * cos_a - w_ly * sin_a)
                wy = self.y + (w_lx * sin_a + w_ly * cos_a)
                w_corners = get_rotated_rect_corners(wx, wy, wheel_l, wheel_w, self.angle)
                pygame.draw.polygon(surface, (18, 18, 20), [(int(p[0]), int(p[1])) for p in w_corners])
        else:
            # Motorcycle wheels (front & rear centerline)
            for w_lx in (hl * 0.7, -hl * 0.7):
                wx = self.x + w_lx * cos_a
                wy = self.y + w_lx * sin_a
                w_corners = get_rotated_rect_corners(wx, wy, 7.0, 3.0, self.angle)
                pygame.draw.polygon(surface, (15, 15, 15), [(int(p[0]), int(p[1])) for p in w_corners])

        # 3. Main Chassis Body
        pts = [(int(p[0]), int(p[1])) for p in corners]
        pygame.draw.polygon(surface, self.color, pts)

        border_col = (max(0, self.color[0] - 45), max(0, self.color[1] - 45), max(0, self.color[2] - 45))
        pygame.draw.polygon(surface, border_col, pts, 2)

        # 4. Custom Styling per Vehicle Class
        if self.v_type == 'BUS':
            # Public Transit Bus: multiple passenger windows along roof
            for win_i in range(-3, 4):
                win_lx = win_i * 7.5
                for win_side in (hw * 0.6, -hw * 0.6):
                    win_x = self.x + (win_lx * cos_a - win_side * sin_a)
                    win_y = self.y + (win_lx * sin_a + win_side * cos_a)
                    w_corners = get_rotated_rect_corners(win_x, win_y, 5.0, 3.5, self.angle)
                    pygame.draw.polygon(surface, (30, 48, 68), [(int(p[0]), int(p[1])) for p in w_corners])
            # Roof vents
            pygame.draw.circle(surface, (200, 210, 220), (int(self.x + 10 * cos_a), int(self.y + 10 * sin_a)), 3)
            pygame.draw.circle(surface, (200, 210, 220), (int(self.x - 10 * cos_a), int(self.y - 10 * sin_a)), 3)

        elif self.v_type == 'TRUCK':
            # Heavy Truck: Cab in front, cargo container in back
            cab_c = (self.x + cos_a * (hl * 0.55), self.y + sin_a * (hl * 0.55))
            cab_corners = get_rotated_rect_corners(cab_c[0], cab_c[1], hl * 0.7, hw * 1.8, self.angle)
            pygame.draw.polygon(surface, (min(255, self.color[0] + 30), min(255, self.color[1] + 30), min(255, self.color[2] + 30)), [(int(p[0]), int(p[1])) for p in cab_corners])
            # Cargo box lines
            cargo_c = (self.x - cos_a * (hl * 0.35), self.y - sin_a * (hl * 0.35))
            cargo_corners = get_rotated_rect_corners(cargo_c[0], cargo_c[1], hl * 1.2, hw * 1.7, self.angle)
            pygame.draw.polygon(surface, (160, 165, 175), [(int(p[0]), int(p[1])) for p in cargo_corners], 2)

        elif self.v_type == 'AMBULANCE':
            # Ambulance Red Cross on roof & Emergency Strobes
            # Red Cross Emblem
            pygame.draw.line(surface, (240, 30, 30), (self.x - 5 * cos_a, self.y - 5 * sin_a), (self.x + 5 * cos_a, self.y + 5 * sin_a), 3)
            pygame.draw.line(surface, (240, 30, 30), (self.x - 5 * sin_a, self.y + 5 * cos_a), (self.x + 5 * sin_a, self.y - 5 * cos_a), 3)
            # Flashing LED Light Bar (Red / Blue strobe)
            strobe_col = (255, 30, 30) if (self.turn_signal_timer // 8) % 2 == 0 else (30, 120, 255)
            bar_c = (self.x + cos_a * (hl * 0.2), self.y + sin_a * (hl * 0.2))
            bar_corners = get_rotated_rect_corners(bar_c[0], bar_c[1], 4.0, hw * 1.4, self.angle)
            pygame.draw.polygon(surface, strobe_col, [(int(p[0]), int(p[1])) for p in bar_corners])

        elif self.v_type == 'MOTORCYCLE':
            # Rider body & helmet
            pygame.draw.circle(surface, (230, 210, 40), (int(self.x), int(self.y)), 4) # Helmet
            # Handlebars
            hb_p1 = (self.x + 6 * cos_a - 4 * sin_a, self.y + 6 * sin_a + 4 * cos_a)
            hb_p2 = (self.x + 6 * cos_a + 4 * sin_a, self.y + 6 * sin_a - 4 * cos_a)
            pygame.draw.line(surface, (40, 40, 40), hb_p1, hb_p2, 2)

        else: # Standard Sedans, SUVs, Sports Cars
            # Windshields & Windows
            fw_c = (self.x + cos_a * (self.length * 0.15), self.y + sin_a * (self.length * 0.15))
            fw_corners = get_rotated_rect_corners(fw_c[0], fw_c[1], self.length * 0.22, self.width * 0.76, self.angle)
            pygame.draw.polygon(surface, (28, 42, 58), [(int(p[0]), int(p[1])) for p in fw_corners])

            rw_c = (self.x - cos_a * (self.length * 0.25), self.y - sin_a * (self.length * 0.25))
            rw_corners = get_rotated_rect_corners(rw_c[0], rw_c[1], self.length * 0.18, self.width * 0.70, self.angle)
            pygame.draw.polygon(surface, (28, 42, 58), [(int(p[0]), int(p[1])) for p in rw_corners])

            # Roof & Metallic Gloss Highlight
            roof_c = (self.x - cos_a * (self.length * 0.05), self.y - sin_a * (self.length * 0.05))
            roof_corners = get_rotated_rect_corners(roof_c[0], roof_c[1], self.length * 0.32, self.width * 0.64, self.angle)
            roof_color = (min(255, self.color[0] + 25), min(255, self.color[1] + 25), min(255, self.color[2] + 25))
            pygame.draw.polygon(surface, roof_color, [(int(p[0]), int(p[1])) for p in roof_corners])

            # Gloss reflection strip
            gloss_p1 = (roof_c[0] + cos_a * (self.length * 0.10), roof_c[1] + sin_a * (self.length * 0.10))
            gloss_p2 = (roof_c[0] - cos_a * (self.length * 0.10), roof_c[1] - sin_a * (self.length * 0.10))
            pygame.draw.line(surface, (min(255, self.color[0] + 65), min(255, self.color[1] + 65), min(255, self.color[2] + 65)), gloss_p1, gloss_p2, 2)

        # 5. Headlights
        front_r = (self.x + hl * cos_a - (hw - 2.5) * sin_a, self.y + hl * sin_a + (hw - 2.5) * cos_a)
        front_l = (self.x + hl * cos_a + (hw - 2.5) * sin_a, self.y + hl * sin_a - (hw - 2.5) * cos_a)
        hl_col = (255, 255, 230) if is_night else (240, 240, 210)
        pygame.draw.circle(surface, hl_col, (int(front_r[0]), int(front_r[1])), 2)
        pygame.draw.circle(surface, hl_col, (int(front_l[0]), int(front_l[1])), 2)

        # 6. Taillights & Brake Lights
        back_r = (self.x - hl * cos_a - (hw - 2.5) * sin_a, self.y - hl * sin_a + (hw - 2.5) * cos_a)
        back_l = (self.x - hl * cos_a + (hw - 2.5) * sin_a, self.y - hl * sin_a - (hw - 2.5) * cos_a)
        if self.is_braking:
            tl_col = (255, 35, 35)
            pygame.draw.circle(surface, tl_col, (int(back_r[0]), int(back_r[1])), 3)
            pygame.draw.circle(surface, tl_col, (int(back_l[0]), int(back_l[1])), 3)
        else:
            tl_col = (190, 25, 25) if is_night else (130, 25, 25)
            pygame.draw.circle(surface, tl_col, (int(back_r[0]), int(back_r[1])), 2)
            pygame.draw.circle(surface, tl_col, (int(back_l[0]), int(back_l[1])), 2)

        # 7. Turn Indicator blinkers
        if (self.turn_signal_timer // 16) % 2 == 0:
            amber_col = (255, 185, 25)
            if self.left_blinker:
                pygame.draw.circle(surface, amber_col, (int(front_l[0]), int(front_l[1])), 3)
                pygame.draw.circle(surface, amber_col, (int(back_l[0]), int(back_l[1])), 3)
            if self.right_blinker:
                pygame.draw.circle(surface, amber_col, (int(front_r[0]), int(front_r[1])), 3)
                pygame.draw.circle(surface, amber_col, (int(back_r[0]), int(back_r[1])), 3)

        # V2V Wireless Ripples
        if getattr(self, 'v2v_active', False):
            ripple_radius = (self.time_alive * 30.0) % 40.0
            pygame.draw.circle(surface, (0, 200, 255), (int(self.x), int(self.y)), int(ripple_radius), 1)
            pygame.draw.circle(surface, (0, 200, 255), (int(self.x), int(self.y)), int((ripple_radius + 15) % 40), 1)

        # Thought Vectors (Action Intent)
        if getattr(self, 'action_name', None):
            try:
                font = pygame.font.SysFont("Segoe UI, Arial", 10, bold=True)
                icon = ""
                if "BRAKE" in self.action_name:
                    icon = "[BRAKE]"
                elif "ACCEL" in self.action_name:
                    icon = "[ACCEL]"
                elif getattr(self, 'is_hydroplaning', False):
                    icon = "[SLIP]"
                elif getattr(self, 'v2v_triggered', False):
                    icon = "[V2V]"
                
                if icon:
                    # Draw a nice fluent pill background for the text
                    txt = font.render(icon, True, (255, 255, 255))
                    txt_w = txt.get_width()
                    pill_rect = pygame.Rect(int(self.x) - txt_w//2 - 2, int(self.y) - 22, txt_w + 4, 14)
                    pygame.draw.rect(surface, (40, 40, 40), pill_rect, border_radius=4)
                    surface.blit(txt, (int(self.x) - txt_w//2, int(self.y) - 21))
            except:
                pass

        # 8. Selection Ring if tracked
        if is_selected:
            pygame.draw.circle(surface, (0, 230, 255), (int(self.x), int(self.y)), int(self.length * 0.75), 2)
