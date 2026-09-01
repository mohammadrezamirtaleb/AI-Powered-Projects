"""
Pedestrian Simulation System for Crossroad Pedestrian Crosswalks.
Manages pedestrians walking across zebra crossings during red traffic phases,
with walking kinematics and detection bounds.
"""
import math
import random
import pygame

class Pedestrian:
    _id_counter = 0

    def __init__(self, crosswalk_name, direction, start_pos, target_pos, speed=1.1, color=None):
        Pedestrian._id_counter += 1
        self.id = Pedestrian._id_counter
        self.crosswalk_name = crosswalk_name # 'N', 'S', 'E', 'W'
        self.direction = direction           # 1 (forward) or -1 (reverse)
        self.x = float(start_pos[0])
        self.y = float(start_pos[1])
        self.target_pos = target_pos
        self.speed = speed * random.uniform(0.85, 1.15)
        self.color = color if color is not None else random.choice([
            (220, 70, 70), (50, 140, 230), (240, 200, 40),
            (60, 200, 120), (200, 100, 220), (230, 130, 40)
        ])
        self.skin_color = (255, 215, 175)
        self.radius = 7.0
        self.is_alive = True
        self.has_finished = False
        self.walk_timer = random.uniform(0, 10)

        # Calculate movement angle
        dx = target_pos[0] - self.x
        dy = target_pos[1] - self.y
        self.angle = math.atan2(dy, dx)
        self.total_dist = math.hypot(dx, dy)
        self.dist_walked = 0.0

    def update(self, dt=1.0/60.0):
        if not self.is_alive:
            return

        dt_scale = dt * 60.0
        self.walk_timer += 0.2 * dt_scale
        step = self.speed * dt_scale
        self.dist_walked += step

        self.x += math.cos(self.angle) * step
        self.y += math.sin(self.angle) * step

        if self.dist_walked >= self.total_dist:
            self.has_finished = True
            self.is_alive = False

    def get_corners(self):
        """Returns 4 bounding box corners for SAT / LiDAR collision avoidance."""
        r = self.radius + 1.0
        return [
            (self.x + r, self.y + r),
            (self.x - r, self.y + r),
            (self.x - r, self.y - r),
            (self.x + r, self.y - r)
        ]

    def draw(self, surface, is_night=False):
        if not self.is_alive:
            return

        ix = int(self.x)
        iy = int(self.y)

        # 1. Subtle shadow
        if not hasattr(Pedestrian, '_shadow_day'):
            Pedestrian._shadow_day = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(Pedestrian._shadow_day, (0, 0, 0, 80), (8, 8), 5)
            Pedestrian._shadow_night = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(Pedestrian._shadow_night, (0, 0, 0, 130), (8, 8), 5)
            
        s_surf = Pedestrian._shadow_night if is_night else Pedestrian._shadow_day
        surface.blit(s_surf, (ix - 8, iy - 6))

        # 2. Swinging shoulders / arms
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        arm_swing = math.sin(self.walk_timer) * 2.5

        left_arm = (ix - sin_a * 4 + cos_a * arm_swing, iy + cos_a * 4 + sin_a * arm_swing)
        right_arm = (ix + sin_a * 4 - cos_a * arm_swing, iy - cos_a * 4 - sin_a * arm_swing)

        pygame.draw.circle(surface, self.skin_color, (int(left_arm[0]), int(left_arm[1])), 2)
        pygame.draw.circle(surface, self.skin_color, (int(right_arm[0]), int(right_arm[1])), 2)

        # 3. Torso / Shirt
        pygame.draw.circle(surface, self.color, (ix, iy), int(self.radius))

        # 4. Head & Hair
        head_radius = 2.5
        head_pos = (int(ix + cos_a * 1.5), int(iy + sin_a * 1.5))
        pygame.draw.circle(surface, self.skin_color, head_pos, int(head_radius))
        pygame.draw.circle(surface, (50, 35, 25), head_pos, int(head_radius), 1)


class PedestrianManager:
    def __init__(self, intersection):
        self.intersection = intersection
        self.pedestrians = []
        self.spawn_timer = 2.0
        self._init_crosswalk_routes()

    def _init_crosswalk_routes(self):
        cx = self.intersection.cx
        cy = self.intersection.cy
        hrw = self.intersection.half_rw

        # Crosswalk crossing paths: (name, start, target)
        # 4px offset keeps them generally centered in the crosswalk
        self.crosswalk_paths = {
            'N': [
                ((cx - hrw + 4, cy - hrw - 4), (cx + hrw - 4, cy - hrw - 4)),
                ((cx + hrw - 4, cy - hrw - 4), (cx - hrw + 4, cy - hrw - 4))
            ],
            'S': [
                ((cx - hrw + 4, cy + hrw + 4), (cx + hrw - 4, cy + hrw + 4)),
                ((cx + hrw - 4, cy + hrw + 4), (cx - hrw + 4, cy + hrw + 4))
            ],
            'W': [
                ((cx - hrw - 4, cy - hrw + 4), (cx - hrw - 4, cy + hrw - 4)),
                ((cx - hrw - 4, cy + hrw - 4), (cx - hrw - 4, cy - hrw + 4))
            ],
            'E': [
                ((cx + hrw + 4, cy - hrw + 4), (cx + hrw + 4, cy + hrw - 4)),
                ((cx + hrw + 4, cy + hrw - 4), (cx + hrw + 4, cy - hrw + 4))
            ]
        }

    def update(self, dt, traffic_controller, jaywalking_enabled=False):
        # Update existing pedestrians
        for ped in self.pedestrians:
            ped.update(dt)
        self.pedestrians = [p for p in self.pedestrians if p.is_alive]

        # Spawning logic: spawn when corresponding traffic light is RED
        self.spawn_timer -= dt
        if self.spawn_timer <= 0 and len(self.pedestrians) < 15:
            self.spawn_timer = random.uniform(0.8, 2.0)

            # Check which crosswalks have RED lights for vehicle traffic
            eligible = []
            
            # Get remaining time in current phase to prevent spawning right before green
            cur_phase = traffic_controller.current_phase
            from src.config import PHASE_DURATIONS
            max_dur = PHASE_DURATIONS.get(cur_phase, 8.0)
            time_left = max_dur - traffic_controller.timer

            if time_left > 3.0:
                for cw_name in ['N', 'S', 'E', 'W']:
                    if traffic_controller.get_light_state(cw_name) == 'RED':
                        eligible.append(cw_name)

            # 25% chance to spawn a jaywalker if enabled
            if jaywalking_enabled and random.random() < 0.25:
                # Random side to random side using dynamic intersection bounds
                cx = self.intersection.cx
                cy = self.intersection.cy
                sides = [
                    (cx - 150, random.uniform(cy - 200, cy + 200)),
                    (cx + 150, random.uniform(cy - 200, cy + 200)),
                    (random.uniform(cx - 200, cx + 200), cy - 150),
                    (random.uniform(cx - 200, cx + 200), cy + 150)
                ]
                start_side = random.choice(sides)
                sides.remove(start_side)
                end_side = random.choice(sides)
                
                ped = Pedestrian('JAYWALKER', 1, start_side, end_side, speed=1.3)
                self.pedestrians.append(ped)
            elif eligible:
                chosen_cw = random.choice(eligible)
                path = random.choice(self.crosswalk_paths[chosen_cw])
                ped = Pedestrian(chosen_cw, 1, path[0], path[1])
                self.pedestrians.append(ped)

    def draw(self, surface, is_night=False):
        for ped in self.pedestrians:
            ped.draw(surface, is_night)
