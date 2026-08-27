"""
Master Graphics Renderer for the 4-Way Autonomous Intersection Simulation.
Pre-renders static environment (asphalt, markings, zebra crossings, curbs) to cached surfaces for maximum FPS,
and renders dynamic elements (traffic lights, vehicles, LiDAR sensor overlays).
"""
import math
import random
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, CENTER_X, CENTER_Y,
    ROAD_WIDTH, LANE_WIDTH, COLOR_BG, COLOR_GRASS_DAY,
    COLOR_GRASS_NIGHT, COLOR_ROAD_DAY, COLOR_ROAD_NIGHT,
    COLOR_ROAD_MARKING, COLOR_ROAD_YELLOW, COLOR_SIDEWALK_DAY,
    COLOR_SIDEWALK_NIGHT, COLOR_STOP_LINE, COLOR_TL_RED,
    COLOR_TL_YELLOW, COLOR_TL_GREEN, COLOR_TL_HOUSING
)

def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors by factor t in [0, 1]."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )

class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.cx = CENTER_X
        self.cy = CENTER_Y
        self.rw = ROAD_WIDTH
        self.hrw = ROAD_WIDTH / 2.0
        self.lw = LANE_WIDTH
        self.ray_surface = None

        # Pre-rendered background caches for maximum 60+ FPS performance
        self.bg_day_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.bg_night_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._build_static_background(self.bg_day_surface, night_factor=0.0)
        self._build_static_background(self.bg_night_surface, night_factor=1.0)

    def _build_static_background(self, surface, night_factor=0.0):
        """Pre-renders grass, sidewalks, asphalt grain, curbs, zebra crossings, and lane markings."""
        # 1. Background Grass
        grass_col = lerp_color(COLOR_GRASS_DAY, COLOR_GRASS_NIGHT, night_factor)
        surface.fill(grass_col)

        # Subtle grass noise / blades
        random.seed(42)
        for _ in range(600):
            gx = random.randint(0, SCREEN_WIDTH - 360)
            gy = random.randint(0, SCREEN_HEIGHT)
            if not (self.cx - self.hrw - 30 <= gx <= self.cx + self.hrw + 30 or
                    self.cy - self.hrw - 30 <= gy <= self.cy + self.hrw + 30):
                blade_c = (max(0, grass_col[0] - 8), min(255, grass_col[1] + 10), max(0, grass_col[2] - 6))
                pygame.draw.line(surface, blade_c, (gx, gy), (gx + random.randint(-2, 2), gy - random.randint(3, 6)), 1)

        # 2. Sidewalk borders with beveled curbs
        sw_col = lerp_color(COLOR_SIDEWALK_DAY, COLOR_SIDEWALK_NIGHT, night_factor)
        curb_light = (min(255, sw_col[0] + 30), min(255, sw_col[1] + 30), min(255, sw_col[2] + 30))
        curb_dark = (max(0, sw_col[0] - 30), max(0, sw_col[1] - 30), max(0, sw_col[2] - 30))
        sw_offset = self.hrw + 14

        # Vertical Sidewalks
        pygame.draw.rect(surface, sw_col, (self.cx - sw_offset, 0, sw_offset * 2, SCREEN_HEIGHT))
        # Horizontal Sidewalks
        pygame.draw.rect(surface, sw_col, (0, self.cy - sw_offset, SCREEN_WIDTH - 360, sw_offset * 2))

        # Curb bevel highlights
        pygame.draw.line(surface, curb_light, (self.cx - sw_offset, 0), (self.cx - sw_offset, SCREEN_HEIGHT), 2)
        pygame.draw.line(surface, curb_dark, (self.cx + sw_offset, 0), (self.cx + sw_offset, SCREEN_HEIGHT), 2)
        pygame.draw.line(surface, curb_light, (0, self.cy - sw_offset), (SCREEN_WIDTH - 360, self.cy - sw_offset), 2)
        pygame.draw.line(surface, curb_dark, (0, self.cy + sw_offset), (SCREEN_WIDTH - 360, self.cy + sw_offset), 2)

        # 3. Main Asphalt Roads
        road_col = lerp_color(COLOR_ROAD_DAY, COLOR_ROAD_NIGHT, night_factor)
        pygame.draw.rect(surface, road_col, (self.cx - self.hrw, 0, self.rw, SCREEN_HEIGHT))
        pygame.draw.rect(surface, road_col, (0, self.cy - self.hrw, SCREEN_WIDTH - 360, self.rw))

        # Asphalt Grain & Texture Speckles
        for _ in range(800):
            rx = random.choice([
                random.randint(int(self.cx - self.hrw), int(self.cx + self.hrw)),
                random.randint(0, SCREEN_WIDTH - 360)
            ])
            ry = random.choice([
                random.randint(0, SCREEN_HEIGHT),
                random.randint(int(self.cy - self.hrw), int(self.cy + self.hrw))
            ])
            grain_val = random.randint(-12, 12)
            grain_c = (
                max(0, min(255, road_col[0] + grain_val)),
                max(0, min(255, road_col[1] + grain_val)),
                max(0, min(255, road_col[2] + grain_val))
            )
            surface.set_at((rx, ry), grain_c)

        # 4. Zebra Crossings (Pedestrian Crosswalks)
        marking_col = COLOR_ROAD_MARKING
        self._draw_zebra_crossings(surface, marking_col)

        # 5. Stop Lines
        stop_line_thick = 4
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx, self.cy - self.hrw), (self.cx + self.hrw, self.cy - self.hrw), stop_line_thick)
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx - self.hrw, self.cy + self.hrw), (self.cx, self.cy + self.hrw), stop_line_thick)
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx + self.hrw, self.cy - self.hrw), (self.cx + self.hrw, self.cy), stop_line_thick)
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx - self.hrw, self.cy), (self.cx - self.hrw, self.cy + self.hrw), stop_line_thick)

        # 6. Yellow Double Center Dividers
        yellow_col = COLOR_ROAD_YELLOW
        self._draw_double_yellow_line(surface, (self.cx, 0), (self.cx, self.cy - self.hrw - 18), yellow_col)
        self._draw_double_yellow_line(surface, (self.cx, self.cy + self.hrw + 18), (self.cx, SCREEN_HEIGHT), yellow_col)
        self._draw_double_yellow_line_h(surface, (0, self.cy), (self.cx - self.hrw - 18, self.cy), yellow_col)
        self._draw_double_yellow_line_h(surface, (self.cx + self.hrw + 18, self.cy), (SCREEN_WIDTH - 360, self.cy), yellow_col)

        # 7. Dashed White Lane Markings
        self._draw_dashed_lane_markings(surface, marking_col)

        # 8. Lane Turn Arrows painted on asphalt
        self._draw_road_turn_arrows(surface, marking_col)

    def render_environment(self, surface, night_factor=0.0):
        """Fast blit of pre-rendered static background with alpha interpolation."""
        if night_factor <= 0.01:
            surface.blit(self.bg_day_surface, (0, 0))
        elif night_factor >= 0.99:
            surface.blit(self.bg_night_surface, (0, 0))
        else:
            surface.blit(self.bg_day_surface, (0, 0))
            self.bg_night_surface.set_alpha(int(255 * night_factor))
            surface.blit(self.bg_night_surface, (0, 0))
            self.bg_night_surface.set_alpha(255)

    def _draw_double_yellow_line(self, surface, p1, p2, color):
        pygame.draw.line(surface, color, (p1[0] - 2, p1[1]), (p2[0] - 2, p2[1]), 2)
        pygame.draw.line(surface, color, (p1[0] + 2, p1[1]), (p2[0] + 2, p2[1]), 2)

    def _draw_double_yellow_line_h(self, surface, p1, p2, color):
        pygame.draw.line(surface, color, (p1[0], p1[1] - 2), (p2[0], p2[1] - 2), 2)
        pygame.draw.line(surface, color, (p1[0], p1[1] + 2), (p2[0], p2[1] + 2), 2)

    def _draw_dashed_lane_markings(self, surface, color):
        dash_len = 16
        dash_gap = 14

        # North road dashed lines
        for offset in (self.lw, -self.lw):
            x = self.cx + offset
            y = 0
            while y < self.cy - self.hrw - 20:
                pygame.draw.line(surface, color, (x, y), (x, min(self.cy - self.hrw - 20, y + dash_len)), 2)
                y += dash_len + dash_gap

        # South road dashed lines
        for offset in (-self.lw, self.lw):
            x = self.cx + offset
            y = self.cy + self.hrw + 20
            while y < SCREEN_HEIGHT:
                pygame.draw.line(surface, color, (x, y), (x, min(SCREEN_HEIGHT, y + dash_len)), 2)
                y += dash_len + dash_gap

        # West road dashed lines
        for offset in (self.lw, -self.lw):
            y = self.cy + offset
            x = 0
            while x < self.cx - self.hrw - 20:
                pygame.draw.line(surface, color, (x, y), (min(self.cx - self.hrw - 20, x + dash_len), y), 2)
                x += dash_len + dash_gap

        # East road dashed lines
        for offset in (-self.lw, self.lw):
            y = self.cy + offset
            x = self.cx + self.hrw + 20
            while x < SCREEN_WIDTH - 360:
                pygame.draw.line(surface, color, (x, y), (min(SCREEN_WIDTH - 360, x + dash_len), y), 2)
                x += dash_len + dash_gap

    def _draw_zebra_crossings(self, surface, color):
        stripe_w = 6
        stripe_gap = 5
        crosswalk_depth = 14

        # North Crosswalk
        cy_top = self.cy - self.hrw - 6
        for x in range(int(self.cx - self.hrw + 4), int(self.cx + self.hrw - 4), stripe_w + stripe_gap):
            pygame.draw.rect(surface, color, (x, cy_top - crosswalk_depth, stripe_w, crosswalk_depth))

        # South Crosswalk
        cy_bot = self.cy + self.hrw + 6
        for x in range(int(self.cx - self.hrw + 4), int(self.cx + self.hrw - 4), stripe_w + stripe_gap):
            pygame.draw.rect(surface, color, (x, cy_bot, stripe_w, crosswalk_depth))

        # West Crosswalk
        cx_left = self.cx - self.hrw - 6
        for y in range(int(self.cy - self.hrw + 4), int(self.cy + self.hrw - 4), stripe_w + stripe_gap):
            pygame.draw.rect(surface, color, (cx_left - crosswalk_depth, y, crosswalk_depth, stripe_w))

        # East Crosswalk
        cx_right = self.cx + self.hrw + 6
        for y in range(int(self.cy - self.hrw + 4), int(self.cy + self.hrw - 4), stripe_w + stripe_gap):
            pygame.draw.rect(surface, color, (cx_right, y, crosswalk_depth, stripe_w))

    def _draw_road_turn_arrows(self, surface, color):
        """Draws subtle lane arrows painted on asphalt."""
        self._draw_arrow(surface, (self.cx + 1.5 * self.lw, self.cy - self.hrw - 70), 0, color)
        self._draw_arrow(surface, (self.cx + 0.5 * self.lw, self.cy - self.hrw - 70), 0, color)
        self._draw_arrow(surface, (self.cx - 1.5 * self.lw, self.cy + self.hrw + 70), math.pi, color)
        self._draw_arrow(surface, (self.cx - 0.5 * self.lw, self.cy + self.hrw + 70), math.pi, color)
        self._draw_arrow(surface, (self.cx - self.hrw - 70, self.cy + 1.5 * self.lw), math.pi / 2, color)
        self._draw_arrow(surface, (self.cx - self.hrw - 70, self.cy + 0.5 * self.lw), math.pi / 2, color)
        self._draw_arrow(surface, (self.cx + self.hrw + 70, self.cy - 1.5 * self.lw), -math.pi / 2, color)
        self._draw_arrow(surface, (self.cx + self.hrw + 70, self.cy - 0.5 * self.lw), -math.pi / 2, color)

    def _draw_arrow(self, surface, pos, angle, color):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tip = (pos[0] + cos_a * 10, pos[1] + sin_a * 10)
        base = (pos[0] - cos_a * 10, pos[1] - sin_a * 10)
        pygame.draw.line(surface, color, base, tip, 3)

    def render_traffic_lights(self, surface, traffic_controller, light_poles_dict, night_factor=0.0):
        """Draw 4 traffic light post enclosures with lit/unlit bulbs."""
        bulb_radius = 5
        box_w = 16
        box_h = 42

        for pole_dir, pos in light_poles_dict.items():
            state = traffic_controller.get_light_state(pole_dir)
            px, py = int(pos[0]), int(pos[1])

            # Pole stand
            pygame.draw.circle(surface, (60, 65, 75), (px, py), 4)

            # Signal housing box
            box_rect = pygame.Rect(px - box_w // 2, py - box_h // 2, box_w, box_h)
            pygame.draw.rect(surface, COLOR_TL_HOUSING, box_rect, border_radius=4)
            pygame.draw.rect(surface, (80, 85, 95), box_rect, width=1, border_radius=4)

            # Red bulb
            r_y = py - 12
            r_col = COLOR_TL_RED if state == 'RED' else (70, 15, 15)
            pygame.draw.circle(surface, r_col, (px, r_y), bulb_radius)

            # Yellow bulb
            y_y = py
            y_col = COLOR_TL_YELLOW if state == 'YELLOW' else (70, 60, 10)
            pygame.draw.circle(surface, y_col, (px, y_y), bulb_radius)

            # Green bulb
            g_y = py + 12
            g_col = COLOR_TL_GREEN if state == 'GREEN' else (10, 65, 25)
            pygame.draw.circle(surface, g_col, (px, g_y), bulb_radius)

    def render_sensor_rays(self, surface, vehicle):
        """Render LiDAR rays and perception targets of tracked vehicle."""
        if vehicle is None or not vehicle.is_alive or not vehicle.sensors.ray_hits:
            return

        if self.ray_surface is None or self.ray_surface.get_size() != surface.get_size():
            self.ray_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        self.ray_surface.fill((0, 0, 0, 0))

        front_x = vehicle.x + math.cos(vehicle.angle) * (vehicle.length / 2.0)
        front_y = vehicle.y + math.sin(vehicle.angle) * (vehicle.length / 2.0)
        origin = (front_x, front_y)

        for hit_pt, dist, hit_car in vehicle.sensors.ray_hits:
            norm_d = dist / vehicle.sensors.max_dist
            if norm_d < 0.25:
                ray_col = (255, 45, 45, 200) # Red hazard
            elif norm_d < 0.65:
                ray_col = (255, 210, 30, 170) # Yellow warning
            else:
                ray_col = (40, 240, 120, 130) # Green clear

            pygame.draw.line(self.ray_surface, ray_col, origin, hit_pt, 2)
            pygame.draw.circle(self.ray_surface, ray_col, (int(hit_pt[0]), int(hit_pt[1])), 3)

            if hit_car is not None:
                # Draw bounding box lock on detected obstacle car
                corners = hit_car.get_corners()
                pygame.draw.polygon(self.ray_surface, (255, 50, 50, 120), [(int(p[0]), int(p[1])) for p in corners], 2)

        surface.blit(self.ray_surface, (0, 0))
