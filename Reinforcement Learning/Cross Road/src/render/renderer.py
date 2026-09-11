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
    ROAD_WIDTH, LANE_WIDTH, NUM_LANES_PER_DIR, COLOR_BG, COLOR_GRASS_DAY,
    COLOR_GRASS_NIGHT, COLOR_ROAD_DAY, COLOR_ROAD_NIGHT,
    COLOR_ROAD_MARKING, COLOR_ROAD_YELLOW, COLOR_SIDEWALK_DAY,
    COLOR_SIDEWALK_NIGHT, COLOR_STOP_LINE, COLOR_TL_RED,
    COLOR_TL_YELLOW, COLOR_TL_GREEN, COLOR_TL_HOUSING,
    COLOR_ISLAND_DAY, COLOR_ISLAND_NIGHT
)
from src.render.cityscape import paint_parks_and_city, draw_roundabout_plaza, draw_extruded_curb, paint_trees

def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors by factor t in [0, 1]."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )

class Renderer:
    def __init__(self, screen, cx=None, cy=None, width=None, height=None, topnav_h=48, sidebar_w=340, city=None):
        self.screen = screen
        self.width = width or SCREEN_WIDTH
        self.height = height or SCREEN_HEIGHT
        self.topnav_h = topnav_h
        self.sidebar_w = sidebar_w
        self.city = city

        if city is not None:
            self.cx, self.cy = city.cx, city.cy
        elif cx is None or cy is None:
            canvas_w = self.width - self.sidebar_w
            canvas_h = self.height - self.topnav_h
            self.cx = canvas_w // 2
            self.cy = self.topnav_h + (canvas_h // 2)
        else:
            self.cx = cx
            self.cy = cy

        self.rw = ROAD_WIDTH
        self.hrw = ROAD_WIDTH / 2.0
        self.lw = LANE_WIDTH
        self.ray_surface = None

        self.bg_day_surface = pygame.Surface((self.width, self.height))
        self.bg_night_surface = pygame.Surface((self.width, self.height))
        self._build_static_background(self.bg_day_surface, night_factor=0.0)
        self._build_static_background(self.bg_night_surface, night_factor=1.0)

    def update_dimensions(self, screen, cx, cy, width, height, topnav_h=48, sidebar_w=340, city=None):
        self.screen = screen
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.topnav_h = topnav_h
        self.sidebar_w = sidebar_w
        if city is not None:
            self.city = city
            self.cx, self.cy = city.cx, city.cy
        self.bg_day_surface = pygame.Surface((self.width, self.height))
        self.bg_night_surface = pygame.Surface((self.width, self.height))
        self._build_static_background(self.bg_day_surface, night_factor=0.0)
        self._build_static_background(self.bg_night_surface, night_factor=1.0)

    def _city_params(self):
        city = self.city
        rbx = getattr(city, 'rbx', self.cx) if city else self.cx
        rby = getattr(city, 'rby', self.cy + 220) if city else self.cy + 220
        r_outer = getattr(city, 'r_outer', 102)
        r_island = getattr(city, 'r_island', 30)
        r_inner = getattr(city, 'r_inner', 48)
        r_circ = getattr(city, 'r_circ', 74)
        return rbx, rby, r_outer, r_island, r_inner, r_circ

    def _build_static_background(self, surface, night_factor=0.0):
        canvas_right = self.width - self.sidebar_w
        canvas_top = self.topnav_h
        canvas_bottom = self.height
        rbx, rby, r_outer, r_island, r_inner, r_circ = self._city_params()

        grass_col = lerp_color(COLOR_GRASS_DAY, COLOR_GRASS_NIGHT, night_factor)
        surface.fill(grass_col)
        night = night_factor > 0.5

        paint_parks_and_city(
            surface, self.cx, self.cy, rbx, rby, self.hrw, r_outer, r_island,
            canvas_right, canvas_top, canvas_bottom, night
        )

        sw_col = lerp_color(COLOR_SIDEWALK_DAY, COLOR_SIDEWALK_NIGHT, night_factor)
        sw_offset = self.hrw + 16
        ns_rect = (self.cx - sw_offset, canvas_top, sw_offset * 2, canvas_bottom - canvas_top)
        ew_rect = (0, self.cy - sw_offset, canvas_right, sw_offset * 2)
        bv_rect = (0, rby - sw_offset, canvas_right, sw_offset * 2)
        pygame.draw.rect(surface, sw_col, ns_rect)
        pygame.draw.rect(surface, sw_col, ew_rect)
        pygame.draw.rect(surface, sw_col, bv_rect)
        pygame.draw.circle(surface, sw_col, (int(rbx), int(rby)), int(r_outer + 16))
        draw_extruded_curb(surface, ns_rect, night)
        draw_extruded_curb(surface, ew_rect, night)
        draw_extruded_curb(surface, bv_rect, night)

        road_col = lerp_color(COLOR_ROAD_DAY, COLOR_ROAD_NIGHT, night_factor)
        pygame.draw.rect(surface, road_col, (self.cx - self.hrw, canvas_top, self.rw, canvas_bottom - canvas_top))
        pygame.draw.rect(surface, road_col, (0, self.cy - self.hrw, canvas_right, self.rw))
        pygame.draw.rect(surface, road_col, (0, rby - self.hrw, canvas_right, self.rw))
        draw_roundabout_plaza(surface, rbx, rby, r_island, r_outer, night, r_inner, r_circ)

        random.seed(7)
        for _ in range(700):
            rx = random.randint(0, max(1, canvas_right - 1))
            ry = random.randint(canvas_top, max(canvas_top + 1, canvas_bottom - 1))
            grain_val = random.randint(-10, 10)
            grain_c = (
                max(0, min(255, road_col[0] + grain_val)),
                max(0, min(255, road_col[1] + grain_val)),
                max(0, min(255, road_col[2] + grain_val))
            )
            on_road = (
                abs(rx - self.cx) <= self.hrw or
                abs(ry - self.cy) <= self.hrw or
                abs(ry - rby) <= self.hrw or
                (r_island < math.hypot(rx - rbx, ry - rby) <= r_outer)
            )
            if on_road:
                surface.set_at((rx, ry), grain_c)

        marking_col = COLOR_ROAD_MARKING
        self._draw_zebra_crossings(surface, marking_col)

        stop_line_thick = 4
        stop_offset = 36
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx - self.hrw, self.cy - self.hrw - stop_offset), (self.cx, self.cy - self.hrw - stop_offset), stop_line_thick)
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx, self.cy + self.hrw + stop_offset), (self.cx + self.hrw, self.cy + self.hrw + stop_offset), stop_line_thick)
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx + self.hrw + stop_offset, self.cy - self.hrw), (self.cx + self.hrw + stop_offset, self.cy), stop_line_thick)
        pygame.draw.line(surface, COLOR_STOP_LINE, (self.cx - self.hrw - stop_offset, self.cy), (self.cx - self.hrw - stop_offset, self.cy + self.hrw), stop_line_thick)

        yellow_col = COLOR_ROAD_YELLOW
        self._draw_double_yellow_line(surface, (self.cx, canvas_top), (self.cx, self.cy - self.hrw - stop_offset), yellow_col)
        self._draw_double_yellow_line(surface, (self.cx, self.cy + self.hrw + stop_offset), (self.cx, rby - r_outer - 8), yellow_col)
        self._draw_double_yellow_line(surface, (self.cx, rby + r_outer + 8), (self.cx, canvas_bottom), yellow_col)
        self._draw_double_yellow_line_h(surface, (0, self.cy), (self.cx - self.hrw - stop_offset, self.cy), yellow_col)
        self._draw_double_yellow_line_h(surface, (self.cx + self.hrw + stop_offset, self.cy), (canvas_right, self.cy), yellow_col)
        self._draw_double_yellow_line_h(surface, (0, rby), (rbx - r_outer - 8, rby), yellow_col)
        self._draw_double_yellow_line_h(surface, (rbx + r_outer + 8, rby), (canvas_right, rby), yellow_col)

        self._draw_dashed_lane_markings(surface, marking_col, canvas_top, canvas_bottom, canvas_right, stop_offset)
        self._draw_road_turn_arrows(surface, marking_col)
        paint_trees(
            surface, self.cx, self.cy, rbx, rby, self.hrw, r_outer, r_island,
            canvas_right, canvas_top, canvas_bottom, night
        )

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

    def _draw_dashed_lane_markings(self, surface, color, canvas_top, canvas_bottom, canvas_right, stop_offset=36):
        dash_len = 16
        dash_gap = 14
        margin = stop_offset + 4
        rbx, rby, r_outer, _, _, _ = self._city_params()
        lane_xs = [self.cx + o * self.lw for o in (-2, -1, 1, 2)]
        lane_ys_main = [self.cy + o * self.lw for o in (-2, -1, 1, 2)]
        lane_ys_blvd = [rby + o * self.lw for o in (-2, -1, 1, 2)]

        def vdash(x, y0, y1):
            y = y0
            while y < y1:
                pygame.draw.line(surface, color, (x, y), (x, min(y1, y + dash_len)), 2)
                y += dash_len + dash_gap

        def hdash(y, x0, x1):
            x = x0
            while x < x1:
                pygame.draw.line(surface, color, (x, y), (min(x1, x + dash_len), y), 2)
                x += dash_len + dash_gap

        for x in lane_xs:
            vdash(x, canvas_top, self.cy - self.hrw - margin)
            vdash(x, self.cy + self.hrw + margin, rby - r_outer - 8)
            vdash(x, rby + r_outer + 8, canvas_bottom)
        for y in lane_ys_main:
            hdash(y, 0, self.cx - self.hrw - margin)
            hdash(y, self.cx + self.hrw + margin, canvas_right)
        for y in lane_ys_blvd:
            hdash(y, 0, rbx - r_outer - 8)
            hdash(y, rbx + r_outer + 8, canvas_right)

    def _draw_zebra_crossings(self, surface, color):
        stripe_w = 7
        stripe_gap = 5
        crosswalk_depth = 24
        cw_margin = 6
        shadow = (32, 34, 38)

        def bars_h(x0, x1, y, h):
            for x in range(int(x0), int(x1), stripe_w + stripe_gap):
                pygame.draw.rect(surface, shadow, (x, y + 2, stripe_w, h))
                pygame.draw.rect(surface, color, (x, y, stripe_w, h))

        def bars_v(y0, y1, x, w):
            for y in range(int(y0), int(y1), stripe_w + stripe_gap):
                pygame.draw.rect(surface, shadow, (x + 2, y, w, stripe_w))
                pygame.draw.rect(surface, color, (x, y, w, stripe_w))

        bars_h(self.cx - self.hrw + 4, self.cx + self.hrw - 4,
               self.cy - self.hrw - cw_margin - crosswalk_depth, crosswalk_depth)
        bars_h(self.cx - self.hrw + 4, self.cx + self.hrw - 4,
               self.cy + self.hrw + cw_margin, crosswalk_depth)
        bars_v(self.cy - self.hrw + 4, self.cy + self.hrw - 4,
               self.cx - self.hrw - cw_margin - crosswalk_depth, crosswalk_depth)
        bars_v(self.cy - self.hrw + 4, self.cy + self.hrw - 4,
               self.cx + self.hrw + cw_margin, crosswalk_depth)

    def _draw_road_turn_arrows(self, surface, color):
        for k in (0.5, 1.5, 2.5):
            self._draw_arrow(surface, (self.cx - k * self.lw, self.cy - self.hrw - 70), math.pi / 2, color)
            self._draw_arrow(surface, (self.cx + k * self.lw, self.cy + self.hrw + 70), -math.pi / 2, color)
            self._draw_arrow(surface, (self.cx - self.hrw - 70, self.cy + k * self.lw), 0, color)
            self._draw_arrow(surface, (self.cx + self.hrw + 70, self.cy - k * self.lw), math.pi, color)

    def _draw_arrow(self, surface, pos, angle, color):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tip = (pos[0] + cos_a * 10, pos[1] + sin_a * 10)
        base = (pos[0] - cos_a * 10, pos[1] - sin_a * 10)
        pygame.draw.line(surface, color, base, tip, 3)

    def render_traffic_lights(self, surface, traffic_controller, light_poles_dict, night_factor=0.0):
        """Draw 3-lamp signal heads on extruded posts."""
        bulb_radius = 5
        box_w = 18
        box_h = 46

        for pole_dir, pos in light_poles_dict.items():
            state = traffic_controller.get_light_state(pole_dir)
            px, py = int(pos[0]), int(pos[1])

            pygame.draw.rect(surface, (18, 20, 24), (px - 3, py, 8, 18))
            pygame.draw.rect(surface, (70, 74, 82), (px - 2, py, 4, 16))
            pygame.draw.circle(surface, (90, 94, 102), (px, py + 16), 4)

            box_rect = pygame.Rect(px - box_w // 2, py - box_h // 2 - 6, box_w, box_h)
            pygame.draw.rect(surface, (16, 16, 18), (box_rect.x + 3, box_rect.y + 3, box_w, box_h), border_radius=6)
            pygame.draw.rect(surface, COLOR_TL_HOUSING, box_rect, border_radius=6)
            pygame.draw.rect(surface, (92, 96, 104), box_rect, width=1, border_radius=6)

            r_col = COLOR_TL_RED if state == 'RED' else (48, 16, 16)
            y_col = COLOR_TL_YELLOW if state == 'YELLOW' else (48, 40, 12)
            g_col = COLOR_TL_GREEN if state == 'GREEN' else (12, 44, 22)
            pygame.draw.circle(surface, r_col, (px, py - 18), bulb_radius)
            pygame.draw.circle(surface, y_col, (px, py - 6), bulb_radius)
            pygame.draw.circle(surface, g_col, (px, py + 6), bulb_radius)
            if state == 'RED':
                pygame.draw.circle(surface, (255, 180, 180), (px - 2, py - 20), 2)
            elif state == 'GREEN':
                pygame.draw.circle(surface, (180, 255, 210), (px - 2, py + 4), 2)

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
