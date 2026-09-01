"""
Dynamic 2D Day/Night Lighting Engine.
Renders real-time headlight projection cones, streetlight spotlights,
traffic light neon bloom, red brake light radiance, and ambient darkness masking.
"""
import math
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_TL_RED,
    COLOR_TL_YELLOW, COLOR_TL_GREEN
)

def create_light_cone_poly(cx, cy, angle, length=150.0, spread_angle=math.radians(38)):
    """
    Creates polygon points for vehicle headlight projective beam.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Origin near front bumper
    p0 = (cx, cy)

    left_a = angle - spread_angle / 2.0
    right_a = angle + spread_angle / 2.0

    p1 = (cx + math.cos(left_a) * length, cy + math.sin(left_a) * length)
    p2 = (cx + math.cos(right_a) * length, cy + math.sin(right_a) * length)

    return [p0, p1, p2]

def draw_soft_circle(surface, color, center, radius):
    """Draws multi-layered radial soft glow circle."""
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r <= 0:
        return
    # Use stepped alpha rings for smooth radial falloff
    r_step = max(2, r // 4)
    base_alpha = color[3] if len(color) > 3 else 255
    for cur_r in range(r, 0, -r_step):
        alpha_ratio = (1.0 - (cur_r / r)) ** 1.5
        alpha = int(base_alpha * alpha_ratio)
        if alpha > 0:
            c = (color[0], color[1], color[2], alpha)
            pygame.draw.circle(surface, c, (cx, cy), cur_r)


class LightingEngine:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.darkness_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Precomputed gradient glow textures for maximum 60+ FPS performance
        self.headlight_tex = self._create_headlight_texture(length=160, fov_deg=42)
        self.glow_circle_red = self._create_radial_glow_tex((255, 45, 45), radius=50)
        self.glow_circle_yellow = self._create_radial_glow_tex((255, 210, 40), radius=50)
        self.glow_circle_green = self._create_radial_glow_tex((40, 240, 110), radius=50)
        self.glow_circle_warm = self._create_radial_glow_tex((255, 235, 180), radius=90)
        self.glow_brake_red = self._create_radial_glow_tex((255, 20, 20), radius=35)

    def _create_radial_glow_tex(self, color, radius):
        size = radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = radius, radius
        for r in range(radius, 0, -2):
            ratio = (1.0 - (r / radius)) ** 1.4
            alpha = int(220 * ratio)
            c = (color[0], color[1], color[2], alpha)
            pygame.draw.circle(surf, c, (cx, cy), r)
        return surf

    def _create_headlight_texture(self, length=160, fov_deg=42):
        # Create a single high-quality beam cone surface
        w = int(length * math.tan(math.radians(fov_deg / 2)) * 2) + 20
        h = length + 20
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        origin = (w // 2, 5)
        # Stepped cones from outer faint to inner intense
        fov_rad = math.radians(fov_deg)
        steps = 5
        for i in range(steps):
            ratio = (i + 1) / steps
            cur_fov = fov_rad * (1.1 - 0.5 * ratio)
            cur_len = length * (0.4 + 0.6 * ratio)
            alpha = int(45 + 160 * ratio)
            p1 = (origin[0] - math.sin(cur_fov / 2) * cur_len, origin[1] + math.cos(cur_fov / 2) * cur_len)
            p2 = (origin[0] + math.sin(cur_fov / 2) * cur_len, origin[1] + math.cos(cur_fov / 2) * cur_len)
            c = (255, 255, 225, alpha)
            pygame.draw.polygon(surf, c, [origin, p1, p2])

        return surf

    def render_lighting(self, target_surface, vehicles, traffic_lights_dict, light_pole_pos, particles, night_factor=0.0):
        """
        Renders ambient darkness and dynamic lights onto target_surface.
        night_factor ranges from 0.0 (bright day) to 1.0 (deep midnight).
        """
        if night_factor <= 0.02:
            return # Pure day mode, no night lighting overlay needed

        # 1. Fill darkness layer
        ambient_darkness = int(225 * night_factor) # Max 225 alpha darkness
        self.darkness_surface.fill((10, 14, 22, ambient_darkness))

        bloom_ops = [] # Store (surface, rect) to blit after darkness

        # 2. Streetlights at 4 corners
        street_light_alpha = int(140 * night_factor)
        for pole_name, pos in light_pole_pos.items():
            r_glow = self.glow_circle_warm
            rect = r_glow.get_rect(center=(int(pos[0]), int(pos[1])))
            self.darkness_surface.blit(r_glow, rect, special_flags=pygame.BLEND_RGBA_SUB)

        # 3. Traffic Light Neon Bloom
        for pole_name, state in traffic_lights_dict.items():
            pos = light_pole_pos.get(pole_name)
            if pos is None:
                continue

            if state == 'RED':
                color_surf = self.glow_circle_red
            elif state == 'YELLOW':
                color_surf = self.glow_circle_yellow
            else:
                color_surf = self.glow_circle_green

            c_rect = color_surf.get_rect(center=(int(pos[0]), int(pos[1])))
            self.darkness_surface.blit(color_surf, c_rect, special_flags=pygame.BLEND_RGBA_SUB)
            bloom_ops.append((color_surf, c_rect))

        # 4. Vehicle Lights
        for car in vehicles:
            if not car.is_alive:
                continue

            # Headlights
            cos_a = math.cos(car.angle)
            sin_a = math.sin(car.angle)
            hl = car.length / 2

            front_cx = car.x + hl * cos_a
            front_cy = car.y + hl * sin_a

            if night_factor > 0:
                angle_deg = int(-math.degrees(car.angle) - 90) % 360
                
                if not hasattr(self, '_hl_cache'):
                    self._hl_cache = {}
                if angle_deg not in self._hl_cache:
                    self._hl_cache[angle_deg] = pygame.transform.rotate(self.headlight_tex, angle_deg)
                
                rotated_hl = self._hl_cache[angle_deg]
                
                hl_dist = rotated_hl.get_width() * 0.35
                shift_x = front_cx + cos_a * hl_dist
                shift_y = front_cy + sin_a * hl_dist
                
                hl_rect = rotated_hl.get_rect(center=(int(shift_x), int(shift_y)))
                self.darkness_surface.blit(rotated_hl, hl_rect, special_flags=pygame.BLEND_RGBA_SUB)
                bloom_ops.append((rotated_hl, hl_rect))

            # Brake light rear glow
            if car.is_braking:
                rear_cx = car.x - hl * cos_a
                rear_cy = car.y - hl * sin_a
                b_rect = self.glow_brake_red.get_rect(center=(int(rear_cx), int(rear_cy)))
                self.darkness_surface.blit(self.glow_brake_red, b_rect, special_flags=pygame.BLEND_RGBA_SUB)

        # 5. Particle explosion / fire glow in night
        for p in particles:
            if p.p_type == "fire":
                fg = self.glow_circle_yellow
                rect = fg.get_rect(center=(int(p.x), int(p.y)))
                self.darkness_surface.blit(fg, rect, special_flags=pygame.BLEND_RGBA_SUB)
                bloom_ops.append((fg, rect))

        # 6. Blit darkness mask onto main screen
        target_surface.blit(self.darkness_surface, (0, 0))

        # 7. Blit additive bloom for glowing neon lights
        for b_surf, b_rect in bloom_ops:
            target_surface.blit(b_surf, b_rect, special_flags=pygame.BLEND_RGBA_ADD)
