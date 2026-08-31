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
        self.light_cutout_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.bloom_surface = pygame.Surface((width, height), pygame.SRCALPHA)

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

        # Clear light accumulator
        self.light_cutout_surface.fill((0, 0, 0, 0))
        self.bloom_surface.fill((0, 0, 0, 0))

        # 2. Streetlights at 4 corners
        street_light_alpha = int(140 * night_factor)
        for pole_name, pos in light_pole_pos.items():
            r_glow = self.glow_circle_warm
            rect = r_glow.get_rect(center=(int(pos[0]), int(pos[1])))
            self.light_cutout_surface.blit(r_glow, rect, special_flags=pygame.BLEND_RGBA_ADD)

        # 3. Traffic Light Neon Bloom
        for pole_name, state in traffic_lights_dict.items():
            pos = light_pole_pos.get(pole_name)
            if pos is None:
                continue

            glow_tex = None
            if state == 'RED':
                glow_tex = self.glow_circle_red
            elif state == 'YELLOW':
                glow_tex = self.glow_circle_yellow
            elif state == 'GREEN':
                glow_tex = self.glow_circle_green

            if glow_tex is not None:
                rect = glow_tex.get_rect(center=(int(pos[0]), int(pos[1])))
                # Strong additive bloom
                self.bloom_surface.blit(glow_tex, rect, special_flags=pygame.BLEND_RGBA_ADD)
                self.light_cutout_surface.blit(glow_tex, rect, special_flags=pygame.BLEND_RGBA_ADD)

        # 4. Vehicle Headlights & Brake Lights
        for car in vehicles:
            if not car.is_alive and not car.has_crashed:
                continue

            cos_a = math.cos(car.angle)
            sin_a = math.sin(car.angle)
            hl = car.length / 2.0
            hw = car.width / 2.0 - 2.5

            # Headlights
            front_cx = car.x + hl * cos_a
            front_cy = car.y + hl * sin_a

            # Use pre-rendered headlight texture, scaled by night factor
            if night_factor > 0:
                # Car angle is in radians, 0 is right. Pygame rotate expects degrees, counter-clockwise.
                # In Pygame, 0 degrees is right, 90 is UP.
                # So we convert radians to degrees and negate.
                # But wait, our `headlight_tex` points DOWN.
                # A DOWN texture needs to be rotated so it points to `car.angle`.
                # DOWN is 90 degrees (or pi/2 radians) in Pygame's y-down coord system.
                # To point it at car.angle (where 0 is right), we subtract 90 degrees and negate.
                angle_deg = -math.degrees(car.angle) - 90
                
                rotated_hl = pygame.transform.rotate(self.headlight_tex, angle_deg)
                
                # Offset to place the tip of the cone exactly at the front bumper
                # The tip in original texture is at (w//2, 5).
                # After rotation, its position changes. We can just center it roughly, 
                # or better, just blit the center of the rotated rect to a shifted point.
                hl_dist = rotated_hl.get_width() * 0.35
                shift_x = front_cx + cos_a * hl_dist
                shift_y = front_cy + sin_a * hl_dist
                
                hl_rect = rotated_hl.get_rect(center=(int(shift_x), int(shift_y)))
                self.light_cutout_surface.blit(rotated_hl, hl_rect, special_flags=pygame.BLEND_RGBA_ADD)
                self.bloom_surface.blit(rotated_hl, hl_rect, special_flags=pygame.BLEND_RGBA_ADD)

            # Brake light rear glow
            if car.is_braking:
                back_cx = car.x - hl * cos_a
                back_cy = car.y - hl * sin_a
                b_rect = self.glow_brake_red.get_rect(center=(int(back_cx), int(back_cy)))
                self.bloom_surface.blit(self.glow_brake_red, b_rect, special_flags=pygame.BLEND_RGBA_ADD)
                self.light_cutout_surface.blit(self.glow_brake_red, b_rect, special_flags=pygame.BLEND_RGBA_ADD)

        # 5. Particle explosion / fire glow in night
        for p in particles:
            if p.p_type == "fire":
                fg = self.glow_circle_yellow
                rect = fg.get_rect(center=(int(p.x), int(p.y)))
                self.bloom_surface.blit(fg, rect, special_flags=pygame.BLEND_RGBA_ADD)
                self.light_cutout_surface.blit(fg, rect, special_flags=pygame.BLEND_RGBA_ADD)

        # 6. Carve illuminated light cutouts out of the darkness mask
        # Using BLEND_RGBA_SUB to subtract darkness where light is cast
        self.darkness_surface.blit(self.light_cutout_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        # 7. Blit darkness mask onto main screen
        target_surface.blit(self.darkness_surface, (0, 0))

        # 8. Blit additive bloom for glowing neon lights
        target_surface.blit(self.bloom_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
