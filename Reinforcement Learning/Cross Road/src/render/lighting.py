"""
Dynamic 2D Day/Night Lighting Engine.
Uses a smooth BLEND_MULT lightmap architecture for realistic illumination:
warm headlight cones, streetlights, neon traffic light blooms, and rear brake lights.
"""
import math
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_TL_RED,
    COLOR_TL_YELLOW, COLOR_TL_GREEN
)

class LightingEngine:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.lightmap = pygame.Surface((width, height))

        # Precomputed radial soft-light textures for 60+ FPS performance
        self.headlight_tex = self._create_soft_headlight_texture(length=140, fov_deg=38)
        self.glow_streetlight = self._create_radial_glow((160, 150, 110), radius=75)
        self.glow_tl_red = self._create_radial_glow((180, 40, 40), radius=45)
        self.glow_tl_yellow = self._create_radial_glow((180, 150, 30), radius=45)
        self.glow_tl_green = self._create_radial_glow((30, 180, 90), radius=45)
        self.glow_brake_red = self._create_radial_glow((170, 30, 30), radius=30)
        self._hl_cache = {}

    def update_dimensions(self, width, height):
        self.width = width
        self.height = height
        self.lightmap = pygame.Surface((width, height))

    def _create_radial_glow(self, color, radius):
        """Creates a smooth radial falloff glow surface for additive lightmap blending."""
        size = radius * 2
        surf = pygame.Surface((size, size))
        surf.fill((0, 0, 0))
        cx, cy = radius, radius
        for r in range(radius, 0, -2):
            ratio = (1.0 - (r / radius)) ** 1.6
            c = (int(color[0] * ratio), int(color[1] * ratio), int(color[2] * ratio))
            pygame.draw.circle(surf, c, (cx, cy), r)
        return surf

    def _create_soft_headlight_texture(self, length=140, fov_deg=38):
        """Creates an illuminated headlight beam texture that fades outward smoothly."""
        w = int(length * math.tan(math.radians(fov_deg / 2)) * 2) + 30
        h = length + 30
        surf = pygame.Surface((w, h))
        surf.fill((0, 0, 0))

        origin = (w // 2, 8)
        fov_rad = math.radians(fov_deg)
        steps = 10
        for i in range(steps):
            ratio = (i + 1) / float(steps)
            cur_fov = fov_rad * (1.15 - 0.45 * ratio)
            cur_len = length * (0.3 + 0.7 * ratio)
            intensity = (1.0 - (ratio - 0.1)**2) * 0.7
            c = (int(180 * intensity), int(175 * intensity), int(150 * intensity))
            
            p1 = (origin[0] - math.sin(cur_fov / 2) * cur_len, origin[1] + math.cos(cur_fov / 2) * cur_len)
            p2 = (origin[0] + math.sin(cur_fov / 2) * cur_len, origin[1] + math.cos(cur_fov / 2) * cur_len)
            pygame.draw.polygon(surf, c, [origin, p1, p2])

        # Soft apex glow at origin
        pygame.draw.circle(surf, (160, 160, 140), origin, 8)
        return surf

    def render_lighting(self, target_surface, vehicles, traffic_lights_dict, light_pole_pos, particles, night_factor=0.0):
        """
        Renders ambient darkness and dynamic lights onto target_surface.
        night_factor ranges from 0.0 (bright day) to 1.0 (deep midnight).
        """
        if night_factor <= 0.02:
            return  # Pure daylight, no lighting pass needed

        # 1. Base ambient night color (day: 255, night: deep cool dark tone)
        # Deep midnight ambient: RGB (48, 55, 75)
        amb_r = int(255 - (255 - 48) * night_factor)
        amb_g = int(255 - (255 - 55) * night_factor)
        amb_b = int(255 - (255 - 75) * night_factor)
        self.lightmap.fill((amb_r, amb_g, amb_b))

        # 2. Additive streetlights at intersection corners
        for pole_name, pos in light_pole_pos.items():
            rect = self.glow_streetlight.get_rect(center=(int(pos[0]), int(pos[1])))
            self.lightmap.blit(self.glow_streetlight, rect, special_flags=pygame.BLEND_RGB_ADD)

        # 3. Additive Traffic Light Glow
        for pole_name, state in traffic_lights_dict.items():
            pos = light_pole_pos.get(pole_name)
            if pos is None:
                continue

            if state == 'RED':
                color_surf = self.glow_tl_red
            elif state == 'YELLOW':
                color_surf = self.glow_tl_yellow
            else:
                color_surf = self.glow_tl_green

            c_rect = color_surf.get_rect(center=(int(pos[0]), int(pos[1])))
            self.lightmap.blit(color_surf, c_rect, special_flags=pygame.BLEND_RGB_ADD)

        # 4. Vehicle Headlights and Brake Lights
        for car in vehicles:
            if not car.is_alive:
                continue

            cos_a = math.cos(car.angle)
            sin_a = math.sin(car.angle)
            hl = car.length / 2.0

            front_cx = car.x + hl * cos_a
            front_cy = car.y + hl * sin_a

            # Headlight projection
            angle_deg = int(-math.degrees(car.angle) - 90) % 360
            if angle_deg not in self._hl_cache:
                self._hl_cache[angle_deg] = pygame.transform.rotate(self.headlight_tex, angle_deg)

            rotated_hl = self._hl_cache[angle_deg]
            shift_dist = rotated_hl.get_width() * 0.32
            shift_x = front_cx + cos_a * shift_dist
            shift_y = front_cy + sin_a * shift_dist
            hl_rect = rotated_hl.get_rect(center=(int(shift_x), int(shift_y)))

            self.lightmap.blit(rotated_hl, hl_rect, special_flags=pygame.BLEND_RGB_ADD)

            # Brake light rear glow
            if car.is_braking:
                rear_cx = car.x - hl * cos_a
                rear_cy = car.y - hl * sin_a
                b_rect = self.glow_brake_red.get_rect(center=(int(rear_cx), int(rear_cy)))
                self.lightmap.blit(self.glow_brake_red, b_rect, special_flags=pygame.BLEND_RGB_ADD)

        # 5. Particle explosion / fire glow in night
        for p in particles:
            if getattr(p, 'p_type', '') == "fire":
                rect = self.glow_tl_yellow.get_rect(center=(int(p.x), int(p.y)))
                self.lightmap.blit(self.glow_tl_yellow, rect, special_flags=pygame.BLEND_RGB_ADD)

        # 6. Apply lightmap to world via hardware BLEND_MULT
        target_surface.blit(self.lightmap, (0, 0), special_flags=pygame.BLEND_MULT)
