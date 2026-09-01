"""
Dynamic Weather Engine for Autonomous Crossroad Simulation.
Simulates rain particle streaks, ground splash ripples, dynamic road wetness,
wind drift, and physical tire friction grip reduction.
"""
import math
import random
import pygame
from src.config import SCREEN_WIDTH, SCREEN_HEIGHT

class RainDrop:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.x = random.uniform(-100, self.width + 100)
        self.y = random.uniform(-60, -10)
        self.speed = random.uniform(12.0, 18.0)
        self.length = random.uniform(10.0, 18.0)
        self.alpha = random.randint(140, 220)

    def update(self, wind_x=2.5):
        self.x += wind_x
        self.y += self.speed
        if self.y > self.height:
            # Chance to spawn ground splash
            splash_pt = (self.x, self.y)
            self.reset()
            return splash_pt
        return None

    def draw(self, surface, wind_x=2.5):
        angle = math.atan2(self.speed, wind_x)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        p1 = (int(self.x), int(self.y))
        p2 = (int(self.x + cos_a * self.length), int(self.y + sin_a * self.length))
        pygame.draw.line(surface, (190, 215, 245, self.alpha), p1, p2, 1)


class SplashRipple:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.radius = 1.0
        self.max_radius = random.uniform(3.0, 6.0)
        self.life = 1.0

    def update(self):
        self.radius += 0.45
        self.life -= 0.12

    @property
    def is_alive(self):
        return self.life > 0

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha = int(180 * self.life)
        s = pygame.Surface((int(self.radius * 4 + 4), int(self.radius * 2 + 4)), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (200, 225, 255, alpha), (2, 2, int(self.radius * 2), int(self.radius)))
        surface.blit(s, (self.x - int(self.radius), self.y - int(self.radius / 2)))


class Puddle:
    def __init__(self):
        from src.config import CENTER_X, CENTER_Y
        self.x = random.uniform(CENTER_X - 250, CENTER_X + 250)
        self.y = random.uniform(CENTER_Y - 200, CENTER_Y + 200)
        self.radius = random.uniform(15, 35)
        self.life = 0.0
        self.max_life = random.uniform(5.0, 15.0)
        self.active = True

    def update(self, dt):
        self.life += dt
        if self.life > self.max_life:
            self.active = False

    def draw(self, surface):
        alpha = int(120 * math.sin((self.life / self.max_life) * math.pi))
        alpha = max(0, min(150, alpha))
        if alpha > 0:
            s = pygame.Surface((int(self.radius * 2), int(self.radius)), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (30, 70, 120, alpha), (0, 0, int(self.radius * 2), int(self.radius)))
            surface.blit(s, (self.x - self.radius, self.y - self.radius / 2))

class WeatherManager:
    def __init__(self):
        self.weather_mode = 'CLEAR' # 'CLEAR', 'RAIN', 'STORM'
        self.wetness = 0.0          # 0.0 (bone dry) to 1.0 (soaked)
        self.target_wetness = 0.0
        self.friction_coeff = 1.0   # 1.0 = normal dry grip, 0.55 = slippery wet asphalt
        self.wind_x = 2.0

        self.raindrops = [RainDrop() for _ in range(250)]
        self.splashes = []
        self.puddles = []
        self.puddle_timer = 0.0
        self.rain_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    def set_mode(self, mode):
        self.weather_mode = mode
        if mode == 'CLEAR':
            self.target_wetness = 0.0
            self.friction_coeff = 1.0
            self.wind_x = 0.5
        elif mode == 'RAIN':
            self.target_wetness = 0.7
            self.friction_coeff = 0.68
            self.wind_x = 2.5
        elif mode == 'STORM':
            self.target_wetness = 1.0
            self.friction_coeff = 0.52
            self.wind_x = 4.5

    def toggle_weather(self):
        modes = ['CLEAR', 'RAIN', 'STORM']
        cur_idx = modes.index(self.weather_mode)
        next_mode = modes[(cur_idx + 1) % len(modes)]
        self.set_mode(next_mode)
        return next_mode

    def update(self, dt=1.0/60.0):
        # Smooth wetness transition
        self.wetness += (self.target_wetness - self.wetness) * (0.04 * dt * 60.0)

        if self.weather_mode in ('RAIN', 'STORM'):
            active_drops = 140 if self.weather_mode == 'RAIN' else 250
            for i in range(active_drops):
                splash_pt = self.raindrops[i].update(self.wind_x)
                if splash_pt and len(self.splashes) < 60 and random.random() < 0.35:
                    self.splashes.append(SplashRipple(splash_pt[0], splash_pt[1]))

        # Update splash ripples
        for s in self.splashes:
            s.update()
        self.splashes = [s for s in self.splashes if s.is_alive]

        # Puddles
        if self.weather_mode == 'STORM' and self.wetness > 0.8:
            self.puddle_timer += dt
            if self.puddle_timer > 2.0 and len(self.puddles) < 5:
                self.puddles.append(Puddle())
                self.puddle_timer = 0.0
        
        for p in self.puddles:
            p.update(dt)
        self.puddles = [p for p in self.puddles if p.active]

    def draw(self, surface):
        if self.weather_mode == 'CLEAR' and self.wetness < 0.05 and not self.splashes:
            return

        # Wet road atmospheric sheen
        sheen_alpha = int(40 * self.wetness) if self.wetness > 0.1 else 0
        self.rain_surface.fill((30, 60, 100, sheen_alpha))

        # Draw splash ripples
        for s in self.splashes:
            s.draw(self.rain_surface)

        # Draw Puddles
        for p in self.puddles:
            p.draw(self.rain_surface)

        # Draw falling raindrops
        if self.weather_mode in ('RAIN', 'STORM'):
            active_drops = 140 if self.weather_mode == 'RAIN' else 250
            for i in range(active_drops):
                self.raindrops[i].draw(self.rain_surface, self.wind_x)

        surface.blit(self.rain_surface, (0, 0))
