"""
Particle system for realistic crash effects, sparks, smoke trails, tire skid marks, and explosion flashes.
"""
import random
import math
import pygame

class Particle:
    def __init__(self, x, y, vx, vy, color, size, life, decay, p_type="spark"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = list(color)
        self.size = size
        self.initial_size = size
        self.life = life          # Total lifetime frames
        self.max_life = life
        self.decay = decay
        self.p_type = p_type      # "spark", "smoke", "fire", "debris"
        self.rotation = random.uniform(0, 360)
        self.rot_speed = random.uniform(-10, 10)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.94
        self.vy *= 0.94
        self.life -= self.decay
        self.rotation += self.rot_speed

        if self.p_type == "smoke":
            self.size += 0.25 # Smoke expands
        elif self.p_type in ("spark", "fire"):
            self.size = max(0.5, self.initial_size * (self.life / self.max_life))

    @property
    def is_alive(self):
        return self.life > 0

    def draw(self, surface):
        if not self.is_alive or self.size <= 0:
            return
        alpha = int(255 * max(0.0, min(1.0, self.life / self.max_life)))
        
        if self.p_type == "smoke":
            # Soft grey/white smoke puff
            grey = self.color[0]
            s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, (grey, grey, grey, int(alpha * 0.45)), (int(self.size), int(self.size)), int(self.size))
            surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))
        elif self.p_type == "spark":
            # Sharp bright orange/yellow spark
            col = (min(255, self.color[0]), min(255, self.color[1]), min(255, self.color[2]), alpha)
            s = pygame.Surface((int(self.size * 2 + 2), int(self.size * 2 + 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, col, (int(self.size + 1), int(self.size + 1)), int(self.size))
            surface.blit(s, (int(self.x - self.size - 1), int(self.y - self.size - 1)))
        elif self.p_type == "fire":
            # Glowing core
            s = pygame.Surface((int(self.size * 3), int(self.size * 3)), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 140, 20, int(alpha * 0.7)), (int(self.size * 1.5), int(self.size * 1.5)), int(self.size * 1.5))
            pygame.draw.circle(s, (255, 240, 100, int(alpha * 0.9)), (int(self.size * 1.5), int(self.size * 1.5)), int(self.size * 0.8))
            surface.blit(s, (int(self.x - self.size * 1.5), int(self.y - self.size * 1.5)))
        elif self.p_type == "debris":
            # Small metal fragment
            col = (self.color[0], self.color[1], self.color[2], alpha)
            s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.rect(s, col, (0, 0, int(self.size), int(self.size * 1.5)))
            rot_s = pygame.transform.rotate(s, self.rotation)
            rect = rot_s.get_rect(center=(int(self.x), int(self.y)))
            surface.blit(rot_s, rect.topleft)


class SkidMark:
    def __init__(self, x1, y1, x2, y2, opacity=140):
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        self.opacity = opacity
        self.life = 600 # lasts ~10 seconds

    def update(self):
        self.life -= 1

    @property
    def is_alive(self):
        return self.life > 0


class ParticleManager:
    def __init__(self):
        self.particles = []
        self.skid_marks = []
        self.skid_surface = None

    def add_skid(self, x1, y1, x2, y2):
        if len(self.skid_marks) > 200:
            self.skid_marks.pop(0)
        self.skid_marks.append(SkidMark(x1, y1, x2, y2))

    def emit_crash(self, x, y, intensity=1.0):
        # Fire explosion puff
        num_fire = int(12 * intensity)
        for _ in range(num_fire):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.5, 6.0) * intensity
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(4, 10)
            self.particles.append(Particle(x, y, vx, vy, (255, 120, 20), size, life=25, decay=1.0, p_type="fire"))

        # Bright sparks flying out
        num_sparks = int(35 * intensity)
        for _ in range(num_sparks):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3.0, 10.0) * intensity
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(1.5, 3.5)
            life = random.uniform(15, 35)
            self.particles.append(Particle(x, y, vx, vy, (255, 230, 80), size, life=life, decay=1.0, p_type="spark"))

        # Billowing smoke
        num_smoke = int(18 * intensity)
        for _ in range(num_smoke):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - random.uniform(0.2, 0.8) # upward drift
            size = random.uniform(6, 14)
            grey = random.randint(50, 120)
            life = random.uniform(40, 80)
            self.particles.append(Particle(x, y, vx, vy, (grey, grey, grey), size, life=life, decay=1.0, p_type="smoke"))

        # Debris chunks
        num_debris = int(8 * intensity)
        for _ in range(num_debris):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.0, 7.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(2, 5)
            life = random.uniform(50, 100)
            self.particles.append(Particle(x, y, vx, vy, (80, 80, 85), size, life=life, decay=1.0, p_type="debris"))

    def emit_smoke_puff(self, x, y):
        # Light exhaust / tire smoke
        vx = random.uniform(-0.5, 0.5)
        vy = random.uniform(-0.5, 0.5)
        self.particles.append(Particle(x, y, vx, vy, (180, 180, 180), size=3.0, life=20, decay=1.0, p_type="smoke"))

    def update(self):
        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.is_alive]

        # Update skid marks
        for sm in self.skid_marks:
            sm.update()
        self.skid_marks = [sm for sm in self.skid_marks if sm.is_alive]

    def draw_skids(self, surface):
        if not self.skid_marks:
            return

        if self.skid_surface is None or self.skid_surface.get_size() != surface.get_size():
            self.skid_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        self.skid_surface.fill((0, 0, 0, 0))
        for sm in self.skid_marks:
            alpha = int(sm.opacity * (sm.life / 600.0))
            if alpha > 0:
                pygame.draw.line(self.skid_surface, (15, 15, 18, alpha), sm.p1, sm.p2, 3)

        surface.blit(self.skid_surface, (0, 0))

    def draw_particles(self, surface):
        for p in self.particles:
            p.draw(surface)

    def clear(self):
        self.particles.clear()
        self.skid_marks.clear()
        if self.skid_surface is not None:
            self.skid_surface.fill((0, 0, 0, 0))
