"""
Windows 11 Fluent Design UI Engine.
Includes:
- Acrylic Frosted Glass Top Navigation Bar (KPIs, Reports, Speed & Mode Controls)
- Fluent Sidebar with Card Layout, High-Contrast Typography, and Interactive Controls
- Real-time Anti-aliased Telemetry Trend Graphs
- Cyber-Aesthetic AI Brain Inspector with Live Neural Activations & Synaptic Pulses
"""
import math
from collections import deque
import numpy as np
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, ACTIONS_MAP
)

# Municipal ops console — copper signal + mint health, not cyan-on-navy.
OPS_BG = (10, 14, 22)
OPS_SIDE = (12, 16, 26)
OPS_CARD = (18, 24, 36)
OPS_CARD_EDGE = (48, 56, 72)
OPS_HOVER = (28, 36, 52)
OPS_BTN = (22, 28, 42)
OPS_ACCENT = (232, 165, 75)
OPS_MINT = (61, 214, 151)
OPS_ROSE = (255, 92, 110)
OPS_SKY = (120, 168, 255)
OPS_TEXT = (244, 246, 250)
OPS_MUTED = (148, 158, 176)
OPS_DIM = (96, 106, 124)


class FluentButton:
    def __init__(self, rect, text, callback, active_fn=None, icon=None, tooltip=""):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.active_fn = active_fn
        self.icon = icon
        self.tooltip = tooltip
        self.is_hovered = False
        self._cache = {}

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface, font):
        is_active = self.active_fn() if self.active_fn else False
        cache_key = (is_active, self.is_hovered, self.rect.size)

        if cache_key not in self._cache:
            btn_surf = pygame.Surface(self.rect.size) # 100% Solid Opaque
            w, h = self.rect.size

            if is_active:
                btn_surf.fill(OPS_ACCENT)
                pygame.draw.rect(btn_surf, (255, 214, 150), (0, 0, w, h), width=1, border_radius=8)
                txt_col = (20, 16, 10)
            elif self.is_hovered:
                btn_surf.fill(OPS_HOVER)
                pygame.draw.rect(btn_surf, (90, 102, 128), (0, 0, w, h), width=1, border_radius=8)
                txt_col = OPS_TEXT
            else:
                btn_surf.fill(OPS_BTN)
                pygame.draw.rect(btn_surf, OPS_CARD_EDGE, (0, 0, w, h), width=1, border_radius=8)
                txt_col = OPS_MUTED

            full_txt = f"{self.icon} {self.text}" if self.icon else self.text
            txt_surf = font.render(full_txt, True, txt_col)
            txt_rect = txt_surf.get_rect(center=(w // 2, h // 2))
            btn_surf.blit(txt_surf, txt_rect)
            self._cache[cache_key] = btn_surf

        surface.blit(self._cache[cache_key], self.rect.topleft)


class UIHud:
    def __init__(self, sim, width=None, height=None, topnav_h=48, sidebar_w=340):
        self.sim = sim
        self.width = width or SCREEN_WIDTH
        self.height = height or SCREEN_HEIGHT
        self.sidebar_w = sidebar_w
        self.sidebar_x = self.width - self.sidebar_w
        self.topnav_h = topnav_h

        # History for telemetry sparkline
        self.history_success = deque(maxlen=120)
        self.chart_tick = 0

        # Typography (Segoe UI for Windows 11 native look)
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)
        self.font_h1 = pygame.font.SysFont("Segoe UI, Arial", 12, bold=True)
        self.font_main = pygame.font.SysFont("Segoe UI, Arial", 12)
        self.font_bold = pygame.font.SysFont("Segoe UI, Arial", 12, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial", 11)
        self.font_mono = pygame.font.SysFont("Consolas, Courier New", 11)
        self.font_badge = pygame.font.SysFont("Segoe UI, Arial", 10, bold=True)

        self.buttons = []
        self._init_buttons()

    def update_dimensions(self, width, height, topnav_h=48, sidebar_w=340):
        self.width = width
        self.height = height
        self.topnav_h = topnav_h
        self.sidebar_w = sidebar_w
        self.sidebar_x = self.width - self.sidebar_w
        self._init_buttons()

    def _init_buttons(self):
        self.buttons.clear()

        # ==========================================
        # TOP NAVBAR CONTROLS (Quick Mode & Speed)
        # ==========================================
        mode_y = 10
        btn_h = 28
        self.buttons.append(FluentButton(
            (self.sidebar_x - 300, mode_y, 90, btn_h), "Untrained",
            lambda: self.sim.set_ai_mode('UNTRAINED'),
            active_fn=lambda: self.sim.agent.mode == 'UNTRAINED'
        ))
        self.buttons.append(FluentButton(
            (self.sidebar_x - 205, mode_y, 90, btn_h), "Live Train",
            lambda: self.sim.set_ai_mode('TRAINING'),
            active_fn=lambda: self.sim.agent.mode == 'TRAINING'
        ))
        self.buttons.append(FluentButton(
            (self.sidebar_x - 110, mode_y, 95, btn_h), "Master AI",
            lambda: self.sim.set_ai_mode('MASTER'),
            active_fn=lambda: self.sim.agent.mode == 'MASTER'
        ))

        # ==========================================
        # SIDEBAR CARD 1: ENVIRONMENT & TIME
        # ==========================================
        sx = self.sidebar_x + 16
        c1_w = self.sidebar_w - 32

        # Time of Day (Day / Sunset / Night / Auto)
        y = 75
        tw = (c1_w - 9) // 4
        self.buttons.append(FluentButton(
            (sx, y, tw, 26), "Day",
            lambda: self.sim.set_day_night(0.0),
            active_fn=lambda: not self.sim.auto_day_night and self.sim.night_factor < 0.2
        ))
        self.buttons.append(FluentButton(
            (sx + tw + 3, y, tw, 26), "Sunset",
            lambda: self.sim.set_day_night(0.5),
            active_fn=lambda: not self.sim.auto_day_night and 0.2 <= self.sim.night_factor <= 0.8
        ))
        self.buttons.append(FluentButton(
            (sx + (tw + 3)*2, y, tw, 26), "Night",
            lambda: self.sim.set_day_night(1.0),
            active_fn=lambda: not self.sim.auto_day_night and self.sim.night_factor > 0.8
        ))
        self.buttons.append(FluentButton(
            (sx + (tw + 3)*3, y, tw, 26), "Auto",
            lambda: self.sim.toggle_auto_day_night(),
            active_fn=lambda: self.sim.auto_day_night
        ))

        # Weather Presets
        y = 106
        ww = (c1_w - 6) // 3
        self.buttons.append(FluentButton(
            (sx, y, ww, 26), "Clear",
            lambda: self.sim.weather.set_weather('CLEAR'),
            active_fn=lambda: self.sim.weather.weather_mode == 'CLEAR'
        ))
        self.buttons.append(FluentButton(
            (sx + ww + 3, y, ww, 26), "Rain",
            lambda: self.sim.weather.set_weather('RAIN'),
            active_fn=lambda: self.sim.weather.weather_mode == 'RAIN'
        ))
        self.buttons.append(FluentButton(
            (sx + (ww + 3)*2, y, ww, 26), "Storm",
            lambda: self.sim.weather.set_weather('STORM'),
            active_fn=lambda: self.sim.weather.weather_mode == 'STORM'
        ))

        # Simulation Speed
        y = 137
        sw = (c1_w - 9) // 4
        self.buttons.append(FluentButton(
            (sx, y, sw, 26), "Pause",
            lambda: self.sim.set_speed(0.0),
            active_fn=lambda: self.sim.sim_speed == 0.0
        ))
        self.buttons.append(FluentButton(
            (sx + sw + 3, y, sw, 26), "1x",
            lambda: self.sim.set_speed(1.0),
            active_fn=lambda: self.sim.sim_speed == 1.0
        ))
        self.buttons.append(FluentButton(
            (sx + (sw + 3)*2, y, sw, 26), "2x",
            lambda: self.sim.set_speed(2.0),
            active_fn=lambda: self.sim.sim_speed == 2.0
        ))
        self.buttons.append(FluentButton(
            (sx + (sw + 3)*3, y, sw, 26), "5x",
            lambda: self.sim.set_speed(5.0),
            active_fn=lambda: self.sim.sim_speed == 5.0
        ))

        # ==========================================
        # SIDEBAR CARD 2: ACTION TOOLBAR
        # ==========================================
        y = 330
        bw = (c1_w - 3) // 2
        self.buttons.append(FluentButton(
            (sx, y, bw, 26), "Manual Phase (T)",
            lambda: self.sim.traffic_controller.switch_manual_phase()
        ))
        self.buttons.append(FluentButton(
            (sx + bw + 3, y, bw, 26), "Adaptive Sig",
            lambda: self.sim.toggle_adaptive_lights(),
            active_fn=lambda: getattr(self.sim, 'adaptive_lights_enabled', True)
        ))

        y = 360
        self.buttons.append(FluentButton(
            (sx, y, bw, 26), "Spawn Car (S)",
            lambda: self.sim.spawn_random_vehicle()
        ))
        self.buttons.append(FluentButton(
            (sx + bw + 3, y, bw, 26), "Ambulance (A)",
            lambda: self.sim.spawn_ambulance()
        ))

        y = 390
        self.buttons.append(FluentButton(
            (sx, y, bw, 26), "Fire (F)",
            lambda: self.sim.spawn_fire()
        ))
        self.buttons.append(FluentButton(
            (sx + bw + 3, y, bw, 26), "Police (P)",
            lambda: self.sim.spawn_police()
        ))

        y = 420
        self.buttons.append(FluentButton(
            (sx, y, bw, 26), "LiDAR Rays (V)",
            lambda: self.sim.toggle_vision_overlay(),
            active_fn=lambda: self.sim.show_vision_overlay
        ))

        self.buttons.append(FluentButton(
            (sx + bw + 3, y, bw, 26), "Reset Sim (R)",
            lambda: self.sim.reset_simulation()
        ))

    def handle_event(self, event):
        for btn in self.buttons:
            if btn.handle_event(event):
                return True
        return False

    def draw(self, surface):
        # 1. Render Solid Opaque Top Navigation Bar (No glass!)
        self._draw_topnav(surface)

        # 2. Render Solid Opaque Sidebar (No glass!)
        self._draw_sidebar(surface)

    def _draw_topnav(self, surface):
        nav_w = self.width
        # 100% Solid Opaque background - NO transparency/glass
        pygame.draw.rect(surface, OPS_BG, (0, 0, nav_w, self.topnav_h))
        pygame.draw.line(surface, OPS_ACCENT, (0, self.topnav_h - 2), (nav_w, self.topnav_h - 2), 2)

        # App Logo & Title
        title_txt = self.font_title.render("CROSS ROAD OPS", True, OPS_TEXT)
        surface.blit(title_txt, (18, 14))

        # Status Pill Badge
        mode = self.sim.agent.mode
        mode_bg = OPS_MINT if mode == 'MASTER' else (OPS_SKY if mode == 'TRAINING' else OPS_ROSE)
        badge_txt = self.font_badge.render(f"● {mode}", True, (20, 24, 34))
        bw = badge_txt.get_width() + 14
        badge_rect = pygame.Rect(title_txt.get_width() + 28, 13, bw, 22)
        pygame.draw.rect(surface, mode_bg, badge_rect, border_radius=11)
        surface.blit(badge_txt, (title_txt.get_width() + 35, 17))

        # Live KPI Chips in Center of Navbar
        fps = int(self.sim.clock.get_fps())
        active_cars = len([c for c in self.sim.vehicles if c.is_alive])
        tot_passed = self.sim.stats['total_passed']
        tot_crashes = self.sim.stats['total_crashes']
        tot_spawned = self.sim.stats['total_spawned']
        rate = (tot_passed / tot_spawned * 100.0) if tot_spawned > 0 else 0.0

        t_secs = int(self.sim.training_time)
        t_str = f"{t_secs // 60:02d}:{t_secs % 60:02d}"

        em_count = sum(1 for c in self.sim.vehicles if c.is_alive and getattr(c, 'is_emergency', False))
        chips = [
            (f"FPS: {fps}", OPS_MINT if fps >= 45 else OPS_ROSE),
            (f"Cars: {active_cars}/{self.sim.max_active_cars}", OPS_SKY),
            (f"EMS: {em_count}", OPS_ROSE if em_count else OPS_MUTED),
            (f"Passed: {tot_passed}", OPS_TEXT),
            (f"Crashes: {tot_crashes}", OPS_ROSE if tot_crashes > 0 else OPS_MUTED),
            (f"Iter: #{self.sim.agent.train_step_count}", OPS_SKY),
            (f"Ep: #{self.sim.training_episodes}", OPS_MINT),
            (f"Train: {t_str}", OPS_ACCENT),
            (f"Success: {rate:.1f}%", OPS_MINT if rate >= 80 else OPS_ACCENT),
        ]

        chip_x = title_txt.get_width() + bw + 45
        for text, col in chips:
            c_surf = self.font_small.render(text, True, col)
            cw = c_surf.get_width() + 16
            c_rect = pygame.Rect(chip_x, 13, cw, 22)
            pygame.draw.rect(surface, (22, 28, 40), c_rect, border_radius=8)
            pygame.draw.rect(surface, OPS_CARD_EDGE, c_rect, width=1, border_radius=8)
            surface.blit(c_surf, (chip_x + 8, 17))
            chip_x += cw + 8

        # Draw Navbar Mode Buttons (first 3 buttons)
        for btn in self.buttons[:3]:
            btn.draw(surface, self.font_small)

    def _draw_sidebar(self, surface):
        # 100% Solid Opaque background - NO transparency/glass
        pygame.draw.rect(surface, OPS_SIDE, (self.sidebar_x, 0, self.sidebar_w, self.height))
        pygame.draw.line(surface, OPS_ACCENT, (self.sidebar_x, 0), (self.sidebar_x, self.height), 2)

        # Section 1: Environment & Speed Control Card
        c_w = self.sidebar_w - 32
        sx = self.sidebar_x + 16
        c1_rect = pygame.Rect(sx, 56, c_w, 116)
        pygame.draw.rect(surface, OPS_CARD, c1_rect, border_radius=8)
        pygame.draw.rect(surface, OPS_CARD_EDGE, c1_rect, width=1, border_radius=10)
        surface.blit(self.font_h1.render("WORLD  LIGHT  SPEED", True, OPS_ACCENT), (sx + 10, 61))

        # Section 2: Real-Time Telemetry & Success Rate Trend Card
        c2_rect = pygame.Rect(sx, 180, c_w, 144)
        pygame.draw.rect(surface, OPS_CARD, c2_rect, border_radius=8)
        pygame.draw.rect(surface, OPS_CARD_EDGE, c2_rect, width=1, border_radius=10)
        surface.blit(self.font_h1.render("TELEMETRY", True, OPS_MINT), (sx + 10, 185))

        # Telemetry metric lines
        tot_spawned = self.sim.stats['total_spawned']
        tot_passed = self.sim.stats['total_passed']
        rate = (tot_passed / tot_spawned * 100.0) if tot_spawned > 0 else 0.0

        self.chart_tick += 1
        if self.chart_tick % 10 == 0:
            self.history_success.append(rate)
        t_secs = int(self.sim.training_time)
        t_str = f"{t_secs // 60:02d}:{t_secs % 60:02d}"
        lr = getattr(self.sim.agent, 'current_lr', 0.0003)
        grad_norm = getattr(self.sim.agent, 'last_grad_norm', 0.0)
        clip_n = getattr(self.sim.agent, 'grad_clip_norm', 1.0)
        opt_status = getattr(self.sim.agent, 'optimizer_status', 'Optimal')

        surface.blit(self.font_small.render(f"Train: Ep #{self.sim.training_episodes} | Iter: #{self.sim.agent.train_step_count} | Time: {t_str}", True, OPS_MINT), (sx + 12, 198))
        surface.blit(self.font_small.render(f"Auto-Tuned LR: {lr:.2e} | Status: {opt_status}", True, OPS_SKY), (sx + 12, 213))
        surface.blit(self.font_small.render(f"Grad: {grad_norm:.2f} | Clip: {clip_n:.2f} | Loss: {self.sim.agent.avg_loss:.4f}", True, OPS_MUTED), (sx + 12, 228))

        # Trend Chart
        self._draw_trend_chart(surface, sx + 10, 246, c_w - 20, 68)

        # Draw interactive control buttons (all sidebar buttons)
        for btn in self.buttons[3:]:
            btn.draw(surface, self.font_small)

        # Section 3: AI Brain Inspector Card & Section 4: Creators Footer
        footer_h = 62
        footer_y = self.height - footer_h - 8
        brain_h = max(100, footer_y - 454 - 8)

        self._draw_brain_inspector(surface, sx, 454, c_w, brain_h)
        self._draw_footer(surface, sx, footer_y, c_w, footer_h)

    def _draw_footer(self, surface, x, y, w, h):
        c_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, (18, 22, 32), c_rect, border_radius=8)
        pygame.draw.rect(surface, OPS_CARD_EDGE, c_rect, width=1, border_radius=10)
        surface.blit(self.font_badge.render("BUILT BY", True, OPS_ACCENT), (x + 10, y + 6))
        surface.blit(self.font_bold.render("1. mohammadrezamirtaleb", True, (240, 245, 255)), (x + 10, y + 23))
        surface.blit(self.font_bold.render("2. mahdiajami", True, (240, 245, 255)), (x + 10, y + 41))

    def _draw_trend_chart(self, surface, x, y, w, h):
        pygame.draw.rect(surface, (16, 20, 28), (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, (55, 65, 85), (x, y, w, h), width=1, border_radius=6)

        # Subtle grid lines
        for r in [0.25, 0.5, 0.75]:
            gy = y + int(h * r)
            pygame.draw.line(surface, (28, 34, 46), (x, gy), (x + w, gy), 1)

        if len(self.history_success) > 1:
            pts = []
            dx = w / max(1, len(self.history_success) - 1)
            for i, val in enumerate(self.history_success):
                norm_v = max(0.0, min(1.0, val / 100.0))
                py = y + h - (norm_v * (h - 14)) - 6
                pts.append((x + i * dx, py))

            if len(pts) >= 2:
                fill = [(pts[0][0], y + h - 2)] + pts + [(pts[-1][0], y + h - 2)]
                pygame.draw.polygon(surface, (32, 72, 58), fill)
                pygame.draw.lines(surface, OPS_MINT, False, pts, 2)
                last_pt = pts[-1]
                pygame.draw.circle(surface, OPS_TEXT, (int(last_pt[0]), int(last_pt[1])), 3)

    def _draw_brain_inspector(self, surface, x, y, w, h):
        c3_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, OPS_CARD, c3_rect, border_radius=8)
        pygame.draw.rect(surface, OPS_CARD_EDGE, c3_rect, width=1, border_radius=10)

        car = self.sim.selected_vehicle
        if car is None or not car.is_alive:
            active = [c for c in self.sim.vehicles if c.is_alive]
            car = active[0] if active else None

        if car is None or car.last_state is None:
            surface.blit(self.font_h1.render("AI BRAIN INSPECTOR", True, OPS_SKY), (x + 10, y + 8))
            surface.blit(self.font_small.render("No active vehicle selected", True, OPS_DIM), (x + 70, y + h // 2))
            return

        # Header with Vehicle ID & Class
        v_title = f"AI BRAIN — {car.v_type} #{car.vehicle_id}"
        surface.blit(self.font_h1.render(v_title, True, OPS_SKY), (x + 10, y + 8))

        # Dynamic vehicle metrics
        spd_kmh = car.speed * 10.0
        dist_s = car.get_distance_to_stop_line()
        stop_txt = f"{dist_s:.0f}px" if dist_s is not None else "Passed"
        live_ttc = getattr(car, 'live_ttc', car.sensors.min_ttc)
        ttc_txt = f"{live_ttc:.1f}s" if live_ttc < 10.0 else "Clear"
        telemetry = (f"Speed: {spd_kmh:.1f} km/h | Stop: {stop_txt} | "
                     f"TTC: {ttc_txt} | Reward: {car.total_reward:+.1f}")
        surface.blit(self.font_small.render(telemetry, True, OPS_MUTED), (x + 10, y + 26))

        # Q-Values horizontal distribution
        surface.blit(self.font_badge.render("DECISION Q-VALUES & PROBABILITIES", True, OPS_DIM), (x + 10, y + 44))

        q_vals = self.sim.agent.get_q_values(car.last_state)
        q_labels = ["COAST", "ACCEL 1", "ACCEL 2", "BRAKE 1", "BRAKE 2"]
        chosen = car.last_action

        bar_x = x + 75
        bar_max_w = w - 145
        min_q = min(-1.0, float(np.min(q_vals)))
        max_q = max(1.0, float(np.max(q_vals)))
        q_rng = max(0.1, max_q - min_q)

        for i, (lbl, qv) in enumerate(zip(q_labels, q_vals)):
            by = y + 58 + i * 16
            is_ch = (i == chosen)
            lbl_col = OPS_SKY if is_ch else OPS_DIM
            surface.blit(self.font_mono.render(f"{lbl:7s}", True, lbl_col), (x + 10, by))

            # Bar track
            pygame.draw.rect(surface, (16, 20, 28), (bar_x, by + 2, bar_max_w, 8), border_radius=4)
            norm_w = max(0.04, min(1.0, (qv - min_q) / q_rng))
            fill_w = int(norm_w * bar_max_w)
            fill_col = OPS_MINT if is_ch else OPS_SKY
            pygame.draw.rect(surface, fill_col, (bar_x, by + 2, fill_w, 8), border_radius=4)

            # Q value text
            v_col = OPS_TEXT if is_ch else OPS_MUTED
            surface.blit(self.font_mono.render(f"{qv:+.1f}", True, v_col), (bar_x + bar_max_w + 8, by - 1))

        # Neural Network Layer Activation Diagram (Expanded 6-Stage Deep Architecture)
        ny = y + 144
        surface.blit(self.font_badge.render("SITUATION-GATED DUELING DQN", True, OPS_DIM), (x + 10, ny))

        activations = self.sim.agent.get_activations(car.last_state)
        step_x = (w - 36) // 5
        layer_cols = [
            (x + 18, activations.get('inputs', [])[:8], "Sensors"),
            (x + 18 + step_x, activations.get('h1', [])[:8], "LiDAR"),
            (x + 18 + step_x * 2, activations.get('h2', [])[:8], "Situation"),
            (x + 18 + step_x * 3, activations.get('h3', [])[:8], "Fuse"),
            (x + 18 + step_x * 4, activations.get('stream', [])[:8], "Val/Adv"),
            (x + 18 + step_x * 5, activations.get('q_values', []), "Q-Out")
        ]

        node_y_start = ny + 20
        for l_idx, (cx, vals, lname) in enumerate(layer_cols):
            surface.blit(self.font_badge.render(lname, True, OPS_DIM), (cx - 14, node_y_start - 6))
            num_nodes = len(vals)
            spacing = 11
            for n_idx, val in enumerate(vals):
                ny_pos = node_y_start + 12 + n_idx * spacing
                intensity = max(0.15, min(1.0, float(abs(val))))
                node_col = (
                    int(232 * intensity),
                    int(165 * intensity),
                    int(75 * intensity)
                )
                pygame.draw.circle(surface, node_col, (cx, int(ny_pos)), 3)

                # Synaptic connections to next layer
                if l_idx < len(layer_cols) - 1:
                    next_cx = layer_cols[l_idx + 1][0]
                    next_vals = layer_cols[l_idx + 1][1]
                    for next_n in range(min(3, len(next_vals))):
                        next_ny = node_y_start + 12 + next_n * spacing
                        alpha = int(85 * intensity)
                        syn_col = (0, 160, 230, alpha)
                        pygame.draw.line(surface, syn_col[:3], (cx + 3, int(ny_pos)), (next_cx - 3, int(next_ny)), 1)
