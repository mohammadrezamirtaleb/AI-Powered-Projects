"""
Cyber-Aesthetic UI HUD, Interactive Controls, Telemetry Dashboard, Real-time Line Charts, Weather Controls, and Neural Network Visualizer.
Renders real-time Q-network layer activations, action bar charts, statistics, and interactive buttons.
"""
import math
from collections import deque
import numpy as np
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, UI_PANEL_BG, UI_PANEL_BORDER,
    UI_ACCENT_CYAN, UI_ACCENT_GREEN, UI_ACCENT_ORANGE, UI_ACCENT_RED,
    UI_TEXT_WHITE, UI_TEXT_MUTED, ACTIONS_MAP
)

class Button:
    def __init__(self, rect, text, callback, tag="", active_fn=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.tag = tag
        self.active_fn = active_fn
        self.is_hovered = False

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

        state_key = (is_active, self.is_hovered)
        if not hasattr(self, '_cache'):
            self._cache = {}
            
        if state_key not in self._cache:
            if is_active:
                bg_col = (0, 120, 212, 255) # Windows 11 Accent Blue
                border_col = (255, 255, 255, 20)
                text_col = (255, 255, 255)
            elif self.is_hovered:
                bg_col = (255, 255, 255, 20) # Subtle white hover
                border_col = (255, 255, 255, 15)
                text_col = (255, 255, 255)
            else:
                bg_col = (255, 255, 255, 10) # Rest state acrylic
                border_col = (255, 255, 255, 10)
                text_col = (230, 230, 230)

            btn_surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            pygame.draw.rect(btn_surf, bg_col, btn_surf.get_rect(), border_radius=4)
            pygame.draw.rect(btn_surf, border_col, btn_surf.get_rect(), width=1, border_radius=4)
            
            txt_surf = font.render(self.text, True, text_col)
            txt_rect = txt_surf.get_rect(center=btn_surf.get_rect().center)
            btn_surf.blit(txt_surf, txt_rect)
            self._cache[state_key] = btn_surf

        surface.blit(self._cache[state_key], self.rect.topleft)


class UIHud:
    def __init__(self, sim):
        self.sim = sim
        self.hud_x = SCREEN_WIDTH - 360
        self.hud_width = 360
        self.hud_rect = pygame.Rect(self.hud_x, 0, self.hud_width, SCREEN_HEIGHT)

        # Real-time Telemetry History for live charts
        self.history_success = deque(maxlen=100)
        self.history_loss = deque(maxlen=100)
        self.chart_tick = 0

        # Fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Segoe UI, Arial", 16, bold=True)
        self.font_main = pygame.font.SysFont("Segoe UI, Arial", 13)
        self.font_bold = pygame.font.SysFont("Segoe UI, Arial", 13, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial", 11)
        self.font_mono = pygame.font.SysFont("Consolas, Courier New", 12)

        self.buttons = []
        self._init_buttons()

    def _init_buttons(self):
        bx = self.hud_x + 15
        bh = 24

        # Row 1: Day / Night mode buttons
        y = 62
        self.buttons.append(Button(
            (bx, y, 70, bh), "Day",
            lambda: self.sim.set_day_night(0.0),
            active_fn=lambda: self.sim.night_factor < 0.2
        ))
        self.buttons.append(Button(
            (bx + 75, y, 75, bh), "Sunset",
            lambda: self.sim.set_day_night(0.5),
            active_fn=lambda: 0.3 <= self.sim.night_factor <= 0.7
        ))
        self.buttons.append(Button(
            (bx + 155, y, 75, bh), "Night",
            lambda: self.sim.set_day_night(1.0),
            active_fn=lambda: self.sim.night_factor > 0.8
        ))
        self.buttons.append(Button(
            (bx + 235, y, 90, bh), "Auto Cycle",
            lambda: self.sim.toggle_auto_day_night(),
            active_fn=lambda: self.sim.auto_day_night
        ))

        # Row 2: Dynamic Weather System buttons
        y = 90
        self.buttons.append(Button(
            (bx, y, 104, bh), "Dry Clear",
            lambda: self.sim.weather.set_mode('CLEAR'),
            active_fn=lambda: self.sim.weather.weather_mode == 'CLEAR'
        ))
        self.buttons.append(Button(
            (bx + 112, y, 104, bh), "Rain (Wet)",
            lambda: self.sim.weather.set_mode('RAIN'),
            active_fn=lambda: self.sim.weather.weather_mode == 'RAIN'
        ))
        self.buttons.append(Button(
            (bx + 224, y, 101, bh), "Storm",
            lambda: self.sim.weather.set_mode('STORM'),
            active_fn=lambda: self.sim.weather.weather_mode == 'STORM'
        ))

        # Row 3: Speed Multipliers
        y = 135
        speeds = [(0.0, "Pause"), (1.0, "1x"), (2.0, "2x"), (5.0, "5x"), (10.0, "10x")]
        sw = 60
        for i, (spd, label) in enumerate(speeds):
            self.buttons.append(Button(
                (bx + i * (sw + 6), y, sw, bh), label,
                (lambda s=spd: lambda: self.sim.set_sim_speed(s))(),
                active_fn=(lambda s=spd: lambda: self.sim.sim_speed == s)()
            ))

        # Row 4: AI Mode Buttons
        y = 180
        mw = 104
        self.buttons.append(Button(
            (bx, y, mw, bh + 4), "Chaos Mode",
            lambda: self.sim.set_ai_mode('UNTRAINED'),
            active_fn=lambda: self.sim.agent.mode == 'UNTRAINED'
        ))
        self.buttons.append(Button(
            (bx + mw + 8, y, mw, bh + 4), "Live Train",
            lambda: self.sim.set_ai_mode('TRAINING'),
            active_fn=lambda: self.sim.agent.mode == 'TRAINING'
        ))
        self.buttons.append(Button(
            (bx + (mw + 8) * 2, y, mw, bh + 4), "Master AI",
            lambda: self.sim.set_ai_mode('MASTER'),
            active_fn=lambda: self.sim.agent.mode == 'MASTER'
        ))

        # Row 5: Traffic light & Adaptive Signals
        y = 405
        self.buttons.append(Button(
            (bx, y, 155, bh), "Switch Light (T)",
            lambda: self.sim.traffic_controller.switch_manual_phase(),
            active_fn=lambda: self.sim.traffic_controller.is_manual
        ))
        self.buttons.append(Button(
            (bx + 165, y, 160, bh), "Adaptive Signals",
            lambda: self.sim.traffic_controller.toggle_adaptive(),
            active_fn=lambda: self.sim.traffic_controller.adaptive_mode
        ))

        # Row 6: Vehicle Spawners
        y = 433
        self.buttons.append(Button(
            (bx, y, 155, bh), "Spawn Car (S)",
            lambda: self.sim.spawn_random_vehicle()
        ))
        self.buttons.append(Button(
            (bx + 165, y, 160, bh), "Spawn Ambulance",
            lambda: self.sim.spawn_ambulance()
        ))

        # Row 7: Vision Rays, Cinematic, Reset
        y = 461
        self.buttons.append(Button(
            (bx, y, 105, bh), "Vision (V)",
            lambda: self.sim.toggle_vision_overlay(),
            active_fn=lambda: self.sim.show_vision_rays
        ))
        self.buttons.append(Button(
            (bx + 110, y, 105, bh), "Cinematic",
            lambda: self.sim.toggle_cinematic(),
            active_fn=lambda: getattr(self.sim, 'cinematic_mode', False)
        ))
        self.buttons.append(Button(
            (bx + 220, y, 105, bh), "Reset (R)",
            lambda: self.sim.reset_statistics()
        ))
        
        # Row 8: V2V, Jaywalkers, Save AI
        y = 489
        self.buttons.append(Button(
            (bx, y, 105, bh), "V2V Sync",
            lambda: self.sim.toggle_v2v(),
            active_fn=lambda: getattr(self.sim, 'v2v_enabled', False)
        ))
        self.buttons.append(Button(
            (bx + 110, y, 105, bh), "Jaywalkers",
            lambda: self.sim.toggle_jaywalking(),
            active_fn=lambda: getattr(self.sim, 'jaywalking_enabled', False)
        ))
        self.buttons.append(Button(
            (bx + 220, y, 105, bh), "Save AI",
            lambda: self.sim.save_model()
        ))

    def handle_event(self, event):
        for btn in self.buttons:
            if btn.handle_event(event):
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if mx < self.hud_x:
                clicked_car = None
                for car in self.sim.vehicles:
                    if car.is_alive and math.hypot(car.x - mx, car.y - my) < 25.0:
                        clicked_car = car
                        break
                self.sim.selected_vehicle = clicked_car
        return False

    def _update_history(self):
        self.chart_tick += 1
        if self.chart_tick % 10 == 0:
            tot_spawned = self.sim.stats['total_spawned']
            tot_passed = self.sim.stats['total_passed']
            rate = (tot_passed / tot_spawned * 100.0) if tot_spawned > 0 else 0.0
            self.history_success.append(rate)
            self.history_loss.append(self.sim.agent.avg_loss)

    def draw(self, surface):
        self._update_history()

        if not hasattr(self, 'bg_surf'):
            self.bg_surf = pygame.Surface((self.hud_width, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(self.bg_surf, (12, 16, 24, 215), (0, 0, self.hud_width, SCREEN_HEIGHT))
            left_edge = pygame.Surface((4, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(left_edge, (0, 200, 255, 150), left_edge.get_rect())
            self.bg_surf.blit(left_edge, (0, 0))
            pygame.draw.line(self.bg_surf, (0, 150, 200, 100), (4, 0), (4, SCREEN_HEIGHT), 1)

            self.txt_title = self.font_title.render("⚡ AUTONOMOUS CROSSROAD AI", True, UI_ACCENT_CYAN)
            self.txt_h1 = self.font_bold.render("LIGHTING & DYNAMIC WEATHER", True, UI_TEXT_MUTED)
            self.txt_h2 = self.font_bold.render("SIMULATION SPEED", True, UI_TEXT_MUTED)
            self.txt_h3 = self.font_bold.render("AI OPERATING MODE", True, UI_TEXT_MUTED)
            self.txt_h4 = self.font_bold.render("REAL-TIME TELEMETRY & CHARTS", True, UI_TEXT_MUTED)

        surface.blit(self.bg_surf, (self.hud_x, 0))
        surface.blit(self.txt_title, (self.hud_x + 15, 10))

        cur_mode = self.sim.agent.mode
        mode_color = UI_ACCENT_RED if cur_mode == 'UNTRAINED' else (UI_ACCENT_GREEN if cur_mode == 'MASTER' else UI_ACCENT_ORANGE)
        mode_txt = self.font_small.render(f"STATUS: {cur_mode} MODE", True, mode_color)
        surface.blit(mode_txt, (self.hud_x + 15, 30))

        # 3. Section Labels
        surface.blit(self.txt_h1, (self.hud_x + 15, 46))
        surface.blit(self.txt_h2, (self.hud_x + 15, 120))
        surface.blit(self.txt_h3, (self.hud_x + 15, 164))
        surface.blit(self.txt_h4, (self.hud_x + 15, 214))

        # 4. Telemetry & Statistics Card
        card_rect = pygame.Rect(self.hud_x + 15, 230, self.hud_width - 30, 165)
        pygame.draw.rect(surface, (22, 28, 38), card_rect, border_radius=6)
        pygame.draw.rect(surface, UI_PANEL_BORDER, card_rect, width=1, border_radius=6)

        sx = self.hud_x + 25
        sy = 234
        line_h = 16

        fps = int(self.sim.clock.get_fps())
        tot_spawned = self.sim.stats['total_spawned']
        tot_passed = self.sim.stats['total_passed']
        tot_crashes = self.sim.stats['total_crashes']
        success_rate = (tot_passed / tot_spawned * 100.0) if tot_spawned > 0 else 0.0

        stats_data = [
            ("FPS & Active Cars:", f"{fps} FPS | {len(self.sim.vehicles)} Cars", UI_TEXT_WHITE),
            ("Successful Trips:", f"{tot_passed} passed", UI_ACCENT_GREEN),
            ("Total Collisions:", f"{tot_crashes} crashes", UI_ACCENT_RED if tot_crashes > 0 else UI_TEXT_WHITE),
            ("Success Rate:", f"{success_rate:.1f} %", UI_ACCENT_GREEN if success_rate > 80 else UI_ACCENT_ORANGE),
            ("Weather & Grip:", f"{self.sim.weather.weather_mode} (μ={self.sim.weather.friction_coeff:.2f})", UI_ACCENT_CYAN),
            ("DQN Epsilon (ε):", f"{self.sim.agent.epsilon:.3f}", UI_TEXT_MUTED),
        ]

        for i, (label, val, col) in enumerate(stats_data):
            surface.blit(self.font_small.render(label, True, UI_TEXT_MUTED), (sx, sy + i * line_h))
            surface.blit(self.font_bold.render(val, True, col), (sx + 145, sy + i * line_h))

        # Mini Sparkline Chart
        chart_x = self.hud_x + 25
        chart_y = sy + len(stats_data) * line_h + 6
        chart_w = self.hud_width - 50
        chart_h = 50
        self._draw_sparkline_chart(surface, chart_x, chart_y, chart_w, chart_h)

        # 5. Draw Buttons
        for btn in self.buttons:
            btn.draw(surface, self.font_main)

        # 6. Selected Vehicle & Neural Network HUD
        self._draw_neural_net_hud(surface)

    def _draw_sparkline_chart(self, surface, x, y, w, h):
        pygame.draw.rect(surface, (15, 20, 28), (x, y, w, h), border_radius=4)
        pygame.draw.rect(surface, (35, 45, 60), (x, y, w, h), width=1, border_radius=4)

        for gy in [y + h * 0.25, y + h * 0.5, y + h * 0.75]:
            pygame.draw.line(surface, (25, 32, 45), (x, gy), (x + w, gy), 1)

        if not hasattr(self, 'txt_trend'):
            self.txt_trend = self.font_small.render("Success Rate % Trend", True, UI_ACCENT_GREEN)
        surface.blit(self.txt_trend, (x + 6, y + 2))

        if len(self.history_success) > 1:
            pts = []
            dx = w / max(1, len(self.history_success) - 1)
            for i, val in enumerate(self.history_success):
                norm_v = max(0.0, min(1.0, val / 100.0))
                py = y + h - (norm_v * (h - 10)) - 3
                pts.append((x + i * dx, py))

            if len(pts) >= 2:
                pygame.draw.lines(surface, UI_ACCENT_GREEN, False, pts, 2)
                last_pt = pts[-1]
                pygame.draw.circle(surface, (255, 255, 255), (int(last_pt[0]), int(last_pt[1])), 3)

    def _draw_neural_net_hud(self, surface):
        nn_top = 525
        nn_rect = pygame.Rect(self.hud_x + 15, nn_top, self.hud_width - 30, SCREEN_HEIGHT - nn_top - 10)
        pygame.draw.rect(surface, (20, 26, 36), nn_rect, border_radius=6)
        pygame.draw.rect(surface, UI_PANEL_BORDER, nn_rect, width=1, border_radius=6)

        car = self.sim.selected_vehicle
        if car is None or not car.is_alive:
            active = [c for c in self.sim.vehicles if c.is_alive]
            car = active[0] if active else None

        if not hasattr(self, 'txt_no_car'):
            self.txt_no_car = self.font_main.render("No active vehicle selected", True, UI_TEXT_MUTED)
            self.txt_q = self.font_bold.render("Action Q-Values (Decision Probabilities):", True, UI_TEXT_MUTED)
            self.txt_nn = self.font_bold.render("Neural Network Layer Activations:", True, UI_TEXT_MUTED)
            self.q_lbl_cache = {}
            self.nn_lbl_cache = {}

        if car is None or car.last_state is None:
            surface.blit(self.txt_no_car, (self.hud_x + 85, nn_top + 100))
            return

        car_info = f"🧠 AI BRAIN INSPECTOR — {car.v_type} #{car.id}"
        surface.blit(self.font_bold.render(car_info, True, UI_ACCENT_CYAN), (self.hud_x + 25, nn_top + 8))

        spd_kmh = car.speed * 12.0
        dist_s = car.get_distance_to_stop_line()
        stop_txt = f"{dist_s:.0f}px" if dist_s is not None else "Passed"
        ttc_txt = f"{car.sensors.min_ttc:.1f}s" if car.sensors.min_ttc < 10.0 else "Clear"
        telemetry = f"Speed: {spd_kmh:.1f} km/h | TTC: {ttc_txt} | Red: {stop_txt}"
        surface.blit(self.font_small.render(telemetry, True, UI_TEXT_WHITE), (self.hud_x + 25, nn_top + 26))

        q_vals = self.sim.agent.get_q_values(car.last_state)
        q_labels = ["COAST", "ACCEL 1", "ACCEL 2", "BRAKE 1", "BRAKE 2"]
        chosen_act = car.last_action

        bar_x = self.hud_x + 95
        bar_max_w = 140

        min_q = min(-1.0, float(np.min(q_vals)))
        max_q = max(1.0, float(np.max(q_vals)))
        q_range = max(0.1, max_q - min_q)

        surface.blit(self.txt_q, (self.hud_x + 25, nn_top + 42))

        for idx, (lbl, qv) in enumerate(zip(q_labels, q_vals)):
            by = nn_top + 58 + idx * 18
            is_chosen = (idx == chosen_act)

            lbl_col = UI_ACCENT_CYAN if is_chosen else UI_TEXT_MUTED
            
            if (lbl, is_chosen) not in self.q_lbl_cache:
                self.q_lbl_cache[(lbl, is_chosen)] = self.font_mono.render(f"{lbl:7s}", True, lbl_col)
            surface.blit(self.q_lbl_cache[(lbl, is_chosen)], (self.hud_x + 25, by))

            norm_w = max(0.05, min(1.0, (qv - min_q) / q_range))
            bar_w = int(norm_w * bar_max_w)

            pygame.draw.rect(surface, (35, 42, 55), (bar_x, by + 2, bar_max_w, 10), border_radius=3)
            fill_col = UI_ACCENT_GREEN if is_chosen else (0, 160, 220)
            pygame.draw.rect(surface, fill_col, (bar_x, by + 2, bar_w, 10), border_radius=3)

            val_str = f"{qv:+.2f}"
            surface.blit(self.font_mono.render(val_str, True, UI_TEXT_WHITE if is_chosen else UI_TEXT_MUTED), (bar_x + bar_max_w + 10, by))

        ny = nn_top + 148
        surface.blit(self.txt_nn, (self.hud_x + 25, ny))

        activations = self.sim.agent.get_activations(car.last_state)
        layer_cols = [
            (self.hud_x + 55, activations['inputs'][:12], "Sensors"),
            (self.hud_x + 135, activations['h1'][:10], "Dense 1"),
            (self.hud_x + 215, activations['h2'][:10], "Dense 2"),
            (self.hud_x + 295, activations['q_values'], "Q-Out")
        ]

        node_y_start = ny + 18
        for l_idx, (col_x, vals, l_name) in enumerate(layer_cols):
            surface.blit(self.font_small.render(l_name, True, UI_TEXT_MUTED), (col_x - 18, node_y_start - 8))
            num_nodes = len(vals)
            node_spacing = 4.5
            for n_idx, val in enumerate(vals):
                nd_y = node_y_start + 10 + n_idx * node_spacing
                intensity = max(0.1, min(1.0, float(abs(val))))
                node_col = (int(0 * intensity), int(220 * intensity), int(255 * intensity))
                pygame.draw.circle(surface, node_col, (col_x, int(nd_y)), 3)

                if l_idx < len(layer_cols) - 1:
                    next_col_x = layer_cols[l_idx + 1][0]
                    next_nodes = len(layer_cols[l_idx + 1][1])
                    for next_n in range(min(4, next_nodes)):
                        next_y = node_y_start + 10 + next_n * (node_spacing * (num_nodes / max(1, next_nodes)))
                        # Glowing connection line
                        if intensity > 0.3:
                            pygame.draw.line(surface, (0, 150, 200), (col_x + 3, int(nd_y)), (next_col_x - 3, int(next_y)), 2)
                            pygame.draw.line(surface, (50, 200, 255), (col_x + 3, int(nd_y)), (next_col_x - 3, int(next_y)), 1)
                        else:
                            pygame.draw.line(surface, (30, 45, 65), (col_x + 3, int(nd_y)), (next_col_x - 3, int(next_y)), 1)
