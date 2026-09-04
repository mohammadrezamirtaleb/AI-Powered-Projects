"""
Main Entry Point for the DeepRL Autonomous Crossroad Simulation.
Handles the simulation loop, vehicle spawner, dynamic weather, pedestrian crosswalks, collision checks, Day/Night rendering, and Pygame events.
"""
import sys
import os
import math
import random
import pygame

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, SIM_NAME,
    SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX, MAX_ACTIVE_CARS,
    ACTION_REPEAT, RL_GAMMA
)
from src.simulation.intersection import Intersection
from src.simulation.traffic_controller import TrafficController
from src.simulation.vehicle import Vehicle, check_sat_collision
from src.simulation.pedestrians import PedestrianManager
from src.simulation.weather import WeatherManager
from src.simulation.particles import ParticleManager
import threading
import time
from src.ai.dqn_agent import DQNAgent
from src.render.renderer import Renderer
from src.render.lighting import LightingEngine
from src.render.ui_hud import UIHud
from src.train_headless import get_expert_action

class TrainingWorker(threading.Thread):
    def __init__(self, sim):
        super().__init__(daemon=True)
        self.sim = sim
        self.running = True

    def run(self):
        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass

        while self.running:
            try:
                if self.sim.agent.mode == 'TRAINING' and len(self.sim.agent.memory) >= 64 and self.sim.sim_speed > 0:
                    self.sim.agent.train_step()
                    time.sleep(0.008) # Smooth GIL yielding keeps 60 FPS
                else:
                    time.sleep(0.025)
            except Exception:
                time.sleep(0.02)

class Simulation:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(f"{SIM_NAME} - [F11 Fullscreen | Esc Quit]")
        # Create sleek cyber AI window icon
        icon_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(icon_surf, (0, 215, 255), (16, 16), 14)
        pygame.draw.circle(icon_surf, (20, 25, 35), (16, 16), 10)
        pygame.draw.circle(icon_surf, (46, 230, 138), (16, 16), 5)
        pygame.display.set_icon(icon_surf)

        # Automatically detect desktop monitor resolution
        info = pygame.display.Info()
        monitor_w = info.current_w if info.current_w > 800 else SCREEN_WIDTH
        monitor_h = info.current_h if info.current_h > 600 else SCREEN_HEIGHT

        # Automatically size window to fit monitor comfortably without taskbar clipping:
        self.width = monitor_w
        self.height = monitor_h - 55 if monitor_h > 700 else monitor_h
        self.topnav_h = 48
        self.sidebar_w = 340

        # Windowed resizable mode with native title bar controls
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.world_surface = pygame.Surface((self.width, self.height))
        self.is_fullscreen = False
        self.clock = pygame.time.Clock()

        # Canvas layout (Simulation area between topnav and sidebar)
        canvas_w = self.width - self.sidebar_w
        canvas_h = self.height - self.topnav_h
        center_x = canvas_w // 2
        center_y = self.topnav_h + (canvas_h // 2)

        # Core Simulation Components
        self.intersection = Intersection(center_x, center_y, self.width, self.height, self.topnav_h, self.sidebar_w)
        self.traffic_controller = TrafficController()
        self.weather = WeatherManager()
        self.pedestrian_mgr = PedestrianManager(self.intersection)
        self.particle_mgr = ParticleManager()
        self.agent = DQNAgent()
        self.renderer = Renderer(self.screen, center_x, center_y, self.width, self.height, self.topnav_h, self.sidebar_w)
        self.lighting = LightingEngine(self.width, self.height)

        # Vehicles
        self.vehicles = []
        self.selected_vehicle = None
        self.spawn_timers = {route.id: random.uniform(0.5, 2.5) for route in self.intersection.routes}

        # Day / Night lighting state
        self.night_factor = 0.0 # 0.0 = day, 1.0 = night
        self.target_night_factor = 0.0
        self.auto_day_night = False
        self.day_night_timer = 0.0

        # Simulation Speed & Controls
        self.sim_speed = 1.0
        self.show_vision_rays = True

        # Viral Features Flags
        self.cinematic_mode = False
        self.v2v_enabled = True
        self.jaywalking_enabled = True

        # Statistics
        self.stats = {
            'total_spawned': 0,
            'total_passed': 0,
            'total_crashes': 0,
            'episodes': 0,
        }

        self.training_time = 0.0
        self.training_episodes = 0
        self.max_active_cars = MAX_ACTIVE_CARS
        self.adaptive_lights_enabled = True

        # UI HUD
        self.hud = UIHud(self, self.width, self.height, self.topnav_h, self.sidebar_w)

        # Load pretrained weights if available
        weights_path = os.path.join(os.path.dirname(__file__), 'ai', 'weights', 'pretrained_master.pt')
        if os.path.exists(weights_path):
            try:
                self.agent.load_weights(weights_path)
                self.agent.set_mode('MASTER')
                print(f"[AI] Loaded pre-trained master weights from {weights_path}")
            except Exception as e:
                print(f"[AI] Error loading weights: {e}. Starting fresh.")
                self.agent.set_mode('TRAINING')
        else:
            self.agent.set_mode('TRAINING')
            print("[AI] Starting with Live Training Mode.")

        # Multi-Threaded Asynchronous Training Worker
        self.training_worker = TrainingWorker(self)
        self.training_worker.start()

    def resize_world(self, new_w, new_h):
        self.width = max(1000, new_w)
        self.height = max(600, new_h)
        if not self.is_fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.world_surface = pygame.Surface((self.width, self.height))

        canvas_w = self.width - self.sidebar_w
        canvas_h = self.height - self.topnav_h
        center_x = canvas_w // 2
        center_y = self.topnav_h + (canvas_h // 2)

        self.intersection.update_dimensions(center_x, center_y, self.width, self.height, self.topnav_h, self.sidebar_w)
        self.renderer.update_dimensions(self.screen, center_x, center_y, self.width, self.height, self.topnav_h, self.sidebar_w)
        self.lighting.update_dimensions(self.width, self.height)
        self.hud.update_dimensions(self.width, self.height, self.topnav_h, self.sidebar_w)
        self.pedestrian_mgr.update_dimensions()
        self.spawn_timers = {route.id: random.uniform(0.5, 2.5) for route in self.intersection.routes}

    def set_speed(self, speed):
        self.sim_speed = speed

    def reset_simulation(self):
        self.vehicles.clear()
        self.reset_statistics()

    def toggle_adaptive_lights(self):
        self.adaptive_lights_enabled = not self.adaptive_lights_enabled
        self.traffic_controller.adaptive_mode = self.adaptive_lights_enabled

    def set_day_night(self, factor):
        self.target_night_factor = factor
        self.auto_day_night = False

    def toggle_auto_day_night(self):
        self.auto_day_night = not self.auto_day_night

    def set_sim_speed(self, speed):
        self.sim_speed = speed

    def set_ai_mode(self, mode):
        self.agent.set_mode(mode)

    def toggle_vision_overlay(self):
        self.show_vision_rays = not self.show_vision_rays

    @property
    def show_vision_overlay(self):
        return self.show_vision_rays

    def toggle_cinematic(self):
        self.cinematic_mode = not self.cinematic_mode

    def toggle_v2v(self):
        self.v2v_enabled = not self.v2v_enabled

    def toggle_jaywalking(self):
        self.jaywalking_enabled = not self.jaywalking_enabled

    def reset_statistics(self):
        self.stats['total_spawned'] = 0
        self.stats['total_passed'] = 0
        self.stats['total_crashes'] = 0
        self.particle_mgr.clear()

    def save_model(self):
        weights_path = os.path.join(os.path.dirname(__file__), 'ai', 'weights', 'pretrained_master.pt')
        self.agent.save_weights(weights_path)
        print(f"[AI] Weights saved to {weights_path}")

    def spawn_random_vehicle(self, v_type=None):
        if len(self.vehicles) >= MAX_ACTIVE_CARS:
            return None

        available_routes = list(self.intersection.routes)
        random.shuffle(available_routes)

        for route in available_routes:
            p_start = route.key_points[0]
            spawn_clear = True
            for car in self.vehicles:
                if math.hypot(car.x - p_start[0], car.y - p_start[1]) < 115.0:
                    spawn_clear = False
                    break

            if spawn_clear:
                car = Vehicle(route, v_type=v_type, spawn_speed=random.uniform(2.0, 3.2))
                self.vehicles.append(car)
                self.stats['total_spawned'] += 1
                return car
        return None

    def spawn_ambulance(self):
        return self.spawn_random_vehicle(v_type='AMBULANCE')

    def update_spawners(self, dt):
        if len(self.vehicles) >= MAX_ACTIVE_CARS:
            return

        for route in self.intersection.routes:
            self.spawn_timers[route.id] -= dt * self.sim_speed
            if self.spawn_timers[route.id] <= 0:
                self.spawn_timers[route.id] = random.uniform(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)

                p_start = route.key_points[0]
                spawn_clear = True
                for car in self.vehicles:
                    if math.hypot(car.x - p_start[0], car.y - p_start[1]) < 115.0:
                        spawn_clear = False
                        break

                if spawn_clear and len(self.vehicles) < MAX_ACTIVE_CARS:
                    car = Vehicle(route, spawn_speed=random.uniform(2.2, 3.2))
                    self.vehicles.append(car)
                    self.stats['total_spawned'] += 1

    def handle_collisions(self):
        """Check SAT OBB collisions between all active vehicle pairs with fault attribution and physical anti-overlap."""
        n = len(self.vehicles)
        for i in range(n):
            v1 = self.vehicles[i]
            for j in range(i + 1, n):
                v2 = self.vehicles[j]

                dist = math.hypot(v1.x - v2.x, v1.y - v2.y)
                max_rad = (v1.length + v2.length) / 2.0
                if dist > max_rad * 1.2:
                    continue

                if check_sat_collision(v1.get_corners(), v2.get_corners()):
                    # Elastic positional separation to prevent vehicles from overlapping
                    if dist > 0.01:
                        sep = max(1.5, (max_rad - dist) * 0.5)
                        nx = (v1.x - v2.x) / dist
                        ny = (v1.y - v2.y) / dist
                        v1.x += nx * sep
                        v1.y += ny * sep
                        v2.x -= nx * sep
                        v2.y -= ny * sep

                    if v1.is_alive or v2.is_alive:
                        cx = (v1.x + v2.x) / 2.0
                        cy = (v1.y + v2.y) / 2.0

                        v1_moving = v1.speed > 0.3
                        v2_moving = v2.speed > 0.3
                        if v1_moving and not v2_moving:
                            v1_fault, v2_fault = True, False
                        elif v2_moving and not v1_moving:
                            v1_fault, v2_fault = False, True
                        else:
                            v1_fault, v2_fault = True, True

                        if v1.is_alive:
                            v1.crash(is_at_fault=v1_fault)
                        if v2.is_alive:
                            v2.crash(is_at_fault=v2_fault)
                        self.stats['total_crashes'] += 1

                        self.particle_mgr.emit_crash(cx, cy, intensity=1.5)
                        self.particle_mgr.add_skid(v1.x, v1.y, v2.x, v2.y)

        # Check vehicle-pedestrian collisions
        for v in self.vehicles:
            if not v.is_alive:
                continue
            for ped in self.pedestrian_mgr.pedestrians:
                if not ped.is_alive:
                    continue
                dist = math.hypot(v.x - ped.x, v.y - ped.y)
                if dist < (v.length / 2 + ped.radius + 2.0):
                    if check_sat_collision(v.get_corners(), ped.get_corners()):
                        v.crash(is_at_fault=True)
                        ped.is_alive = False
                        self.stats['total_crashes'] += 1
                        self.particle_mgr.emit_crash(ped.x, ped.y, intensity=0.8)
                        break

    def step_simulation(self, dt):
        scaled_dt = dt * self.sim_speed

        if self.sim_speed <= 0.0:
            return

        if self.agent.mode == 'TRAINING':
            self.training_time += scaled_dt

        # 1. Update Traffic Lights (with adaptive queue & emergency preemption)
        self.traffic_controller.update(scaled_dt, self.vehicles)

        # 2. Update Weather Engine
        self.weather.update(scaled_dt)

        # 3. Update Pedestrians
        self.pedestrian_mgr.update(scaled_dt, self.traffic_controller, jaywalking_enabled=self.jaywalking_enabled)

        # 4. Update Day/Night Transition
        if self.auto_day_night:
            self.day_night_timer += scaled_dt * 0.15
            self.night_factor = (math.sin(self.day_night_timer) + 1.0) / 2.0
        else:
            self.night_factor += (self.target_night_factor - self.night_factor) * 0.08

        # 5. Update Vehicle Spawning
        self.update_spawners(dt)

        # 6. Perception & Action (Pass 1)
        grip = self.weather.friction_coeff

        for car in self.vehicles:
            if not car.is_alive:
                continue

            tl_state = self.traffic_controller.get_light_state(car.route.start_dir)

            # Deep RL Decision only when starting a new macro-action
            if car.frames_in_action == 0 or car.macro_start_state is None:
                raw_state = car.sensors.update(
                    self.vehicles, tl_state,
                    self.intersection.junction_bounds,
                    friction_coeff=grip,
                    pedestrians=self.pedestrian_mgr.pedestrians
                )
                state = car.get_stacked_state(raw_state)
                car.last_state = state
                car.macro_start_state = state
                car.accumulated_reward = 0.0

                expert_prob = max(0.0, (self.agent.epsilon - 0.2) / 0.8) if self.agent.mode == 'TRAINING' else 0.0
                if random.random() < expert_prob:
                    action = get_expert_action(car, tl_state, car.get_distance_to_stop_line(), self.vehicles, self.intersection.junction_bounds)
                else:
                    action = self.agent.select_action(state)
                
                car.macro_action = action
                car.apply_action(action)

        # 7. Physics Update (Pass 2)
        for car in self.vehicles:
            tl_state = self.traffic_controller.get_light_state(car.route.start_dir)
            car.update_physics(
                scaled_dt, 
                friction_coeff=grip, 
                current_tl_state=tl_state,
                all_vehicles=self.vehicles,
                puddles=self.weather.puddles,
                v2v_enabled=self.v2v_enabled,
                pedestrians=self.pedestrian_mgr.pedestrians
            )

        # 8. Collision Checks (Pass 3)
        self.handle_collisions()

        # 9. Step Reward Accumulation & Transition Storage (Pass 4)
        for car in self.vehicles:
            if car.macro_start_state is None:
                continue

            tl_state = self.traffic_controller.get_light_state(car.route.start_dir)
            step_reward = self.agent.calculate_reward(car, tl_state, car.get_distance_to_stop_line())
            car.total_reward += step_reward
            car.accumulated_reward += (RL_GAMMA ** car.frames_in_action) * step_reward
            car.frames_in_action += 1

            done = car.has_crashed or car.has_finished
            if done:
                # Do not record innocent non-at-fault vehicle crash as agent failure
                if car.has_crashed and not getattr(car, 'is_at_fault', True):
                    car.macro_start_state = None
                    car.frames_in_action = 0
                    continue

                raw_next_state = car.sensors.update(
                    self.vehicles, tl_state,
                    self.intersection.junction_bounds,
                    friction_coeff=grip,
                    pedestrians=self.pedestrian_mgr.pedestrians
                )
                next_state = car.get_stacked_state(raw_next_state)
                self.agent.store_transition(car.macro_start_state, car.macro_action, car.accumulated_reward, next_state, True)
                car.macro_start_state = None
                car.frames_in_action = 0
            elif car.frames_in_action >= ACTION_REPEAT:
                raw_next_state = car.sensors.update(
                    self.vehicles, tl_state,
                    self.intersection.junction_bounds,
                    friction_coeff=grip,
                    pedestrians=self.pedestrian_mgr.pedestrians
                )
                next_state = car.get_stacked_state(raw_next_state)
                self.agent.store_transition(car.macro_start_state, car.macro_action, car.accumulated_reward, next_state, False)
                car.frames_in_action = 0
                car.accumulated_reward = 0.0
                car.macro_start_state = None

        # 8. Deep RL Optimization runs asynchronously in TrainingWorker thread!
        # This keeps the main render loop locked at 60 FPS without stutter.

        # 9. Update Particles
        self.particle_mgr.update()

        # 10. Clean up finished vehicles
        surviving = []
        for car in self.vehicles:
            if car.has_finished:
                self.stats['total_passed'] += 1
                self.training_episodes += 1
            elif car.has_crashed and getattr(car, 'time_since_crash', 0.0) >= 3.0:
                self.training_episodes += 1
            else:
                surviving.append(car)
        self.vehicles = surviving

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            
            # RL requires a stable physics time-step, regardless of rendering lag
            fixed_dt = 1.0 / 60.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.resize_world(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if mx < self.hud.sidebar_x and my > self.hud.topnav_h:
                        clicked_car = None
                        for car in self.vehicles:
                            if car.is_alive and math.hypot(car.x - mx, car.y - my) < 32.0:
                                clicked_car = car
                                break
                        if clicked_car:
                            self.selected_vehicle = clicked_car
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F11:
                        self.is_fullscreen = not self.is_fullscreen
                        info = pygame.display.Info()
                        if self.is_fullscreen:
                            self.screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
                            self.resize_world(info.current_w, info.current_h)
                        else:
                            self.resize_world(info.current_w, info.current_h - 55)
                    elif event.key == pygame.K_SPACE:
                        self.sim_speed = 0.0 if self.sim_speed > 0 else 1.0
                    elif event.key == pygame.K_1:
                        self.set_ai_mode('UNTRAINED')
                    elif event.key == pygame.K_2:
                        self.set_ai_mode('TRAINING')
                    elif event.key == pygame.K_3:
                        self.set_ai_mode('MASTER')
                    elif event.key == pygame.K_n:
                        self.set_day_night(1.0 if self.night_factor < 0.5 else 0.0)
                    elif event.key == pygame.K_w:
                        self.weather.toggle_weather()
                    elif event.key == pygame.K_t:
                        self.traffic_controller.switch_manual_phase()
                    elif event.key == pygame.K_v:
                        self.toggle_vision_overlay()
                    elif event.key == pygame.K_s:
                        self.spawn_random_vehicle()
                    elif event.key == pygame.K_a:
                        self.spawn_ambulance()
                    elif event.key == pygame.K_r:
                        self.reset_statistics()

                self.hud.handle_event(event)

            self.step_simulation(fixed_dt)

            # 1. Asphalt Road, zebra markings, grass
            self.renderer.render_environment(self.world_surface, self.night_factor)

            # 2. Tire Skid Marks
            self.particle_mgr.draw_skids(self.world_surface)

            # 3. Traffic Light Post Enclosures
            self.renderer.render_traffic_lights(
                self.world_surface, self.traffic_controller,
                self.intersection.light_poles, self.night_factor
            )

            # 4. Pedestrians on Zebra Crossings
            is_night_bool = (self.night_factor > 0.35)
            self.pedestrian_mgr.draw(self.world_surface, is_night=is_night_bool)

            # 5. Vehicles (Sedans, SUVs, Trucks, Buses, Sports, Motorcycles, Ambulances)
            for car in self.vehicles:
                is_sel = (self.selected_vehicle is not None and self.selected_vehicle.id == car.id)
                car.draw(self.world_surface, is_night=is_night_bool, is_selected=is_sel)

            # 6. Vision / LiDAR Raycasts Overlay
            if self.show_vision_rays:
                if self.selected_vehicle and self.selected_vehicle.is_alive:
                    self.renderer.render_sensor_rays(self.world_surface, self.selected_vehicle)
                elif self.vehicles:
                    for car in self.vehicles:
                        if car.is_alive:
                            self.renderer.render_sensor_rays(self.world_surface, car)
                            break

            # 7. Particles (crash sparks, smoke, explosions)
            self.particle_mgr.draw_particles(self.world_surface)

            # 8. Dynamic 2D Day/Night Lighting Engine
            light_dict = {
                'N': self.traffic_controller.get_light_state('N'),
                'S': self.traffic_controller.get_light_state('S'),
                'E': self.traffic_controller.get_light_state('E'),
                'W': self.traffic_controller.get_light_state('W')
            }
            self.lighting.render_lighting(
                self.world_surface, self.vehicles, light_dict,
                self.intersection.light_poles, self.particle_mgr.particles,
                self.night_factor
            )

            # 9. Dynamic Rain, Wind Streaks, Splashes & Wet Road Sheen
            self.weather.draw(self.world_surface)

            # Apply Cinematic Mode Transformation & HUD rendering
            if self.cinematic_mode and self.selected_vehicle:
                zoom = 1.6
                cw, ch = SCREEN_WIDTH, SCREEN_HEIGHT
                target_x = self.selected_vehicle.x
                target_y = self.selected_vehicle.y
                
                scaled_w = int(cw * zoom)
                scaled_h = int(ch * zoom)
                scaled_world = pygame.transform.scale(self.world_surface, (scaled_w, scaled_h))
                
                offset_x = (cw / 2) - (target_x * zoom)
                offset_y = (ch / 2) - (target_y * zoom)
                
                temp_surf = pygame.Surface((self.width, self.height))
                temp_surf.fill((0, 0, 0))
                temp_surf.blit(scaled_world, (offset_x, offset_y))
                self.hud.draw(temp_surf)
                self.screen.blit(temp_surf, (0, 0))
            else:
                self.hud.draw(self.world_surface)
                self.screen.blit(self.world_surface, (0, 0))

            pygame.display.flip()

        pygame.quit()

if __name__ == '__main__':
    sim = Simulation()
    sim.run()
