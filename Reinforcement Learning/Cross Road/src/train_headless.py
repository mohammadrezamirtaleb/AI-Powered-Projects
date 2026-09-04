"""
Fast Headless Training Script for Deep Q-Network Agent.
Runs high-speed simulation steps without graphics overhead to train and export pre-trained master weights.
"""
import sys
import os
import math
import random
import time
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MAX_ACTIVE_CARS, ACTION_REPEAT, RL_GAMMA
from src.simulation.intersection import Intersection
from src.simulation.traffic_controller import TrafficController
from src.simulation.vehicle import Vehicle, check_sat_collision
from src.simulation.pedestrians import PedestrianManager
from src.ai.dqn_agent import DQNAgent

def get_expert_action(car, traffic_light_state, dist_to_stop, all_vehicles, junction_bounds):
    """
    Expert rule-based policy for generating high-quality demonstrations for DQfD.
    Includes conflict-zone yielding and safe stopping distances.
    """
    # Check all forward and angular LiDAR rays (rays 2, 3, 4, 5, 6)
    min_front_dist = 160.0
    if car.sensors.ray_hits:
        for i in [2, 3, 4, 5, 6]:
            if i < len(car.sensors.ray_hits):
                min_front_dist = min(min_front_dist, car.sensors.ray_hits[i][1])

    is_red_or_yellow = traffic_light_state in ('RED', 'YELLOW')

    # 1. Emergency collision avoidance
    if min_front_dist < 26.0:
        return 4 # BRAKE_HARD
    elif min_front_dist < 46.0:
        return 3 # BRAKE_MILD

    # 2. Red/Yellow light stopping logic (Hold brake until green, never accelerate past stop line)
    if is_red_or_yellow and dist_to_stop is not None and dist_to_stop <= 120.0 and not car.has_passed_intersection:
        if dist_to_stop < 22.0:
            return 4 if car.speed > 0.1 else 3 # Full stop (hold brake firmly)
        elif dist_to_stop < 55.0:
            return 4 if car.speed > 1.8 else 3 # Firm deceleration
        elif dist_to_stop < 105.0 and car.speed > 1.5:
            return 3 # Prepare to slow down

    # 3. Left Turn Conflict Zone Yielding (Only yield to straight oncoming)
    if car.route.turn_type == 'LEFT' and dist_to_stop is not None and -15.0 <= dist_to_stop <= 30.0:
        jx_min, jy_min, jx_max, jy_max = junction_bounds
        for other in all_vehicles:
            if other.id != car.id and other.is_alive:
                if jx_min <= other.x <= jx_max and jy_min <= other.y <= jy_max:
                    if other.route.turn_type == 'STRAIGHT':
                        if car.speed > 0.5:
                            return 3 # BRAKE_MILD

    # 4. Green or clear road: accelerate to target cruising speed
    if car.speed < car.target_speed * 0.92:
        return 2 if car.speed < 1.8 else 1
    return 0 # Coast

def train_headless(total_steps=22000, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'ai', 'weights', 'pretrained_master.pt')
    print("=" * 60)
    print("[TRAINER] Starting Fast Headless Deep RL Training (Phase 2 with PER & DQfD)...")
    print(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Target steps: {total_steps}")
    print("=" * 60)

    intersection = Intersection()
    traffic_controller = TrafficController()
    pedestrian_mgr = PedestrianManager(intersection)
    agent = DQNAgent()
    agent.set_mode('TRAINING')

    vehicles = []
    stats = {'spawned': 0, 'passed': 0, 'crashes': 0}
    spawn_timers = {route.id: random.uniform(0.2, 1.5) for route in intersection.routes}

    start_time = time.time()
    dt = 1.0 / 60.0 # Fixed DT for stable physical RL time!

    for step in range(1, total_steps + 1):
        # 1. Update Traffic Lights and Pedestrians
        traffic_controller.update(dt)
        pedestrian_mgr.update(dt, traffic_controller)

        # 2. Spawning
        if len(vehicles) < MAX_ACTIVE_CARS:
            for route in intersection.routes:
                spawn_timers[route.id] -= dt
                if spawn_timers[route.id] <= 0:
                    spawn_timers[route.id] = random.uniform(1.2, 2.8)
                    spawn_clear = True
                    for car in vehicles:
                        if car.is_alive and car.route.id == route.id and car.path_distance < 65.0:
                            spawn_clear = False
                            break
                    if spawn_clear and len(vehicles) < MAX_ACTIVE_CARS:
                        car = Vehicle(route, spawn_speed=random.uniform(2.0, 3.2))
                        car.decision_step = random.randint(0, ACTION_REPEAT - 1)
                        vehicles.append(car)
                        stats['spawned'] += 1

        # 3. Perception & Action (Pass 1)
        for car in vehicles:
            if not car.is_alive:
                continue

            tl_state = traffic_controller.get_light_state(car.route.start_dir)

            if car.frames_in_action == 0 or car.macro_start_state is None:
                raw_state = car.sensors.update(
                    vehicles, tl_state,
                    intersection.junction_bounds,
                    pedestrians=pedestrian_mgr.pedestrians
                )
                state = car.get_stacked_state(raw_state)
                car.macro_start_state = state
                car.accumulated_reward = 0.0

                # Early bootstrap with expert demonstrations transitioning to pure RL
                expert_prob = max(0.0, 1.0 - (step / (total_steps * 0.45)))
                if random.random() < expert_prob:
                    action = get_expert_action(car, tl_state, car.get_distance_to_stop_line(), vehicles, intersection.junction_bounds)
                else:
                    action = agent.select_action(state)

                car.macro_action = action
                car.apply_action(action)

        # 4. Physics (Pass 2)
        for car in vehicles:
            tl_state = traffic_controller.get_light_state(car.route.start_dir)
            car.update_physics(dt, current_tl_state=tl_state, all_vehicles=vehicles, pedestrians=pedestrian_mgr.pedestrians)

        # 5. Collision checking with fault attribution (Pass 3)
        n = len(vehicles)
        for i in range(n):
            v1 = vehicles[i]
            if not v1.is_alive:
                continue
            for j in range(i + 1, n):
                v2 = vehicles[j]
                if not v2.is_alive:
                    continue
                if math.hypot(v1.x - v2.x, v1.y - v2.y) < (v1.length + v2.length):
                    if check_sat_collision(v1.get_corners(), v2.get_corners()):
                        v1_moving = v1.speed > 0.3
                        v2_moving = v2.speed > 0.3
                        if v1_moving and not v2_moving:
                            v1_fault, v2_fault = True, False
                        elif v2_moving and not v1_moving:
                            v1_fault, v2_fault = False, True
                        else:
                            v1_fault, v2_fault = True, True

                        v1.crash(is_at_fault=v1_fault)
                        v2.crash(is_at_fault=v2_fault)
                        stats['crashes'] += 1
                        
                        if not v1.is_alive:
                            break
                            
        # 6. Step Reward Accumulation & Transition Storage (Pass 4)
        for car in vehicles:
            if car.macro_start_state is None:
                continue

            tl_state = traffic_controller.get_light_state(car.route.start_dir)
            step_reward = agent.calculate_reward(car, tl_state, car.get_distance_to_stop_line())
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
                    vehicles, tl_state,
                    intersection.junction_bounds,
                    pedestrians=pedestrian_mgr.pedestrians
                )
                next_state = car.get_stacked_state(raw_next_state)
                agent.store_transition(car.macro_start_state, car.macro_action, car.accumulated_reward, next_state, True)
                car.macro_start_state = None
                car.frames_in_action = 0
            elif car.frames_in_action >= ACTION_REPEAT:
                raw_next_state = car.sensors.update(
                    vehicles, tl_state,
                    intersection.junction_bounds,
                    pedestrians=pedestrian_mgr.pedestrians
                )
                next_state = car.get_stacked_state(raw_next_state)
                agent.store_transition(car.macro_start_state, car.macro_action, car.accumulated_reward, next_state, False)
                car.frames_in_action = 0
                car.accumulated_reward = 0.0
                car.macro_start_state = None

        # 7. Multi-gradient update steps per simulation step (4 mini-batches per sim frame for faster learning)
        for _ in range(4):
            agent.train_step()

        # 8. Cleanup (Crashed vehicles remain for 3.0s as obstacles, synced with main simulation)
        surviving = []
        for car in vehicles:
            if car.has_finished:
                stats['passed'] += 1
            elif car.has_crashed and getattr(car, 'time_since_crash', 0.0) >= 3.0:
                pass # Removed after 3 seconds so other agents learn to detect and avoid stationary wreckage
            else:
                surviving.append(car)
        vehicles = surviving

        # Progress logs
        if step % 1000 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            success_rate = (stats['passed'] / stats['spawned'] * 100.0) if stats['spawned'] > 0 else 0.0
            print(f"[{step:6d}/{total_steps}] "
                  f"Speed: {sps:6.1f} steps/s | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {agent.avg_loss:.4f} | "
                  f"Passed: {stats['passed']:4d} | "
                  f"Crashes: {stats['crashes']:4d} | "
                  f"Success: {success_rate:5.1f}%")

    # Save trained master model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save_weights(save_path)
    print("=" * 60)
    print(f"[SUCCESS] Training completed successfully! Weights saved to: {save_path}")
    print("=" * 60)

if __name__ == '__main__':
    train_headless()
