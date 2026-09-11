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
from src.simulation.vehicle import (
    Vehicle, check_sat_collision, resolve_vehicle_collisions, ring_has_priority_traffic
)
from src.simulation.pedestrians import PedestrianManager
from src.ai.dqn_agent import DQNAgent

def _in_roundabout(car):
    rbx = getattr(car.route, 'rbx', None)
    rby = getattr(car.route, 'rby', None)
    if rbx is None or rby is None:
        return False
    return math.hypot(car.x - rbx, car.y - rby) < getattr(car.route, 'r_outer', 88.0) + 12.0


def forward_hazard(car, all_vehicles=None, pedestrians=None):
    """
    Distance to the nearest obstacle actually in this vehicle's path.

    Raw LiDAR is not path-aware. On a curved roundabout arc the wide rays graze
    vehicles on neighbouring arcs, and braking on those readings made the ring
    deadlock: cars stopped for obstacles that were never in their way, and once
    stopped they became priority traffic for every approach. The corridor scan is
    authoritative; the three central rays are the backstop only off the ring.
    """
    hazard = getattr(car, 'last_lead_clearance', None)
    if hazard is None and all_vehicles is not None:
        hazard, _, _ = car.get_leading_obstacle(all_vehicles, pedestrians)
    if hazard is None:
        hazard = 160.0
    if not _in_roundabout(car):
        hits = car.sensors.ray_hits
        if hits:
            for i in (3, 4, 5):
                if i < len(hits):
                    hazard = min(hazard, hits[i][1])
    return min(hazard, 160.0)


def path_ttc(car):
    """Corridor TTC. LiDAR TTC is ignored on the ring where rays graze other arcs."""
    ttc = getattr(car, 'live_ttc', 99.0)
    if not _in_roundabout(car):
        ttc = min(ttc, getattr(car.sensors, 'min_ttc', 99.0))
    return ttc


def braking_gaps(car):
    """
    Clearance at which this vehicle should brake, derived from its stopping
    distance rather than a fixed number of pixels.

    Fixed thresholds livelock a queue: the old expert braked below 46px while the
    headway controller only enforced 26px, so a car stopped 45px behind another
    stopped car braked forever and the queue could never close up and restart.
    At a standstill the gaps collapse to a nose-to-tail buffer, so a stationary
    vehicle with clear road ahead always pulls away.

    Returns (hard_gap, soft_gap) in pixels.
    """
    brake = max(0.05, car.max_brake)
    stop_dist = (car.speed * car.speed) / (2.0 * brake)
    return 12.0 + stop_dist * 0.6, 22.0 + stop_dist * 1.15


def get_expert_action(car, traffic_light_state, dist_to_stop, all_vehicles, junction_bounds):
    """
    Expert rule-based policy for generating high-quality demonstrations for DQfD.
    Includes conflict-zone yielding and safe stopping distances.
    """
    min_front_dist = forward_hazard(car, all_vehicles)
    hard_gap, soft_gap = braking_gaps(car)

    is_red_or_yellow = traffic_light_state in ('RED', 'YELLOW')
    is_yield = traffic_light_state == 'YIELD'

    # 1. Emergency collision avoidance
    if min_front_dist < hard_gap:
        return 4
    elif min_front_dist < soft_gap:
        return 3

    yld = getattr(car.route, 'yield_line_dist', None)
    if is_yield and yld is not None and not getattr(car, 'has_cleared_yield', False):
        dist_y = yld - car.path_distance
        if 0.0 < dist_y < 40.0 and ring_has_priority_traffic(car, all_vehicles):
            return 4 if car.speed > 0.8 else 3

    # 2. Red/Yellow light stopping logic (Hold brake until green, never accelerate past stop line)
    if is_red_or_yellow and dist_to_stop is not None and dist_to_stop <= 120.0 and not car.has_passed_intersection:
        if dist_to_stop < 22.0:
            return 4 if car.speed > 0.1 else 3 # Full stop (hold brake firmly)
        elif dist_to_stop < 55.0:
            return 4 if car.speed > 1.8 else 3 # Firm deceleration
        elif dist_to_stop < 105.0 and car.speed > 1.5:
            return 3 # Prepare to slow down

    ttc = path_ttc(car)
    if ttc < 1.2:
        return 4
    if ttc < 2.0 and min_front_dist < soft_gap * 2.0:
        return 3

    if dist_to_stop is not None and -10.0 <= dist_to_stop <= 36.0 and not car.has_passed_intersection:
        jx_min, jy_min, jx_max, jy_max = junction_bounds
        for other in all_vehicles:
            if other.id == car.id or not other.is_alive:
                continue
            if other.speed < 0.25 and getattr(other, 'time_stalled', 0.0) > 2.0:
                continue
            if not (jx_min <= other.x <= jx_max and jy_min <= other.y <= jy_max):
                continue
            angle_diff = abs((other.angle - car.angle + math.pi) % (2 * math.pi) - math.pi)
            if math.radians(50) <= angle_diff <= math.radians(130):
                return 4 if car.speed > 0.4 else 3

    if car.speed < car.target_speed * 0.92:
        return 2 if car.speed < 1.8 else 1
    return 0


def apply_safety_shield(car, action, traffic_light_state, all_vehicles, junction_bounds):
    """Hard override so the policy cannot accelerate into an imminent collision."""
    min_front = forward_hazard(car, all_vehicles)
    ttc = path_ttc(car)
    hard_gap, soft_gap = braking_gaps(car)
    dist_to_stop = car.get_distance_to_stop_line()
    if dist_to_stop is not None and dist_to_stop > 4000:
        dist_to_stop = None

    if min_front < hard_gap or ttc < 1.0:
        return 4
    if min_front < soft_gap or ttc < 1.7:
        return max(action, 3)

    if traffic_light_state in ('RED', 'YELLOW') and not car.has_passed_intersection and not car.is_emergency:
        if dist_to_stop is not None and 0.0 < dist_to_stop < 70.0 and action in (1, 2):
            return 4 if dist_to_stop < 28.0 else 3

    if traffic_light_state == 'YIELD' and not getattr(car, 'has_cleared_yield', False):
        yld = getattr(car.route, 'yield_line_dist', None)
        if yld is not None:
            dist_y = yld - car.path_distance
            if 0.0 < dist_y < 45.0 and action in (1, 2) and ring_has_priority_traffic(car, all_vehicles):
                return 3

    if dist_to_stop is not None and -8.0 <= dist_to_stop <= 32.0 and not car.has_passed_intersection:
        jx_min, jy_min, jx_max, jy_max = junction_bounds
        for other in all_vehicles:
            if other.id == car.id or not other.is_alive:
                continue
            if other.speed < 0.25 and getattr(other, 'time_stalled', 0.0) > 2.0:
                continue
            if jx_min <= other.x <= jx_max and jy_min <= other.y <= jy_max:
                angle_diff = abs((other.angle - car.angle + math.pi) % (2 * math.pi) - math.pi)
                if math.radians(50) <= angle_diff <= math.radians(130) and action in (0, 1, 2):
                    return 3

    # Anti-deadlock: a standing vehicle with a legal gap must pull away.
    # Untrained / coast policies otherwise freeze the whole city at speed 0.
    held_at_red = (
        traffic_light_state in ('RED', 'YELLOW')
        and not car.has_passed_intersection
        and not car.is_emergency
        and dist_to_stop is not None
        and 0.0 < dist_to_stop < 70.0
    )
    held_at_yield = False
    if traffic_light_state == 'YIELD' and not getattr(car, 'has_cleared_yield', False):
        yld = getattr(car.route, 'yield_line_dist', None)
        if yld is not None:
            dist_y = yld - car.path_distance
            held_at_yield = 0.0 < dist_y < 45.0 and ring_has_priority_traffic(car, all_vehicles)
    if car.speed < 0.4 and not held_at_red and not held_at_yield and min_front > 14.0 and ttc > 1.6:
        if action in (0, 3, 4):
            return 2 if car.speed < 0.2 else 1

    return action

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
    grip = 1.0      # dry road; the live sim varies this with the weather engine

    for step in range(1, total_steps + 1):
        # 1. Update Traffic Lights and Pedestrians
        traffic_controller.update(dt, vehicles)
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

            tl_state = intersection.signal_for(car, traffic_controller)
            bounds = intersection.bounds_for(car)

            if car.frames_in_action == 0 or car.macro_start_state is None:
                raw_state = car.sensors.update(
                    vehicles, tl_state, bounds,
                    friction_coeff=grip,
                    pedestrians=pedestrian_mgr.pedestrians
                )
                state = car.get_stacked_state(raw_state)
                car.macro_start_state = state
                car.accumulated_reward = 0.0

                # Early bootstrap with expert demonstrations transitioning to pure RL
                expert_prob = max(0.0, 1.0 - (step / (total_steps * 0.45)))
                if random.random() < expert_prob:
                    action = get_expert_action(car, tl_state, car.get_distance_to_stop_line(), vehicles, bounds)
                else:
                    action = agent.select_action(state)
                action = apply_safety_shield(car, action, tl_state, vehicles, bounds)

                car.macro_action = action
                car.apply_action(action)

        # 4. Physics (Pass 2)
        for car in vehicles:
            tl_state = intersection.signal_for(car, traffic_controller)
            car.update_physics(
                dt, friction_coeff=grip, current_tl_state=tl_state,
                all_vehicles=vehicles, pedestrians=pedestrian_mgr.pedestrians,
                conflict_bounds=intersection.bounds_for(car)
            )

        # 5. Collision checking with fault attribution (Pass 3)
        stats['crashes'] += resolve_vehicle_collisions(vehicles)

        # Vehicle-pedestrian collisions, matching the live simulation
        for v in vehicles:
            if not v.is_alive:
                continue
            for ped in pedestrian_mgr.pedestrians:
                if not ped.is_alive:
                    continue
                if math.hypot(v.x - ped.x, v.y - ped.y) < (v.length / 2 + ped.radius + 2.0):
                    if check_sat_collision(v.get_corners(), ped.get_corners()):
                        v.crash(is_at_fault=True, hit_pedestrian=True)
                        ped.is_alive = False
                        stats['crashes'] += 1
                        break


        # 6. Step Reward Accumulation & Transition Storage (Pass 4)
        for car in vehicles:
            if car.macro_start_state is None:
                continue

            tl_state = intersection.signal_for(car, traffic_controller)
            bounds = intersection.bounds_for(car)
            step_reward = agent.calculate_reward(
                car, tl_state, car.get_distance_to_stop_line(), dt=dt)
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
                    vehicles, tl_state, bounds,
                    friction_coeff=grip,
                    pedestrians=pedestrian_mgr.pedestrians
                )
                next_state = car.get_stacked_state(raw_next_state)
                agent.store_transition(car.macro_start_state, car.macro_action, car.accumulated_reward, next_state, True)
                car.macro_start_state = None
                car.frames_in_action = 0
            elif car.frames_in_action >= ACTION_REPEAT:
                raw_next_state = car.sensors.update(
                    vehicles, tl_state, bounds,
                    friction_coeff=grip,
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
            viol = ", ".join(f"{k}:{v}" for k, v in agent.top_violations(4))
            print(f"[{step:6d}/{total_steps}] "
                  f"Speed: {sps:6.1f} steps/s | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {agent.avg_loss:.4f} | "
                  f"Passed: {stats['passed']:4d} | "
                  f"Crashes: {stats['crashes']:4d} | "
                  f"Success: {success_rate:5.1f}%")
            if viol:
                print(f"{'':9}penalties -> {viol}")

    # Save trained master model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save_weights(save_path)
    print("=" * 60)
    print(f"[SUCCESS] Training completed successfully! Weights saved to: {save_path}")
    print("=" * 60)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Headless Deep RL trainer for the city crossroad.")
    ap.add_argument('--steps', type=int, default=22000, help="simulation steps to run")
    ap.add_argument('--out', type=str, default=None, help="checkpoint path")
    args = ap.parse_args()
    train_headless(total_steps=args.steps, save_path=args.out)
