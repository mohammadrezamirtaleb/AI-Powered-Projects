"""
Deep Q-Network Agent for Autonomous Intersection Driving.
Implements Prioritized Experience Replay (PER), Target Network, Double Q-learning updates,
reward shaping, epsilon exploration, and mode management.
"""
import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    VISION_STATE_SIZE, STACKED_STATE_SIZE, NUM_ACTIONS, RL_GAMMA, RL_LR,
    RL_BATCH_SIZE, RL_BUFFER_CAPACITY, ACTION_REPEAT,
    RL_EPSILON_START, RL_EPSILON_MIN, RL_EPSILON_DECAY,
    REWARD_PROGRESS, REWARD_IDLE_RED, PENALTY_TIME, PENALTY_TAILGATE,
    PENALTY_TTC, PENALTY_SPEEDING, PENALTY_BLOCK_BOX, PENALTY_STALL,
    PENALTY_PED_PROXIMITY, PENALTY_RED_CREEP,
    REWARD_PASS_EVENT, REWARD_ROUTE_COMPLETE, REWARD_SMOOTH_STOP_RED,
    REWARD_ROUNDABOUT_CLEAR, REWARD_ROUNDABOUT_COURTESY, REWARD_YIELD_EMERGENCY,
    PENALTY_CRASH, PENALTY_CRASH_PEDESTRIAN, PENALTY_RED_LIGHT_RUN,
    PENALTY_STOP_LINE_OVERRUN, PENALTY_YIELD_VIOLATION, PENALTY_BLOCK_EMERGENCY,
    PENALTY_JERK, SAFE_TTC_SECONDS, SAFE_HEADWAY_FACTOR, MIN_SAFE_HEADWAY,
    SPEED_LIMIT_TOLERANCE
)
from src.ai.network import DuelingDQN

class SumTree:
    """
    Binary SumTree data structure for Prioritized Experience Replay.
    Provides O(log N) priority updates and sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree size: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Data storage for transitions
        self.data = [None] * capacity
        self.data_pointer = 0
        self.size = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate change up to the root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Traverse down tree to find leaf index corresponding to cumulative priority value v.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    def __init__(self, capacity=RL_BUFFER_CAPACITY, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.min_priority = 0.01

    def push(self, state, action, reward, next_state, done):
        priority = (self.max_priority ** self.alpha)
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        if len(self) < batch_size:
            return None

        batch = []
        indices = []
        priorities = []
        total_p = max(1e-5, self.tree.total_priority)
        segment = total_p / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            if data is None:
                # Fallback if unpopulated leaf is retrieved
                idx, priority, data = self.tree.get_leaf(random.uniform(0, total_p))
            priorities.append(priority)
            batch.append(data)
            indices.append(idx)

        sampling_probabilities = np.array(priorities, dtype=np.float32) / total_p
        weights = (len(self) * sampling_probabilities) ** (-beta)
        max_w = weights.max()
        if max_w > 0:
            weights /= max_w
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights
        )

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            # Clip priorities to [0.01, 10.0] to prevent priority explosion
            p = float(np.clip(abs(err) + 1e-5, self.min_priority, 10.0))
            self.max_priority = max(self.min_priority, min(10.0, max(self.max_priority * 0.999, p)))
            self.tree.update(idx, p ** self.alpha)

    def __len__(self):
        return self.tree.size


import threading

class DQNAgent:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Enable multi-core CPU parallelism if on CPU
        if self.device.type == 'cpu':
            try:
                torch.set_num_threads(4)
            except Exception:
                pass

        self.lock = threading.Lock()
        self.state_dim = STACKED_STATE_SIZE
        self.num_actions = NUM_ACTIONS

        # Networks
        self.q_network = DuelingDQN(self.state_dim, self.num_actions).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.num_actions).to(self.device)
        self.update_target_network(tau=1.0)

        # Auto-Tuning Dynamic Optimizer & PyTorch LR Scheduler
        self.min_lr = 2e-5
        self.max_lr = 1.5e-3
        self.current_lr = RL_LR
        self.grad_clip_norm = 1.0
        self.last_grad_norm = 0.0
        self.optimizer_status = "Optimal"

        # Modern AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=RL_LR, weight_decay=1e-5, amsgrad=True)

        # PyTorch Auto-Scheduler: Dynamically adapts learning rate based on real-time loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=80,
            threshold=0.01, min_lr=self.min_lr
        )

        self.memory = PrioritizedReplayBuffer()

        # Exploration
        self.epsilon = RL_EPSILON_START
        self.epsilon_min = RL_EPSILON_MIN
        self.epsilon_decay = RL_EPSILON_DECAY
        self.beta = 0.4 # PER beta annealing parameter

        # Operating Modes: 'UNTRAINED', 'TRAINING', 'MASTER'
        self.mode = 'TRAINING'

        # Metrics
        self.steps_done = 0
        self.train_step_count = 0
        self.recent_losses = deque(maxlen=100)
        self.avg_loss = 0.0

        # Violation telemetry: how often each penalty term has fired
        self.violation_counts = {}
        self.last_penalty = 0.0

    def reset_violation_counts(self):
        self.violation_counts = {}

    def top_violations(self, limit=5):
        """Most frequently triggered penalty terms, for the HUD and for tuning."""
        items = sorted(self.violation_counts.items(), key=lambda kv: -kv[1])
        return items[:limit]

    def update_target_network(self, tau=0.005):
        if tau >= 1.0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def set_mode(self, mode):
        """Set mode to 'UNTRAINED', 'TRAINING', or 'MASTER'."""
        self.mode = mode
        if mode == 'UNTRAINED':
            self.epsilon = 0.90 # High randomness & chaotic behavior
        elif mode == 'TRAINING':
            self.epsilon = max(self.epsilon_min, self.epsilon)
        elif mode == 'MASTER':
            self.epsilon = 0.0 # Pure greedy exploitation

    def select_action(self, state, evaluate=False):
        """
        Choose action based on epsilon-greedy policy or neural net.
        """
        if self.mode == 'UNTRAINED':
            if random.random() < 0.65:
                return random.choice([1, 2, 0]) # accelerate / full throttle
            return random.randint(0, self.num_actions - 1)

        # Exploration in training mode
        if not evaluate and self.mode == 'TRAINING' and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Greedy choice from Q-network
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=-1).item())

    def get_q_values(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return q_values.squeeze().cpu().numpy()

    def get_activations(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q_network.get_layer_activations(state_t)

    def store_transition(self, state, action, reward, next_state, done):
        if self.mode == 'TRAINING':
            with self.lock:
                self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform Double DQN training step on a minibatch from Prioritized Replay Buffer.
        """
        if self.mode != 'TRAINING':
            return None

        # Anneal beta towards 1.0 over ~100k train steps
        self.beta = min(1.0, 0.4 + (self.train_step_count / 100000.0) * 0.6)

        with self.lock:
            if len(self.memory) < RL_BATCH_SIZE:
                return None
            batch = self.memory.sample(RL_BATCH_SIZE, beta=self.beta)
            
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, indices, weights = batch

        s_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        w_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q(s, a)
        curr_q = self.q_network(s_t).gather(1, a_t)

        # Double DQN target computation:
        with torch.no_grad():
            best_actions = self.q_network(ns_t).argmax(dim=1, keepdim=True)
            next_q = self.target_network(ns_t).gather(1, best_actions)
            gamma_macro = RL_GAMMA ** ACTION_REPEAT
            target_q = r_t + (1.0 - d_t) * gamma_macro * next_q

        # Weighted Huber Loss for PER
        td_errors = np.atleast_1d((curr_q - target_q).detach().squeeze().cpu().numpy())
        with self.lock:
            self.memory.update_priorities(indices, td_errors)

        loss_unweighted = nn.functional.smooth_l1_loss(curr_q, target_q, reduction='none')
        loss = (loss_unweighted * w_t).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Dynamic Gradient Norm computation & Adaptive Clipping
        total_norm = nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.last_grad_norm = float(total_norm.item()) if hasattr(total_norm, 'item') else float(total_norm)

        # Auto-tune gradient clipping threshold based on gradient dynamics
        if self.last_grad_norm > self.grad_clip_norm * 1.5:
            self.grad_clip_norm = max(0.5, self.grad_clip_norm * 0.95)
            self.optimizer_status = "Guarded (Volatile)"
        elif self.last_grad_norm < self.grad_clip_norm * 0.3 and self.grad_clip_norm < 3.0:
            self.grad_clip_norm = min(3.0, self.grad_clip_norm * 1.05)
            self.optimizer_status = "Optimal"

        self.optimizer.step()

        self.train_step_count += 1
        self.steps_done += 1
        loss_val = float(loss.item())
        self.recent_losses.append(loss_val)
        self.avg_loss = float(np.mean(self.recent_losses))

        # Real-time Auto-Scheduler step every 20 gradient steps
        if self.train_step_count % 20 == 0:
            self.scheduler.step(self.avg_loss)
            self.current_lr = float(self.optimizer.param_groups[0]['lr'])

        # Auto-recover LR if loss is exceptionally stable and low
        if self.train_step_count % 250 == 0 and len(self.recent_losses) >= 50:
            loss_std = float(np.std(self.recent_losses))
            if loss_std < 0.05 and self.avg_loss < 0.2 and self.current_lr < self.max_lr * 0.7:
                self.current_lr = min(self.max_lr, self.current_lr * 1.1)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.current_lr
                self.optimizer_status = "Accelerated"

        # Soft Target network update (Polyak Averaging) for stability
        self.update_target_network(tau=0.005)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss_val

    # ------------------------------------------------------------------
    # Punishment function
    # ------------------------------------------------------------------
    def calculate_penalty(self, vehicle, traffic_light_state, dist_to_stop, dt=1.0 / 60.0):
        """
        All negative shaping for one vehicle on one physics frame.

        Kept separate from the reward so violation weights can be tuned and
        inspected on their own. Returns (total, breakdown) where breakdown maps
        a violation name to the penalty it contributed this frame.

        Dense terms are declared per second in config and scaled by dt here, so
        the balance no longer shifts with the frame rate or the action repeat.
        """
        pen = 0.0
        parts = {}

        def add(name, value):
            nonlocal pen
            if value:
                pen += value
                parts[name] = parts.get(name, 0.0) + value

        # --- Terminal: collisions ---
        if vehicle.has_crashed:
            if getattr(vehicle, 'hit_pedestrian', False):
                add('pedestrian_hit', PENALTY_CRASH_PEDESTRIAN)
            elif getattr(vehicle, 'is_at_fault', True):
                add('at_fault_crash', PENALTY_CRASH)
            # A vehicle struck from behind while standing still is blameless.
            return pen, parts

        is_red = traffic_light_state in ('RED', 'YELLOW')
        approaching = dist_to_stop is not None and -5.0 <= dist_to_stop <= 90.0
        held_at_red = is_red and approaching and vehicle.speed < 0.3

        # --- Dense: cost of time, waived while waiting properly at a red ---
        if not held_at_red:
            add('time', PENALTY_TIME * dt)

        # --- Event: failed to stop behind the line ---
        if getattr(vehicle, 'stop_line_overrun', False) and not vehicle.penalized_overrun:
            vehicle.penalized_overrun = True
            add('stop_line_overrun', PENALTY_STOP_LINE_OVERRUN)

        # --- Dense: throttle commanded while being held at a red light ---
        if getattr(vehicle, 'red_light_intent', False):
            add('red_creep', PENALTY_RED_CREEP * dt)

        # --- Event: crossed the junction on red or yellow ---
        if vehicle.has_passed_intersection and not vehicle.passed_reward_granted:
            entry = getattr(vehicle, 'tl_state_at_entry', traffic_light_state)
            if entry in ('RED', 'YELLOW'):
                add('red_light_run', PENALTY_RED_LIGHT_RUN)

        # --- Event: entered the roundabout into circulating traffic ---
        if getattr(vehicle, 'yield_violated', False) and not vehicle.penalized_yield_violation:
            vehicle.penalized_yield_violation = True
            add('yield_violation', PENALTY_YIELD_VIOLATION)

        # --- Dense: headway and time-to-collision, both graded not cliff-edged ---
        clearance = getattr(vehicle, 'last_lead_clearance', None)
        if clearance is not None and vehicle.speed > 0.4:
            safe_gap = max(MIN_SAFE_HEADWAY, vehicle.speed * SAFE_HEADWAY_FACTOR)
            if clearance < safe_gap:
                severity = min(1.0, (safe_gap - clearance) / safe_gap)
                add('tailgate', PENALTY_TAILGATE * severity * dt)

        ttc = getattr(vehicle, 'live_ttc', 99.0)
        if ttc < SAFE_TTC_SECONDS and vehicle.speed > 0.4:
            severity = min(1.0, (SAFE_TTC_SECONDS - ttc) / SAFE_TTC_SECONDS)
            add('low_ttc', PENALTY_TTC * (severity ** 2) * dt)

        # --- Dense: pedestrian in the path at speed ---
        if getattr(vehicle, 'last_lead_is_pedestrian', False) and clearance is not None:
            if clearance < 70.0 and vehicle.speed > 1.2:
                severity = min(1.0, (70.0 - clearance) / 70.0)
                add('pedestrian_risk', PENALTY_PED_PROXIMITY * severity * dt)

        # --- Dense: exceeding the vehicle's own speed limit ---
        limit = max(0.5, vehicle.target_speed * SPEED_LIMIT_TOLERANCE)
        if vehicle.speed > limit:
            over = min(1.0, (vehicle.speed - limit) / limit)
            add('speeding', PENALTY_SPEEDING * over * dt)

        # --- Dense: sitting still inside the conflict box, i.e. grid-lock ---
        if getattr(vehicle, 'time_stalled', 0.0) > 0.6 and not held_at_red:
            if getattr(vehicle, 'in_conflict_box', False):
                add('blocking_box', PENALTY_BLOCK_BOX * dt)

        # --- Dense: frozen on a green light with the road ahead clear ---
        road_clear = clearance is None or clearance > 45.0
        if (traffic_light_state in ('GREEN', 'NONE') and vehicle.speed < 0.4
                and road_clear and getattr(vehicle, 'time_stalled', 0.0) > 0.8):
            add('stall', PENALTY_STALL * dt)

        # --- Event: obstructing an emergency vehicle ---
        if not getattr(vehicle, 'is_emergency', False) and vehicle.speed > 3.2:
            if not getattr(vehicle, 'penalized_block_emergency', False):
                hits = getattr(getattr(vehicle, 'sensors', None), 'ray_hits', None)
                if hits:
                    for hit in hits:
                        other = hit[2] if len(hit) > 2 else None
                        if other is not None and getattr(other, 'is_emergency', False) and hit[1] < 50.0:
                            vehicle.penalized_block_emergency = True
                            add('block_emergency', PENALTY_BLOCK_EMERGENCY)
                            break

        return pen, parts

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    def calculate_reward(self, vehicle, traffic_light_state, dist_to_stop, dt=1.0 / 60.0):
        """
        Step reward for one vehicle: positive shaping plus calculate_penalty.
        """
        # The roundabout routes carry a sentinel stop line, treat it as absent.
        if dist_to_stop is not None and dist_to_stop > 4000:
            dist_to_stop = None

        penalty, parts = self.calculate_penalty(vehicle, traffic_light_state, dist_to_stop, dt)
        for name in parts:
            self.violation_counts[name] = self.violation_counts.get(name, 0) + 1

        if vehicle.has_crashed:
            self.last_penalty = penalty
            return penalty

        reward = penalty

        is_red = traffic_light_state in ('RED', 'YELLOW')
        approaching = dist_to_stop is not None and -5.0 <= dist_to_stop <= 90.0

        # --- Dense: forward progress, normalized by this vehicle's own limit ---
        if vehicle.speed > 0.1:
            reward += REWARD_PROGRESS * (vehicle.speed / max(0.5, vehicle.target_speed)) * dt

        # --- Dense: small credit for holding still at a red ---
        if is_red and approaching and vehicle.speed < 0.3:
            reward += REWARD_IDLE_RED * dt

        # --- Event: accel/brake reversal between successive macro-actions ---
        prev = getattr(vehicle, 'prev_action_name', vehicle.action_name)
        if prev != vehicle.action_name:
            if ("ACCEL" in vehicle.action_name and "BRAKE" in prev) or \
               ("BRAKE" in vehicle.action_name and "ACCEL" in prev):
                reward += PENALTY_JERK
        vehicle.prev_action_name = vehicle.action_name

        # --- Event: came to rest behind the stop line on red ---
        if is_red and approaching and vehicle.speed < 0.3 and dist_to_stop is not None \
                and dist_to_stop < 35.0 and not vehicle.rewarded_for_stop:
            vehicle.rewarded_for_stop = True
            reward += REWARD_SMOOTH_STOP_RED

        # --- Event: cleared the signalized junction legally ---
        if vehicle.has_passed_intersection and not vehicle.passed_reward_granted:
            vehicle.passed_reward_granted = True
            entry = getattr(vehicle, 'tl_state_at_entry', traffic_light_state)
            if entry not in ('RED', 'YELLOW'):
                reward += REWARD_PASS_EVENT

        # --- Event: completed the roundabout manoeuvre ---
        if getattr(vehicle, 'roundabout_cleared', False) and not vehicle.rewarded_roundabout_clear:
            vehicle.rewarded_roundabout_clear = True
            if not getattr(vehicle, 'yield_violated', False):
                reward += REWARD_ROUNDABOUT_CLEAR

        if getattr(vehicle, 'gave_roundabout_yield', False) and not vehicle.rewarded_roundabout:
            vehicle.rewarded_roundabout = True
            reward += REWARD_ROUNDABOUT_COURTESY

        # --- Event: reached the end of the route alive. This is the terminal
        # success signal, and the only one a roundabout-only route can earn.
        if vehicle.has_finished and not vehicle.rewarded_route_complete:
            vehicle.rewarded_route_complete = True
            reward += REWARD_ROUTE_COMPLETE

        # --- Event: pulled over for an emergency vehicle ---
        if getattr(vehicle, 'yielded_to_emergency', False) and vehicle.speed < 1.2:
            if not vehicle.rewarded_emergency_yield:
                vehicle.rewarded_emergency_yield = True
                reward += REWARD_YIELD_EMERGENCY

        self.last_penalty = penalty
        return reward

    def save_weights(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'state_dim': self.state_dim,
            'num_actions': self.num_actions
        }, filepath)

    def load_weights(self, filepath):
        if not os.path.exists(filepath):
            return False
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        saved_dim = checkpoint.get('state_dim')
        if saved_dim is not None and saved_dim != self.state_dim:
            print(f"[AI] Checkpoint was trained on {saved_dim}-dim observations, "
                  f"this build uses {self.state_dim}. Starting fresh.")
            return False
        try:
            self.q_network.load_state_dict(checkpoint['q_network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.steps_done = checkpoint.get('steps_done', 0)
        except RuntimeError:
            print("[AI] Error loading weights, likely a dimension mismatch "
                  "(e.g. added sensors). Starting fresh.")
            return False
        loaded_eps = checkpoint.get('epsilon', self.epsilon_min)
        # Clamp: if the saved epsilon is lower than new minimum, boost it back up
        self.epsilon = max(loaded_eps, self.epsilon_min)
        return True
