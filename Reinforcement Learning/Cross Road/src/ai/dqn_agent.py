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
    REWARD_CRASH, REWARD_RED_LIGHT_RUN, REWARD_SMOOTH_STOP_RED, REWARD_IDLE_RED,
    REWARD_PASS_EVENT, REWARD_PROGRESS, REWARD_TIME_PENALTY, REWARD_JERK_PENALTY
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
        td_errors = (curr_q - target_q).detach().squeeze().cpu().numpy()
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

    def calculate_reward(self, vehicle, traffic_light_state, dist_to_stop):
        """
        Calculate precise step reward for a vehicle given current state.
        """
        if vehicle.has_crashed:
            # Innocent vehicle hit by another vehicle is not penalized
            if not getattr(vehicle, 'is_at_fault', True):
                return 0.0
            return REWARD_CRASH

        reward = 0.0

        # Traffic light rules
        is_red_or_yellow = traffic_light_state in ('RED', 'YELLOW')
        is_approaching_stop = dist_to_stop is not None and -5.0 <= dist_to_stop <= 90.0
        is_stopped_at_red = is_red_or_yellow and is_approaching_stop and vehicle.speed < 0.3

        # Base time penalty: waive penalty and award positive idle reward if stopped at red light
        if is_stopped_at_red:
            reward += REWARD_IDLE_RED
        else:
            reward += REWARD_TIME_PENALTY

        # Progress reward based on safe forward movement
        if vehicle.speed > 0.1:
            reward += REWARD_PROGRESS * (vehicle.speed / 5.0)

        # Jerk Penalty: penalize sudden shifts in acceleration/braking
        if hasattr(vehicle, 'prev_action_name') and vehicle.prev_action_name != vehicle.action_name:
            if "ACCEL" in vehicle.action_name and "BRAKE" in vehicle.prev_action_name:
                reward += REWARD_JERK_PENALTY
            elif "BRAKE" in vehicle.action_name and "ACCEL" in vehicle.prev_action_name:
                reward += REWARD_JERK_PENALTY

        vehicle.prev_action_name = vehicle.action_name

        # Tailgating penalty if too close to vehicle ahead
        min_front_dist = 160.0
        if hasattr(vehicle, 'sensors') and vehicle.sensors.ray_hits:
            for idx in (3, 4, 5):
                if idx < len(vehicle.sensors.ray_hits):
                    min_front_dist = min(min_front_dist, vehicle.sensors.ray_hits[idx][1])
        if min_front_dist < 32.0 and vehicle.speed > 1.5:
            reward -= 3.0

        if is_red_or_yellow and is_approaching_stop:
            if vehicle.speed < 0.3 and dist_to_stop < 35.0:
                # Great job stopping smoothly behind red light
                if not getattr(vehicle, 'rewarded_for_stop', False):
                    reward += REWARD_SMOOTH_STOP_RED
                    vehicle.rewarded_for_stop = True
            elif vehicle.speed > 2.2 and dist_to_stop < 30.0:
                # Approaching red light too fast
                reward -= 3.0
            elif vehicle.speed > 3.5 and dist_to_stop < 60.0:
                reward -= 1.5
        elif not vehicle.has_passed_intersection and traffic_light_state == 'GREEN' and vehicle.speed < 0.5 and min_front_dist > 40.0:
            # Way is clear (green light), but vehicle is cowardly stopping
            reward -= 1.5

        # One-time event bonus for passing the intersection
        if vehicle.has_passed_intersection and not getattr(vehicle, 'passed_reward_granted', False):
            vehicle.passed_reward_granted = True
            entry_state = getattr(vehicle, 'tl_state_at_entry', traffic_light_state)
            if entry_state not in ('RED', 'YELLOW'):
                reward += REWARD_PASS_EVENT
            else:
                reward += REWARD_RED_LIGHT_RUN # Severe penalty (-100.0) and pass reward is 0

        return reward

    def save_weights(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)

    def load_weights(self, filepath):
        if not os.path.exists(filepath):
            return False
        checkpoint = torch.load(filepath, map_location=self.device)
        try:
            self.q_network.load_state_dict(checkpoint['q_network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.steps_done = checkpoint.get('steps_done', 0)
        except RuntimeError as e:
            print(f"[AI] Error loading weights, likely dimension mismatch (e.g., added sensors). Starting fresh.")
            return False
        loaded_eps = checkpoint.get('epsilon', self.epsilon_min)
        # Clamp: if the saved epsilon is lower than new minimum, boost it back up
        self.epsilon = max(loaded_eps, self.epsilon_min)
        return True
