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
    VISION_STATE_SIZE, NUM_ACTIONS, RL_GAMMA, RL_LR,
    RL_BATCH_SIZE, RL_BUFFER_CAPACITY, RL_TARGET_UPDATE_FREQ,
    RL_EPSILON_START, RL_EPSILON_MIN, RL_EPSILON_DECAY,
    REWARD_CRASH, REWARD_RED_LIGHT_RUN, REWARD_SMOOTH_STOP_RED,
    REWARD_PASS_EVENT, REWARD_PROGRESS, REWARD_TIME_PENALTY, REWARD_JERK_PENALTY
)
from src.ai.network import DuelingDQN

class PrioritizedReplayBuffer:
    def __init__(self, capacity=RL_BUFFER_CAPACITY, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        if n == 0:
            return None
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        sum_p = probs.sum()
        if sum_p <= 0:
            probs = np.ones(n, dtype=np.float32) / n
        else:
            probs /= sum_p

        indices = np.random.choice(n, batch_size, p=probs, replace=(n < batch_size))
        samples = [self.buffer[idx] for idx in indices]

        weights = (n * probs[indices]) ** (-beta)
        max_w = weights.max()
        if max_w > 0:
            weights /= max_w
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
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
            p = float(abs(err) + 1e-5)
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.state_dim = VISION_STATE_SIZE
        self.num_actions = NUM_ACTIONS

        # Networks
        self.q_network = DuelingDQN(self.state_dim, self.num_actions).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.num_actions).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=RL_LR)
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
            self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform Double DQN training step on a minibatch from Prioritized Replay Buffer.
        """
        if self.mode != 'TRAINING' or len(self.memory) < RL_BATCH_SIZE:
            return None

        # Anneal beta towards 1.0 over ~100k train steps
        self.beta = min(1.0, 0.4 + (self.train_step_count / 100000.0) * 0.6)

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
            target_q = r_t + (1.0 - d_t) * RL_GAMMA * next_q

        # Weighted Huber Loss for PER
        td_errors = (curr_q - target_q).detach().squeeze().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss_unweighted = nn.functional.smooth_l1_loss(curr_q, target_q, reduction='none')
        loss = (loss_unweighted * w_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
        self.optimizer.step()

        self.train_step_count += 1
        loss_val = float(loss.item())
        self.recent_losses.append(loss_val)
        self.avg_loss = float(np.mean(self.recent_losses))

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
            return REWARD_CRASH if getattr(vehicle, 'is_at_fault', True) else -5.0

        reward = 0.0

        # Small base time penalty
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

        # Traffic light rules
        is_red_or_yellow = traffic_light_state in ('RED', 'YELLOW')
        is_approaching_stop = dist_to_stop is not None and 0 <= dist_to_stop <= 90.0

        if is_red_or_yellow and is_approaching_stop:
            if vehicle.speed < 0.3 and dist_to_stop < 35.0:
                # Great job stopping smoothly behind red light
                reward += REWARD_SMOOTH_STOP_RED * 0.15
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
                reward += REWARD_RED_LIGHT_RUN

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
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        return True
