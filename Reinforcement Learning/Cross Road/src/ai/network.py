"""
Situation-gated Dueling DQN.

Two encoders read different parts of the stacked observation:

  * LiDAR stream  — range and closing speed, the reactive safety channel
  * Semantic stream — signals, yield, leader intent, and the 14-D situation
    slice (where we are, whether we may go, what the car ahead is doing)

A sigmoid gate from the *latest* situation slice scales the semantic embedding
so the value head can treat "stopped at a red" and "frozen in a clear ring"
as different states instead of one blob of low speed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import (
    STACKED_STATE_SIZE, NUM_ACTIONS, RL_HIDDEN_DIM, RL_TRUNK_DIM, RL_STREAM_DIM,
    VISION_STATE_SIZE, FRAME_STACK_SIZE, LIDAR_FEATURE_SIZE
)


def _init_linear(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class DuelingDQN(nn.Module):
    def __init__(self, state_dim=STACKED_STATE_SIZE, num_actions=NUM_ACTIONS,
                 hidden_dim=RL_HIDDEN_DIM, trunk_dim=RL_TRUNK_DIM, stream_dim=RL_STREAM_DIM):
        super(DuelingDQN, self).__init__()
        self.vision_size = VISION_STATE_SIZE
        self.stack = FRAME_STACK_SIZE
        self.lidar_per_frame = LIDAR_FEATURE_SIZE
        self.sem_per_frame = VISION_STATE_SIZE - LIDAR_FEATURE_SIZE
        self.lidar_dim = self.lidar_per_frame * FRAME_STACK_SIZE
        self.sem_dim = self.sem_per_frame * FRAME_STACK_SIZE

        self.lidar1 = nn.Linear(self.lidar_dim, 128)
        self.lidar_ln1 = nn.LayerNorm(128)
        self.lidar2 = nn.Linear(128, 96)
        self.lidar_ln2 = nn.LayerNorm(96)

        self.sem1 = nn.Linear(self.sem_dim, hidden_dim)
        self.sem_ln1 = nn.LayerNorm(hidden_dim)
        self.sem2 = nn.Linear(hidden_dim, 96)
        self.sem_ln2 = nn.LayerNorm(96)

        self.gate = nn.Linear(self.sem_per_frame, 96)

        self.fuse = nn.Linear(192, trunk_dim)
        self.fuse_ln = nn.LayerNorm(trunk_dim)

        self.val_dense1 = nn.Linear(trunk_dim, stream_dim)
        self.val_dense2 = nn.Linear(stream_dim, stream_dim // 2)
        self.val_out = nn.Linear(stream_dim // 2, 1)

        self.adv_dense1 = nn.Linear(trunk_dim, stream_dim)
        self.adv_dense2 = nn.Linear(stream_dim, stream_dim // 2)
        self.adv_out = nn.Linear(stream_dim // 2, num_actions)

        self._reset_parameters()

    def _reset_parameters(self):
        root_two = 2.0 ** 0.5
        for layer in (self.lidar1, self.lidar2, self.sem1, self.sem2, self.fuse,
                      self.val_dense1, self.val_dense2,
                      self.adv_dense1, self.adv_dense2):
            _init_linear(layer, root_two)
        _init_linear(self.gate, 1.0)
        _init_linear(self.val_out, 1.0)
        _init_linear(self.adv_out, 0.01)

    def _split(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        frames = state.view(batch, self.stack, self.vision_size)
        lidar = frames[:, :, :self.lidar_per_frame].reshape(batch, -1)
        sem = frames[:, :, self.lidar_per_frame:].reshape(batch, -1)
        latest_sem = frames[:, -1, self.lidar_per_frame:]
        return lidar, sem, latest_sem

    def _trunk(self, state):
        lidar, sem, latest_sem = self._split(state)
        h_l = F.silu(self.lidar_ln1(self.lidar1(lidar)))
        h_l = F.silu(self.lidar_ln2(self.lidar2(h_l)))

        h_s = F.silu(self.sem_ln1(self.sem1(sem)))
        h_s = F.silu(self.sem_ln2(self.sem2(h_s)))
        gate = torch.sigmoid(self.gate(latest_sem))
        h_s = h_s * (0.35 + 0.65 * gate)

        h3 = F.silu(self.fuse_ln(self.fuse(torch.cat([h_l, h_s], dim=-1))))
        return h_l, h_s, h3

    def _heads(self, h3):
        v = F.silu(self.val_dense1(h3))
        v2 = F.silu(self.val_dense2(v))
        values = self.val_out(v2)

        a = F.silu(self.adv_dense1(h3))
        a2 = F.silu(self.adv_dense2(a))
        advantages = self.adv_out(a2)
        return values, advantages, v2, a2

    def forward(self, state):
        _, _, h3 = self._trunk(state)
        values, advantages, _, _ = self._heads(h3)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))

    def get_layer_activations(self, state_tensor):
        """
        Returns internal layer activations across all deep layers for the
        real-time HUD visualizer.
        """
        with torch.no_grad():
            h1, h2, h3 = self._trunk(state_tensor)
            values, advantages, v2, a2 = self._heads(h3)
            q = values + (advantages - advantages.mean(dim=-1, keepdim=True))

            h1_s = h1.squeeze()
            h2_s = h2.squeeze()
            h3_s = h3.squeeze()
            v2_s = v2.squeeze()
            a2_s = a2.squeeze()
            stream_s = torch.cat([v2_s[:4], a2_s[:4]], dim=0) if v2_s.dim() > 0 else torch.zeros(8)

            return {
                'inputs': state_tensor.squeeze().cpu().numpy(),
                'h1': h1_s[:10].cpu().numpy(),
                'h2': h2_s[:10].cpu().numpy(),
                'h3': h3_s[:10].cpu().numpy(),
                'stream': stream_s.cpu().numpy(),
                'q_values': q.squeeze().cpu().numpy()
            }
