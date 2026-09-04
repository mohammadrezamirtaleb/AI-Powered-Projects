"""
PyTorch Deep Dueling Deep Q-Network (Deep Dueling DQN) Architecture.
Expanded multi-layer architecture with Dense 1, Dense 2, Dense 3,
and deep State-Value V(s) and Action-Advantage A(s, a) dual streams.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import STACKED_STATE_SIZE, NUM_ACTIONS

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=STACKED_STATE_SIZE, num_actions=NUM_ACTIONS, hidden_dim=256):
        super(DuelingDQN, self).__init__()

        # Deep Multi-Layer Feature Extractor (Dense 1, Dense 2, Dense 3)
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dense3 = nn.Linear(hidden_dim, 128)
        self.ln3 = nn.LayerNorm(128)

        # Deep State-Value Stream V(s)
        self.val_dense1 = nn.Linear(128, 64)
        self.val_dense2 = nn.Linear(64, 32)
        self.val_out = nn.Linear(32, 1)

        # Deep Action-Advantage Stream A(s, a)
        self.adv_dense1 = nn.Linear(128, 64)
        self.adv_dense2 = nn.Linear(64, 32)
        self.adv_out = nn.Linear(32, num_actions)

    def forward(self, state):
        """
        Forward pass computing Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        """
        h1 = F.relu(self.ln1(self.dense1(state)))
        h2 = F.relu(self.ln2(self.dense2(h1)))
        h3 = F.relu(self.ln3(self.dense3(h2)))

        # Value stream
        v1 = F.relu(self.val_dense1(h3))
        v2 = F.relu(self.val_dense2(v1))
        values = self.val_out(v2)

        # Advantage stream
        a1 = F.relu(self.adv_dense1(h3))
        a2 = F.relu(self.adv_dense2(a1))
        advantages = self.adv_out(a2)

        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    def get_layer_activations(self, state_tensor):
        """
        Returns internal layer activations across all deep layers for real-time HUD visualizer.
        """
        with torch.no_grad():
            h1 = F.relu(self.ln1(self.dense1(state_tensor)))
            h2 = F.relu(self.ln2(self.dense2(h1)))
            h3 = F.relu(self.ln3(self.dense3(h2)))
            
            v1 = F.relu(self.val_dense1(h3))
            v2 = F.relu(self.val_dense2(v1))
            val = self.val_out(v2)

            a1 = F.relu(self.adv_dense1(h3))
            a2 = F.relu(self.adv_dense2(a1))
            adv = self.adv_out(a2)

            q = val + (adv - adv.mean(dim=-1, keepdim=True))

            # Sample first N neurons per layer for clean HUD display
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
