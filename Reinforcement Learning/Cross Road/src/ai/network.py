"""
PyTorch Dueling Deep Q-Network (Dueling DQN) Architecture.
Separates State-Value V(s) stream and Advantage A(s, a) stream for stable policy evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=29, num_actions=5, hidden_dim=128):
        super(DuelingDQN, self).__init__()

        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        """
        Forward pass computing Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        """
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    def get_layer_activations(self, state_tensor):
        """
        Returns internal layer activations for real-time Neural Network HUD visualizer.
        """
        with torch.no_grad():
            h1 = F.relu(self.feature_layer[1](self.feature_layer[0](state_tensor)))
            h2 = F.relu(self.feature_layer[4](self.feature_layer[3](h1)))
            val = self.value_stream(h2)
            adv = self.advantage_stream(h2)
            q = val + (adv - adv.mean(dim=-1, keepdim=True))
            return {
                'inputs': state_tensor.squeeze().cpu().numpy(),
                'h1': h1.squeeze()[:16].cpu().numpy(), # Sample first 16 neurons for clean HUD display
                'h2': h2.squeeze()[:16].cpu().numpy(),
                'q_values': q.squeeze().cpu().numpy()
            }
