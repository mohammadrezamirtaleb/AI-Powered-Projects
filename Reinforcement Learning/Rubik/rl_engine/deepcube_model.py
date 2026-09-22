"""
Astra-DeepCube: Deep Neural Value & Policy Network (DeepCubeA Architecture)
Implements Deep Residual MLP with LayerNorm, ELU activations, and dual Value/Policy heads.
"""

import math
import threading
from typing import Tuple, Dict, Optional, Union, List
import numpy as np


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_device(device_input: Union[str, object]) -> "torch.device":
    """Safely parses device strings (e.g. 'cuda', 'cuda:0', 'cpu') or torch.device."""
    if not TORCH_AVAILABLE:
        return "cpu"
    dev_str = str(device_input).strip()
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev_str)
    return torch.device("cpu")


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """Residual Dense Block with Layer Normalization and ELU activation."""
        def __init__(self, hidden_dim: int, dropout: float = 0.05):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.act(self.ln1(self.fc1(x)))
            out = self.dropout(out)
            out = self.ln2(self.fc2(out))
            return self.act(out + residual)


    class DeepCubeNetwork(nn.Module):
        """
        DeepCubeA Neural Value and Policy Network.
        Estimates:
          1. h(s): Minimum moves to solved state (cost-to-go scalar)
          2. p(a|s): Prior action probability distribution (logits over 12 moves)
        """
        def __init__(self, input_dim: int = 324, hidden_dim: int = 256, num_res_blocks: int = 2, num_actions: int = 12):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_res_blocks = num_res_blocks
            self.num_actions = num_actions
            self.model_lock = threading.RLock()

            # Input Embedding Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU()
            )

            # Residual Backbone
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
            ])

            # Value Head (Cost-to-go scalar)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ELU(),
                nn.Linear(256, 1),
                nn.ReLU()
            )

            # Policy Head (Move probabilities)
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ELU(),
                nn.Linear(256, num_actions)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Input shape: (batch_size, input_dim)
            Returns:
              value: shape (batch_size, 1)
              policy_logits: shape (batch_size, num_actions)
            """
            features = self.input_proj(x)
            for block in self.blocks:
                features = block(features)

            value = self.value_head(features)
            policy_logits = self.policy_head(features)
            return value, policy_logits

        def predict_single(self, state_tensor_np: np.ndarray) -> Tuple[float, np.ndarray]:
            """Thread-safe inference for a single state array."""
            with self.model_lock:
                self.eval()
                with torch.no_grad():
                    device = next(self.parameters()).device
                    # Truncate or pad if dimension is not 324
                    if len(state_tensor_np) != self.input_dim:
                        vec = np.zeros(self.input_dim, dtype=np.float32)
                        copy_len = min(len(state_tensor_np), self.input_dim)
                        vec[:copy_len] = state_tensor_np[:copy_len]
                    else:
                        vec = state_tensor_np

                    inp = torch.from_numpy(vec).float().unsqueeze(0).to(device)
                    val, pol = self.forward(inp)
                    h_val = float(val.item())
                    pol_probs = F.softmax(pol, dim=-1).cpu().numpy()[0]
                    return max(0.0, h_val), pol_probs

        def predict_batch(self, state_tensors_list: List[np.ndarray]) -> Tuple[List[float], List[np.ndarray]]:
            """Thread-safe batched inference for multiple neighbor states (10x-20x speedup)."""
            if not state_tensors_list:
                return [], []

            with self.model_lock:
                self.eval()
                with torch.no_grad():
                    device = next(self.parameters()).device
                    batch_arr = np.zeros((len(state_tensors_list), self.input_dim), dtype=np.float32)
                    for i, st in enumerate(state_tensors_list):
                        copy_len = min(len(st), self.input_dim)
                        batch_arr[i, :copy_len] = st[:copy_len]

                    inp = torch.from_numpy(batch_arr).float().to(device)
                    val, pol = self.forward(inp)
                    
                    h_vals = [max(0.0, float(v.item())) for v in val]
                    pol_probs = F.softmax(pol, dim=-1).cpu().numpy()
                    return h_vals, [p.copy() for p in pol_probs]

else:
    # Pure Python / NumPy fallback
    class DeepCubeNetwork:
        def __init__(self, *args, **kwargs):
            self.num_actions = 12
            self.model_lock = threading.RLock()

        def predict_single(self, state_tensor_np: np.ndarray) -> Tuple[float, np.ndarray]:
            h = float(np.sum(state_tensor_np) % 20)
            p = np.ones(self.num_actions) / self.num_actions
            return h, p

        def predict_batch(self, state_tensors_list: List[np.ndarray]) -> Tuple[List[float], List[np.ndarray]]:
            res_h = [float(np.sum(st) % 20) for st in state_tensors_list]
            res_p = [np.ones(self.num_actions) / self.num_actions for _ in state_tensors_list]
            return res_h, res_p


def create_default_model(device_str: str = "cpu") -> DeepCubeNetwork:
    """Factory helper to build and initialize model on device."""
    if not TORCH_AVAILABLE:
        return DeepCubeNetwork()

    device = parse_device(device_str)
    model = DeepCubeNetwork(input_dim=324, hidden_dim=256, num_res_blocks=2, num_actions=12)
    model.to(device)
    
    # Initialize weights with Xavier uniform
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    return model
