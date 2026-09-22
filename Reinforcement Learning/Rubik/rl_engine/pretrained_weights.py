"""
Astra-DeepCube: Model Initializer & Pretrained Heuristics
Loads smart weights or rapid autodidactic initialization for instant AI solving.
"""

import os
from typing import Optional
from rl_engine.deepcube_model import DeepCubeNetwork, TORCH_AVAILABLE, create_default_model
from rl_engine.autodidactic_iteration import AutodidacticTrainer

if TORCH_AVAILABLE:
    import torch


MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "deepcube_checkpoint.pt")
_GLOBAL_MODEL_CACHE: Optional[DeepCubeNetwork] = None


def get_ready_model(device: str = "cpu", fast_warmup_steps: int = 40) -> DeepCubeNetwork:
    """
    Returns an initialized DeepCube model using in-memory singleton caching.
    """
    global _GLOBAL_MODEL_CACHE
    if _GLOBAL_MODEL_CACHE is not None:
        return _GLOBAL_MODEL_CACHE

    model = create_default_model(device_str=device)
    
    if not TORCH_AVAILABLE:
        _GLOBAL_MODEL_CACHE = model
        return model

    if os.path.exists(MODEL_CACHE_PATH):
        try:
            try:
                ckpt = torch.load(MODEL_CACHE_PATH, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(MODEL_CACHE_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            _GLOBAL_MODEL_CACHE = model
            return model
        except Exception:
            pass

    # Rapid warm-up bootstrap
    trainer = AutodidacticTrainer(model, lr=2e-3, device=device)
    for _ in range(fast_warmup_steps):
        trainer.train_step(batch_size=32, max_scramble_depth=8)

    # Save initial checkpoint
    try:
        torch.save({"model_state": model.state_dict()}, MODEL_CACHE_PATH)
    except Exception:
        pass

    _GLOBAL_MODEL_CACHE = model
    return model
