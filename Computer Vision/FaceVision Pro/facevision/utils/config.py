"""
Configuration Management
=========================
Loads and validates FaceVision Pro configuration from YAML files
with sensible defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Using default config only.")

DEFAULT_CONFIG: Dict[str, Any] = {
    "detector": {
        "method": "dnn",          # "dnn" or "haar"
        "confidence_threshold": 0.60,
        "min_face_size": [30, 30],
    },
    "landmarks": {
        "enabled": True,
        "max_faces": 5,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "refine_landmarks": True,
    },
    "analyzer": {
        "enabled": True,
        "actions": ["emotion", "age", "gender"],
        "cooldown_seconds": 1.5,
    },
    "recognizer": {
        "enabled": True,
        "tolerance": 0.50,
        "model": "hog",
        "auto_load_database": True,
    },
    "pose": {
        "enabled": True,
        "smoothing": 5,
        "draw_axes": True,
    },
    "pipeline": {
        "camera_index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "flip_horizontal": True,
        "show_landmarks": True,
        "show_emotion_bars": True,
        "record_output": False,
        "output_path": "outputs/recording.mp4",
    },
    "display": {
        "window_title": "FaceVision Pro",
        "fullscreen": False,
        "overlay_alpha": 0.75,
    },
    "logging": {
        "level": "INFO",
        "file": None,
    },
}

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class Config:
    """
    Hierarchical configuration manager with YAML support.

    Loads from the default config, then merges a YAML file on top.

    Args:
        config_path: Path to a YAML config file. Defaults to configs/default.yaml.

    Example:
        >>> cfg = Config()
        >>> method = cfg.get("detector.method")
        >>> cfg.set("pipeline.flip_horizontal", False)
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._data: Dict[str, Any] = _deep_copy(DEFAULT_CONFIG)
        config_path = config_path or (_CONFIG_DIR / "default.yaml")
        if config_path.exists() and _YAML_AVAILABLE:
            self._load_yaml(config_path)

    def _load_yaml(self, path: Path) -> None:
        """Merge YAML file into current config."""
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(self._data, user_cfg)
        logger.info("Loaded config from %s", path)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated key.

        Example:
            >>> cfg.get("detector.method")
            "dnn"
        """
        parts = key.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot-separated key."""
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> Dict[str, Any]:
        """Return an entire config section as a dict."""
        return self._data.get(name, {})

    def save(self, path: Optional[Path] = None) -> None:
        """Save current config to YAML file."""
        if not _YAML_AVAILABLE:
            logger.error("PyYAML not installed. Cannot save config.")
            return
        path = path or (_CONFIG_DIR / "default.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True)
        logger.info("Config saved to %s", path)

    def __repr__(self) -> str:
        return f"Config({self._data})"


def _deep_copy(d: Dict) -> Dict:
    """Deep copy a dictionary."""
    import copy
    return copy.deepcopy(d)


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base dict in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
