#!/usr/bin/env python3
"""
FaceVision Pro — Real-Time Webcam Demo
Usage: python scripts/run_webcam.py [--camera 0] [--config configs/default.yaml]
"""

import argparse
import logging
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceVision Pro — Real-Time Face Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a custom YAML config file",
    )
    parser.add_argument(
        "--no-recognition", action="store_true",
        help="Disable face recognition",
    )
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="Disable age/gender/emotion analysis (faster)",
    )
    parser.add_argument(
        "--method", choices=["dnn", "haar"], default=None,
        help="Override face detector method",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from facevision.utils.config import Config
    from facevision.pipeline.realtime import RealtimePipeline

    cfg = Config(config_path=args.config)

    if args.no_recognition:
        cfg.set("recognizer.enabled", False)
    if args.no_analysis:
        cfg.set("analyzer.enabled", False)
    if args.method:
        cfg.set("detector.method", args.method)

    pipeline = RealtimePipeline(config=cfg)
    pipeline.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
