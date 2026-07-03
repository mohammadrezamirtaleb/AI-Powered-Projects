#!/usr/bin/env python3
"""
FaceVision Pro — Single Image Analysis
Usage: python scripts/run_image.py --image photo.jpg [--output outputs/result.jpg]
"""

import argparse
import logging
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceVision Pro — Analyze a Single Image",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path for annotated output image (default: outputs/annotated_<name>.jpg)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display result in a window after processing",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a custom YAML config file",
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
    )

    if not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        raise SystemExit(1)

    from facevision.utils.config import Config
    from facevision.pipeline.batch import BatchPipeline

    cfg = Config(config_path=args.config)
    # Disable cooldown for single-image analysis
    cfg.set("analyzer.cooldown_seconds", 0)

    out_dir = args.output.parent if args.output else Path("outputs/batch")
    pipeline = BatchPipeline(config=cfg, output_dir=out_dir)

    print(f"Analyzing: {args.image}")
    result = pipeline.process_image(args.image, save=True)

    if result is None:
        print("[ERROR] Failed to process image.")
        raise SystemExit(1)

    if args.output:
        from facevision.utils.io import ImageIO
        ImageIO.write_image(result, args.output)
        print(f"Saved to: {args.output}")

    if args.show:
        import cv2
        cv2.imshow("FaceVision Pro — Result", result)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Done ✓")


if __name__ == "__main__":
    main()
