#!/usr/bin/env python3
"""
Download Pre-Trained Model Weights
====================================
Downloads the OpenCV DNN face detector model files
(SSD ResNet) from the OpenCV GitHub release.

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "filename": "deploy.prototxt",
        "description": "SSD ResNet face detector prototxt",
    },
    {
        "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "filename": "res10_300x300_ssd_iter_140000.caffemodel",
        "description": "SSD ResNet face detector caffemodel (10 MB)",
    },
]


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file from URL to dest path with a progress indicator."""
    if dest.exists():
        print(f"  [OK] Already exists: {dest.name}")
        return True

    print(f"  [->] Downloading {description} ...")
    try:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = block_num * block_size * 100 / total_size
                mb_done = block_num * block_size / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                sys.stdout.write(f"\r    {min(percent, 100):.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)    ")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest), reporthook)
        print(f"\r    [OK] Saved to: {dest}                    ")
        return True
    except Exception as exc:
        print(f"\n    [FAIL] Failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print("=" * 55)
    print("  FaceVision Pro -- Model Downloader")
    print("=" * 55)
    print(f"  Target directory: {MODELS_DIR}\n")

    success = 0
    for model in MODELS:
        dest = MODELS_DIR / model["filename"]
        if download_file(model["url"], dest, model["description"]):
            success += 1

    print(f"\n{'=' * 55}")
    print(f"  Downloaded {success}/{len(MODELS)} model file(s).")
    if success == len(MODELS):
        print("  [OK] All models ready. DNN detector will now work.")
    else:
        print("  [!]  Some downloads failed. Haar Cascade fallback will be used.")
    print("=" * 55)


if __name__ == "__main__":
    main()
