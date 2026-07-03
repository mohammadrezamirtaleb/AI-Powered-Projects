#!/usr/bin/env python3
"""
FaceVision Pro — Face Enrollment Script
=========================================
Enroll your face (or others') into the recognition database.

Usage:
    # Enroll from webcam:
    python scripts/enroll_face.py --name "Alice" --camera 0

    # Enroll from image file(s):
    python scripts/enroll_face.py --name "Bob" --images photos/bob1.jpg photos/bob2.jpg

    # Enroll entire directory (folder name = person name):
    python scripts/enroll_face.py --from-dir data/known_faces/

    # List enrolled people:
    python scripts/enroll_face.py --list

    # Remove a person:
    python scripts/enroll_face.py --remove "Alice"
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceVision Pro — Face Enrollment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", "-n", type=str, help="Person's name to enroll")
    group.add_argument("--from-dir", type=Path, metavar="DIR",
                       help="Auto-enroll from directory (subfolder = name)")
    group.add_argument("--list", action="store_true", help="List all enrolled names")
    group.add_argument("--remove", type=str, metavar="NAME", help="Remove a person from database")

    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--images", type=Path, nargs="+", help="Image file(s) to enroll from")
    parser.add_argument("--count", type=int, default=5, help="Number of webcam captures (default: 5)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def enroll_from_webcam(recognizer, name: str, camera: int, count: int) -> int:
    """Capture N photos from webcam and enroll them."""
    import cv2
    from facevision.core.detector import FaceDetector

    detector = FaceDetector(method="dnn")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera)
        return 0

    enrolled = 0
    print(f"\nLook at the camera. Capturing {count} frames for '{name}' …")
    print("Press SPACE to capture, Q to quit.\n")

    cv2.namedWindow("Enrollment — FaceVision Pro", cv2.WINDOW_NORMAL)

    while enrolled < count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        faces = detector.detect(frame)

        for face in faces:
            cv2.rectangle(frame, (face.x, face.y), (face.x2, face.y2), (0, 220, 100), 2)

        status = f"Enrolled: {enrolled}/{count}  |  Name: {name}  |  SPACE=capture  Q=quit"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow("Enrollment — FaceVision Pro", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and faces:
            crop = faces[0].crop(frame)
            if recognizer.enroll(name, crop):
                enrolled += 1
                print(f"  ✓ Captured {enrolled}/{count}")
                time.sleep(0.3)
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return enrolled


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from facevision.core.recognizer import FaceRecognizer

    recognizer = FaceRecognizer()
    recognizer.load_database()

    # ── List
    if args.list:
        names = recognizer.enrolled_names
        print(f"\n📋 Enrolled faces ({recognizer.database_size} total encodings):")
        if not names:
            print("  (none)")
        for name in names:
            count = recognizer._known_names.count(name)
            print(f"  • {name} ({count} encoding{'s' if count != 1 else ''})")
        return

    # ── Remove
    if args.remove:
        removed = recognizer.remove(args.remove)
        print(f"Removed {removed} encoding(s) for '{args.remove}'.")
        recognizer.save_database()
        return

    # ── Auto-enroll from directory
    if args.from_dir:
        count = recognizer.enroll_from_directory(args.from_dir)
        print(f"✓ Enrolled {count} faces from {args.from_dir}")
        recognizer.save_database()
        return

    # ── Enroll by name
    name = args.name
    total_enrolled = 0

    if args.images:
        for img_path in args.images:
            if recognizer.enroll_from_file(name, img_path):
                total_enrolled += 1
                print(f"  ✓ Enrolled from: {img_path}")
    else:
        total_enrolled = enroll_from_webcam(recognizer, name, args.camera, args.count)

    if total_enrolled > 0:
        recognizer.save_database()
        print(f"\n✅ Enrolled {total_enrolled} face encoding(s) for '{name}'.")
        print(f"   Database now contains {recognizer.database_size} total encodings.")
    else:
        print(f"\n⚠ No faces enrolled for '{name}'.")


if __name__ == "__main__":
    main()
