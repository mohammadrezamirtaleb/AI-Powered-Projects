import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarker
mp_hands_options = mp.tasks.vision.HandLandmarkerOptions
mp_running_mode = mp.tasks.vision.RunningMode
mp_image = mp.Image


class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", max_hands=2):
        options = mp_hands_options(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_running_mode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_hands.create_from_options(options)

    def find_hands(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        detected = []
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                h, w, _ = img.shape
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                label = handedness[0].category_name
                fingers = self._count_fingers(pts, label)
                detected.append({
                    "landmarks": pts,
                    "label": label,
                    "fingers": fingers,
                    "score": handedness[0].score,
                })
        return detected

    def draw(self, img, detected):
        h, w, _ = img.shape
        for hand in detected:
            pts = hand["landmarks"]
            label = hand["label"]
            fingers = hand["fingers"]

            for i, (x, y) in enumerate(pts):
                cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17),
            ]
            for i, j in connections:
                if i < len(pts) and j < len(pts):
                    cv2.line(img, pts[i], pts[j], (255, 0, 0), 2)

            cx, cy = pts[0]
            (tw, th), _ = cv2.getTextSize(f"{label} Hand", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = max(0, min(cx - tw // 2, w - tw))
            ty = max(th + 10, min(cy - 30, h - 10))
            cv2.putText(img, f"{label} Hand", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        return img

    def _count_fingers(self, pts, label):
        if len(pts) < 21:
            return 0

        fingers = []

        if label == "Right":
            fingers.append(1 if pts[4][0] > pts[3][0] else 0)
        else:
            fingers.append(1 if pts[4][0] < pts[3][0] else 0)

        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(1 if pts[tip][1] < pts[pip][1] else 0)

        return sum(fingers)
