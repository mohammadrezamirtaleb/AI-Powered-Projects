import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections, HandLandmarkerResult

SMOOTHING = 0.7

HAND_CONNS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
FINGER_TIPS = {4, 8, 12, 16, 20}

MODEL_PATH = "hand_landmarker.task"


def draw_virtual_hand(img, landmarks):
    h, w = img.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

    for i, j in HAND_CONNS:
        cv2.line(img, pts[i], pts[j], (60, 160, 255), 3, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        if i in FINGER_TIPS:
            color, r = (50, 255, 255), 10
        elif i == 0:
            color, r = (255, 50, 255), 7
        else:
            color, r = (255, 180, 50), 5
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found")
        return

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)
    smoothed_hands = []
    frame_count = 0

    cv2.namedWindow("Hand Tracking - Virtual Hand", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Tracking - Virtual Hand", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1
        timestamp = frame_count * 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect_for_video(mp_image, timestamp)
        except ValueError:
            result = vision.HandLandmarkerResult([], [], [])

        right = np.zeros((h, w, 3), dtype=np.uint8)

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                raw = [(lm.x, lm.y, lm.z) for lm in hand_lms]

                if idx >= len(smoothed_hands):
                    smoothed_hands.append([list(r) for r in raw])
                    smoothed = smoothed_hands[-1]
                else:
                    smoothed = smoothed_hands[idx]
                    for i in range(21):
                        smoothed[i][0] = SMOOTHING * smoothed[i][0] + (1 - SMOOTHING) * raw[i][0]
                        smoothed[i][1] = SMOOTHING * smoothed[i][1] + (1 - SMOOTHING) * raw[i][1]

                draw_virtual_hand(right, smoothed)

        out = np.hstack((frame, right))

        cv2.putText(out, "Webcam", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out, "Virtual Hand (NN)", (w + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2)

        cv2.imshow("Hand Tracking - Virtual Hand", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
