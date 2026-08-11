import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import argparse
import time
from typing import List
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections, HandLandmarkerResult

# Constants
SMOOTHING = 0.7
HAND_CONNS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
FINGER_TIPS = {4, 8, 12, 16, 20}
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def download_model_if_missing(model_path: str, model_url: str) -> None:
    """Downloads the required model file if it is not present in the current directory."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Downloading from MediaPipe models...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            print("Please check your internet connection or download the model manually.")
            exit(1)


class HandTracker:
    """A class to encapsulate real-time hand tracking and visualization using MediaPipe."""

    def __init__(self, camera_index: int = 0, num_hands: int = 2, smoothing: float = SMOOTHING):
        self.camera_index = camera_index
        self.num_hands = num_hands
        self.smoothing = smoothing
        self.smoothed_hands: List[List[List[float]]] = []
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam with index {self.camera_index}.")
        
        # Initialize MediaPipe Hand Landmarker
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def draw_virtual_hand(self, img: np.ndarray, landmarks: List[List[float]]) -> None:
        """Draws the virtual hand representation on the provided image array."""
        h, w = img.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

        # Draw connections (bones)
        for i, j in HAND_CONNS:
            cv2.line(img, pts[i], pts[j], (60, 160, 255), 3, cv2.LINE_AA)

        # Draw joints and fingertips
        for i, (x, y) in enumerate(pts):
            if i in FINGER_TIPS:
                color, r = (50, 255, 255), 10
            elif i == 0:
                color, r = (255, 50, 255), 7
            else:
                color, r = (255, 180, 50), 5
            
            # Outer filled circle
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
            # Inner white border for contrast
            cv2.circle(img, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)

    def process_frame(self, frame: np.ndarray, timestamp: int) -> np.ndarray:
        """Processes a single video frame for hand tracking."""
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            result = self.landmarker.detect_for_video(mp_image, timestamp)
        except ValueError:
            result = vision.HandLandmarkerResult([], [], [])
            
        right_panel = np.zeros((h, w, 3), dtype=np.uint8)
        
        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                raw = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                
                if idx >= len(self.smoothed_hands):
                    self.smoothed_hands.append([list(r) for r in raw])
                    smoothed = self.smoothed_hands[-1]
                else:
                    smoothed = self.smoothed_hands[idx]
                    for i in range(21):
                        smoothed[i][0] = self.smoothing * smoothed[i][0] + (1 - self.smoothing) * raw[i][0]
                        smoothed[i][1] = self.smoothing * smoothed[i][1] + (1 - self.smoothing) * raw[i][1]
                        
                self.draw_virtual_hand(right_panel, smoothed)
                
        # Combine real frame and virtual hand frame
        out = np.hstack((frame, right_panel))
        return out, w

    def run(self) -> None:
        """Runs the main tracking loop."""
        window_name = "Hand Tracking - Virtual Hand"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Toggle fullscreen just as before
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        frame_count = 0
        prev_time = time.time()
        fps = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Exiting...")
                break

            frame_count += 1
            # MediaPipe expects monotonically increasing timestamps in milliseconds
            timestamp = int(time.time() * 1000)

            out, w = self.process_frame(frame, timestamp)

            # Calculate FPS
            curr_time = time.time()
            if curr_time - prev_time > 1.0:
                fps = frame_count / (curr_time - prev_time)
                prev_time = curr_time
                frame_count = 0

            # UI Overlays
            # FPS Text
            cv2.putText(out, f"FPS: {int(fps)}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 0), 2, cv2.LINE_AA)
            
            # Panel Labels
            cv2.putText(out, "Webcam View", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(out, "Virtual Hand (NN)", (w + 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2, cv2.LINE_AA)

            # Exit instructions
            cv2.putText(out, "Press 'Q' to Exit", (30, out.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self) -> None:
        """Releases resources."""
        self.landmarker.close()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Hand Tracking & Virtual Hand Visualization")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--hands", type=int, default=2, help="Maximum number of hands to detect (default: 2)")
    parser.add_argument("--smoothing", type=float, default=SMOOTHING, help="Landmark smoothing factor [0.0 - 1.0] (default: 0.7)")
    args = parser.parse_args()

    # Ensure model is available
    download_model_if_missing(MODEL_PATH, MODEL_URL)

    try:
        tracker = HandTracker(camera_index=args.camera, num_hands=args.hands, smoothing=args.smoothing)
        tracker.run()
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
