import cv2
import numpy as np
import math

from hand_tracker import HandTracker
from volume_controller import VolumeController


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tracker = HandTracker()
    vol_ctrl = VolumeController()

    smooth_vol = 50
    SMOOTH_FACTOR = 0.15

    print("Controls: Show 0-5 fingers to control volume.")
    print("Press 'q' to quit.")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        img = cv2.flip(img, 1)
        detected = tracker.find_hands(img)

        # We still draw the hand skeleton for visualization
        img = tracker.draw(img, detected)

        if detected:
            hand = detected[0]
            lmList = hand["landmarks"]
            
            if len(lmList) >= 9:
                x1, y1 = lmList[4]  # Thumb tip
                x2, y2 = lmList[8]  # Index tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), -1)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                length = math.hypot(x2 - x1, y2 - y1)
                
                # Map length (approx 20 to 200 pixels) to volume
                target_vol = np.interp(length, [20, 200], [0, 100])
                smooth_vol = smooth_vol * (1 - SMOOTH_FACTOR) + target_vol * SMOOTH_FACTOR
                vol_ctrl.set_volume_percent(smooth_vol)
                
                if length < 30:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)
                elif length > 180:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
                else:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), -1)

        current_vol = vol_ctrl.get_volume_percent()
        bar_x, bar_y, bar_w, bar_h = 50, 150, 30, 300
        # For a vertical bar, fill_h goes up
        fill_h = int(np.interp(current_vol, [0, 100], [0, bar_h]))

        # Draw the background of the bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        
        # Determine color based on volume (Green -> Yellow -> Red)
        if current_vol < 50:
            color = (0, 255, 0) # Green
        elif current_vol < 80:
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 0, 255) # Red

        # Draw the filled part
        cv2.rectangle(img, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), color, -1)
        
        # Put volume text
        cv2.putText(
            img,
            f"{int(current_vol)}%",
            (bar_x - 10, bar_y + bar_h + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3,
        )

        cv2.imshow("Finger Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
