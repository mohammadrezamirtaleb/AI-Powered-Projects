"""
Real-Time Face Analysis Pipeline
==================================
Orchestrates all FaceVision Pro modules into a single real-time
webcam pipeline with configurable feature flags.

Controls:
    Q / ESC  - Quit
    L        - Toggle landmarks
    P        - Toggle pose axes
    E        - Toggle emotion bars
    R        - Toggle recording
    S        - Save current frame snapshot
    SPACE    - Pause / Resume
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from facevision.core.analyzer import FaceAnalyzer
from facevision.core.detector import FaceDetector
from facevision.core.landmarks import LandmarkDetector
from facevision.core.pose import HeadPoseEstimator
from facevision.core.recognizer import FaceRecognizer
from facevision.utils.config import Config
from facevision.utils.drawing import FaceOverlay
from facevision.utils.io import FPSCounter, VideoCapture, VideoWriter

logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = Path("outputs/snapshots")
RECORDINGS_DIR = Path("outputs/recordings")


class RealtimePipeline:
    """
    End-to-end real-time face analysis pipeline.

    Combines face detection, landmark detection, analysis (emotion/age/gender),
    face recognition, and head pose estimation into a single camera loop.

    Args:
        config: Config instance (loads default.yaml if not provided).

    Example:
        >>> pipeline = RealtimePipeline()
        >>> pipeline.run()   # Blocking call — opens webcam window.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self._setup_logging()

        det_cfg = self.cfg.section("detector")
        lm_cfg = self.cfg.section("landmarks")
        an_cfg = self.cfg.section("analyzer")
        rec_cfg = self.cfg.section("recognizer")
        pose_cfg = self.cfg.section("pose")
        disp_cfg = self.cfg.section("display")

        # ── Modules
        self.detector = FaceDetector(
            method=det_cfg.get("method", "dnn"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.6),
        )
        self.landmark_detector: Optional[LandmarkDetector] = None
        if lm_cfg.get("enabled", True):
            self.landmark_detector = LandmarkDetector(
                max_faces=lm_cfg.get("max_faces", 5),
                min_detection_confidence=lm_cfg.get("min_detection_confidence", 0.5),
                min_tracking_confidence=lm_cfg.get("min_tracking_confidence", 0.5),
                refine_landmarks=lm_cfg.get("refine_landmarks", True),
            )

        self.analyzer: Optional[FaceAnalyzer] = None
        if an_cfg.get("enabled", True):
            self.analyzer = FaceAnalyzer(
                actions=an_cfg.get("actions", ["emotion", "age", "gender"]),
                cooldown_seconds=an_cfg.get("cooldown_seconds", 1.5),
            )

        self.recognizer: Optional[FaceRecognizer] = None
        if rec_cfg.get("enabled", True):
            self.recognizer = FaceRecognizer(
                tolerance=rec_cfg.get("tolerance", 0.50),
                model=rec_cfg.get("model", "hog"),
            )
            if rec_cfg.get("auto_load_database", True):
                self.recognizer.load_database()

        self.pose_estimator: Optional[HeadPoseEstimator] = None
        if pose_cfg.get("enabled", True):
            self.pose_estimator = HeadPoseEstimator(
                smoothing=pose_cfg.get("smoothing", 5)
            )

        self.overlay = FaceOverlay(
            alpha=disp_cfg.get("overlay_alpha", 0.75)
        )
        self.fps_counter = FPSCounter()

        # ── State flags (toggled via keyboard)
        self.show_landmarks = self.cfg.get("pipeline.show_landmarks", True)
        self.show_pose_axes = pose_cfg.get("draw_axes", True)
        self.show_emotion_bars = self.cfg.get("pipeline.show_emotion_bars", True)
        self.paused = False
        self.frame_id = 0
        self._recording = False
        self._video_writer: Optional[VideoWriter] = None

    def _setup_logging(self) -> None:
        log_cfg = self.cfg.section("logging")
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        handlers = [logging.StreamHandler()]
        log_file = log_cfg.get("file")
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full analysis pipeline on a single frame.

        Args:
            frame: BGR image from webcam or video.

        Returns:
            Annotated BGR frame.
        """
        self.frame_id += 1
        h, w = frame.shape[:2]

        # ── Face Detection
        faces = self.detector.detect(frame)

        # ── Per-face analysis
        all_landmarks = []
        if self.landmark_detector and faces:
            all_landmarks = self.landmark_detector.detect(frame)

        for i, face in enumerate(faces):
            face_crop = face.crop(frame)

            # Analysis (emotion, age, gender)
            analysis = None
            if self.analyzer and face_crop.size > 0:
                analysis = self.analyzer.analyze(face_crop, face_id=i)

            # Recognition
            recognition = None
            if self.recognizer and face_crop.size > 0:
                recognition = self.recognizer.recognize(face_crop, face_id=i)

            # Pose
            pose = None
            axis_pts = None
            landmark_result = all_landmarks[i] if i < len(all_landmarks) else None
            if self.pose_estimator and landmark_result:
                pose = self.pose_estimator.estimate(landmark_result, frame.shape)
                if self.show_pose_axes:
                    axis_pts = self.pose_estimator.get_axis_points(landmark_result, frame.shape)

            # ── Draw
            name_label = recognition.display_name if recognition else None
            self.overlay.draw_face_box(frame, face, label=name_label)

            if landmark_result and self.show_landmarks:
                self.overlay.draw_landmarks(frame, landmark_result)

            self.overlay.draw_analysis_panel(
                frame, face, analysis, recognition, pose, landmark_result
            )

            if axis_pts:
                self.overlay.draw_pose_axes(frame, axis_pts)

            if analysis and self.show_emotion_bars and analysis.emotion_scores:
                self.overlay.draw_emotion_bars(
                    frame,
                    analysis.emotion_scores,
                    x=face.x,
                    y=face.y2 + 10,
                )

        # ── Global HUD
        fps = self.fps_counter.tick()
        self.overlay.draw_hud(
            frame, fps, len(faces), self.detector.active_method, self.frame_id
        )
        self.overlay.draw_watermark(frame)

        # Recording indicator
        if self._recording:
            cv2.circle(frame, (w - 25, 25), 8, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (w - 55, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

        # Paused banner
        if self.paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "PAUSED — Press SPACE to resume", (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (230, 210, 0), 2, cv2.LINE_AA)

        return frame

    def _handle_key(self, key: int, frame: np.ndarray, cap: VideoCapture) -> bool:
        """
        Handle keyboard input. Returns False to signal quit.
        """
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            return False
        elif key == ord(" "):
            self.paused = not self.paused
            logger.info("Pipeline %s.", "paused" if self.paused else "resumed")
        elif key in (ord("l"), ord("L")):
            self.show_landmarks = not self.show_landmarks
            logger.info("Landmarks: %s", "ON" if self.show_landmarks else "OFF")
        elif key in (ord("p"), ord("P")):
            self.show_pose_axes = not self.show_pose_axes
            logger.info("Pose axes: %s", "ON" if self.show_pose_axes else "OFF")
        elif key in (ord("e"), ord("E")):
            self.show_emotion_bars = not self.show_emotion_bars
            logger.info("Emotion bars: %s", "ON" if self.show_emotion_bars else "OFF")
        elif key in (ord("s"), ord("S")):
            self._save_snapshot(frame)
        elif key in (ord("r"), ord("R")):
            self._toggle_recording(cap)
        return True

    def _save_snapshot(self, frame: np.ndarray) -> None:
        """Save current frame as a snapshot."""
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = SNAPSHOTS_DIR / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info("Snapshot saved: %s", path)

    def _toggle_recording(self, cap: VideoCapture) -> None:
        """Start/stop video recording."""
        if not self._recording:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = RECORDINGS_DIR / f"recording_{ts}.mp4"
            w, h = cap.resolution
            self._video_writer = VideoWriter(path, fps=cap.actual_fps or 30.0, width=w, height=h)
            self._video_writer.open()
            self._recording = True
            logger.info("Recording started: %s", path)
        else:
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None
            self._recording = False
            logger.info("Recording stopped.")

    def run(self, camera_index: Optional[int] = None) -> None:
        """
        Start the real-time pipeline (blocking).

        Args:
            camera_index: Override the camera index from config.
        """
        pipe_cfg = self.cfg.section("pipeline")
        disp_cfg = self.cfg.section("display")
        cam_idx = camera_index if camera_index is not None else pipe_cfg.get("camera_index", 0)
        win_title = disp_cfg.get("window_title", "FaceVision Pro")

        logger.info("Starting FaceVision Pro pipeline on camera %d ...", cam_idx)
        print(
            "\n+------------------------------------------+\n"
            "|        FaceVision Pro  v1.0.0            |\n"
            "+------------------------------------------+\n"
            "|  Q / ESC  -> Quit                        |\n"
            "|  SPACE    -> Pause / Resume              |\n"
            "|  L        -> Toggle Landmarks            |\n"
            "|  P        -> Toggle Pose Axes            |\n"
            "|  E        -> Toggle Emotion Bars         |\n"
            "|  S        -> Save Snapshot               |\n"
            "|  R        -> Start / Stop Recording      |\n"
            "+------------------------------------------+\n"
        )

        with VideoCapture(
            source=cam_idx,
            width=pipe_cfg.get("width", 1280),
            height=pipe_cfg.get("height", 720),
            fps=pipe_cfg.get("fps", 30),
        ) as cap:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            if disp_cfg.get("fullscreen", False):
                cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            last_frame = None
            for frame in cap.frames():
                if pipe_cfg.get("flip_horizontal", True):
                    frame = cv2.flip(frame, 1)

                if not self.paused:
                    last_frame = self.process_frame(frame)
                else:
                    # Re-draw pause overlay on last processed frame
                    if last_frame is not None:
                        frame = last_frame.copy()
                    frame = self.process_frame(frame)

                cv2.imshow(win_title, last_frame if last_frame is not None else frame)

                if self._recording and self._video_writer:
                    self._video_writer.write(last_frame)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, last_frame if last_frame is not None else frame, cap):
                    break

        # Cleanup
        if self._recording and self._video_writer:
            self._video_writer.release()
        if self.landmark_detector:
            self.landmark_detector.close()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")
