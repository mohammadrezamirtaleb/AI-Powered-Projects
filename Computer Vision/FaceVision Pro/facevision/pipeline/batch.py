"""
Batch Image/Video Processing Pipeline
=======================================
Processes images or video files through the FaceVision Pro analysis pipeline
and saves annotated results to disk.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from facevision.core.analyzer import FaceAnalyzer
from facevision.core.detector import FaceDetector
from facevision.core.landmarks import LandmarkDetector
from facevision.core.pose import HeadPoseEstimator
from facevision.core.recognizer import FaceRecognizer
from facevision.utils.config import Config
from facevision.utils.drawing import FaceOverlay
from facevision.utils.io import FPSCounter, ImageIO, VideoCapture, VideoWriter

logger = logging.getLogger(__name__)


class BatchPipeline:
    """
    Processes images or video files through the FaceVision Pro pipeline.

    Args:
        config: Config instance (loads default.yaml if not provided).
        output_dir: Directory to save processed outputs.

    Example:
        >>> pipeline = BatchPipeline(output_dir=Path("outputs/batch"))
        >>> pipeline.process_image("photo.jpg")
        >>> pipeline.process_video("video.mp4")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = config or Config()
        self.output_dir = output_dir or Path("outputs/batch")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        det_cfg = self.cfg.section("detector")
        self.detector = FaceDetector(
            method=det_cfg.get("method", "dnn"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.6),
        )
        lm_cfg = self.cfg.section("landmarks")
        self.landmark_detector = LandmarkDetector(
            max_faces=lm_cfg.get("max_faces", 5),
        ) if lm_cfg.get("enabled", True) else None

        an_cfg = self.cfg.section("analyzer")
        self.analyzer = FaceAnalyzer(
            actions=an_cfg.get("actions", ["emotion", "age", "gender"]),
            cooldown_seconds=0,  # No cooldown in batch mode
        ) if an_cfg.get("enabled", True) else None

        rec_cfg = self.cfg.section("recognizer")
        self.recognizer = FaceRecognizer(
            tolerance=rec_cfg.get("tolerance", 0.50),
        ) if rec_cfg.get("enabled", True) else None
        if self.recognizer and rec_cfg.get("auto_load_database", True):
            self.recognizer.load_database()

        pose_cfg = self.cfg.section("pose")
        self.pose_estimator = HeadPoseEstimator(
            smoothing=1,  # No smoothing needed for stills
        ) if pose_cfg.get("enabled", True) else None

        self.overlay = FaceOverlay()

    def process_image(
        self, image_path: Union[str, Path], save: bool = True
    ) -> Optional[np.ndarray]:
        """
        Analyze a single image file.

        Args:
            image_path: Path to the input image.
            save: Whether to save the annotated image to output_dir.

        Returns:
            Annotated image as NumPy array, or None on failure.
        """
        image_path = Path(image_path)
        frame = ImageIO.read_image(image_path)
        if frame is None:
            return None

        annotated = self._annotate_frame(frame)

        if save:
            out_path = self.output_dir / f"annotated_{image_path.stem}.jpg"
            ImageIO.write_image(annotated, out_path)
            logger.info("Saved annotated image: %s", out_path)

        return annotated

    def process_directory(
        self, directory: Union[str, Path], save: bool = True
    ) -> List[np.ndarray]:
        """
        Process all images in a directory.

        Args:
            directory: Directory containing images.
            save: Whether to save annotated images.

        Returns:
            List of annotated images.
        """
        images = ImageIO.list_images(directory)
        logger.info("Found %d images in %s.", len(images), directory)
        results = []
        for img_path in images:
            result = self.process_image(img_path, save=save)
            if result is not None:
                results.append(result)
        return results

    def process_video(
        self,
        video_path: Union[str, Path],
        save: bool = True,
        max_frames: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Process a video file frame-by-frame.

        Args:
            video_path: Path to input video.
            save: Whether to save annotated video.
            max_frames: Limit processing to N frames (None = all frames).

        Returns:
            Path to output video, or None on failure.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error("Video not found: %s", video_path)
            return None

        fps_counter = FPSCounter()
        out_path: Optional[Path] = None
        writer: Optional[VideoWriter] = None

        with VideoCapture(str(video_path)) as cap:
            w, h = cap.resolution
            out_path = self.output_dir / f"annotated_{video_path.stem}.mp4"

            if save:
                writer = VideoWriter(out_path, fps=cap.actual_fps or 30.0, width=w, height=h)
                writer.open()

            frame_count = 0
            for frame in cap.frames():
                if max_frames and frame_count >= max_frames:
                    break
                annotated = self._annotate_frame(frame)
                fps = fps_counter.tick()
                logger.debug("Frame %d — FPS: %.1f", frame_count, fps)
                if writer:
                    writer.write(annotated)
                frame_count += 1

            if writer:
                writer.release()

        logger.info("Processed %d frames from '%s'.", frame_count, video_path.name)
        return out_path

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the full analysis pipeline on a single frame."""
        faces = self.detector.detect(frame)
        all_landmarks = []
        if self.landmark_detector and faces:
            all_landmarks = self.landmark_detector.detect(frame)

        for i, face in enumerate(faces):
            face_crop = face.crop(frame)
            analysis = self.analyzer.analyze(face_crop, i) if self.analyzer and face_crop.size > 0 else None
            recognition = self.recognizer.recognize(face_crop, i) if self.recognizer and face_crop.size > 0 else None
            landmark_result = all_landmarks[i] if i < len(all_landmarks) else None
            pose = None
            if self.pose_estimator and landmark_result:
                pose = self.pose_estimator.estimate(landmark_result, frame.shape)

            self.overlay.draw_face_box(frame, face, label=recognition.display_name if recognition else None)
            if landmark_result:
                self.overlay.draw_landmarks(frame, landmark_result)
            self.overlay.draw_analysis_panel(frame, face, analysis, recognition, pose, landmark_result)

        self.overlay.draw_watermark(frame)
        return frame
