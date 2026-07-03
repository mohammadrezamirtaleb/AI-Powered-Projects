"""
Image and Video I/O Utilities
==============================
Provides helpers for reading/writing images, video files, and webcam streams.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageIO:
    """Static helpers for image and video I/O."""

    @staticmethod
    def read_image(path: str | Path) -> Optional[np.ndarray]:
        """Read a BGR image from disk. Returns None if reading fails."""
        img = cv2.imread(str(path))
        if img is None:
            logger.error("Could not read image: %s", path)
        return img

    @staticmethod
    def write_image(image: np.ndarray, path: str | Path) -> bool:
        """Write a BGR image to disk. Returns True on success."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), image)
        if success:
            logger.info("Saved image to %s", path)
        else:
            logger.error("Failed to save image to %s", path)
        return success

    @staticmethod
    def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
        """Resize image to target width preserving aspect ratio."""
        h, w = image.shape[:2]
        scale = width / w
        new_h = int(h * scale)
        return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def resize_to_fit(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        """Resize image to fit within a bounding box, preserving aspect ratio."""
        h, w = image.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def add_padding(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        """Add black padding around an image."""
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    @staticmethod
    def list_images(directory: str | Path, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
        """List all image files in a directory recursively."""
        directory = Path(directory)
        images = []
        for ext in extensions:
            images.extend(directory.rglob(f"*{ext}"))
        return sorted(images)


class VideoCapture:
    """
    Context-manager wrapper around cv2.VideoCapture.

    Supports both webcam (int index) and video file (path string) inputs.

    Example:
        >>> with VideoCapture(0) as cap:
        ...     for frame in cap.frames():
        ...         process(frame)
    """

    def __init__(
        self,
        source: int | str = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        fps: Optional[int] = 30,
    ) -> None:
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "VideoCapture":
        """Open the video capture source."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        if self.target_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        if self.target_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        logger.info(
            "Opened video source %s: %dx%d @ %d fps",
            self.source,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._cap.get(cv2.CAP_PROP_FPS)),
        )
        return self

    def read(self) -> Optional[np.ndarray]:
        """Read a single frame. Returns None if source is exhausted."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously until source is exhausted."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    @property
    def frame_count(self) -> int:
        """Total number of frames (for video files)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def actual_fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoCapture":
        return self.open()

    def __exit__(self, *args) -> None:
        self.release()


class VideoWriter:
    """
    Context-manager wrapper around cv2.VideoWriter for saving output video.

    Example:
        >>> with VideoWriter("output.mp4", fps=30, width=1280, height=720) as writer:
        ...     writer.write(frame)
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        codec: str = "mp4v",
    ) -> None:
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> "VideoWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        logger.info("VideoWriter opened: %s (%dx%d @ %.1f fps)", self.output_path, self.width, self.height, self.fps)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
            logger.info("VideoWriter closed: %s", self.output_path)

    def __enter__(self) -> "VideoWriter":
        return self.open()

    def __exit__(self, *args) -> None:
        self.release()


class FPSCounter:
    """Smooth rolling FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list = []
        self.window = window
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame. Returns current smoothed FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) > self.window:
            self._times.pop(0)
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0
