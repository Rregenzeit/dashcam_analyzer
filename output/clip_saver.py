# output/clip_saver.py - Ring-buffer based violation clip saver

import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import config
from analyzers.violation_checker import Violation


class ClipSaver:
    """Keeps a rolling ring-buffer of recent frames and saves clips on violation."""

    def __init__(self, fps: float, width: int, height: int, output_dir: str = config.CLIP_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self._fps = fps
        self._width = width
        self._height = height
        self._output_dir = output_dir

        buffer_frames = int(fps * config.CLIP_BUFFER_SEC)
        self._buffer: deque = deque(maxlen=buffer_frames)

        # pending clips: list of (save_until_frame, writer, path)
        self._pending: list[tuple[int, cv2.VideoWriter, str]] = []

    def push(self, frame: np.ndarray, frame_number: int):
        """Add frame to ring buffer and flush any pending clip writers."""
        self._buffer.append((frame_number, frame.copy()))
        self._flush_pending(frame, frame_number)

    def trigger(self, violation: Violation):
        """Start saving a clip for the given violation."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{violation.tracker_id}_{timestamp_str}.mp4"
        path = os.path.join(self._output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self._fps, (self._width, self._height))

        # Write buffered pre-violation frames
        for buffered_frame_no, buffered_frame in self._buffer:
            writer.write(buffered_frame)

        post_frames = int(self._fps * config.CLIP_BUFFER_SEC)
        save_until = violation.frame_number + post_frames
        self._pending.append((save_until, writer, path))
        print(f"[CLIP] Saving clip → {path}")

    def finalize(self):
        """Release all open writers."""
        for _, writer, path in self._pending:
            writer.release()
        self._pending.clear()

    # ------------------------------------------------------------------

    def _flush_pending(self, frame: np.ndarray, frame_number: int):
        still_pending = []
        for save_until, writer, path in self._pending:
            writer.write(frame)
            if frame_number >= save_until:
                writer.release()
                print(f"[CLIP] Saved → {path}")
            else:
                still_pending.append((save_until, writer, path))
        self._pending = still_pending
