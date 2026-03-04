# detectors/blinker_detector.py - Amber blinker detection via HSV + flicker analysis

import cv2
import numpy as np
from collections import deque
import config


class BlinkerDetector:
    """Per-vehicle blinker state tracker using a sliding window of amber ratios."""

    def __init__(self):
        # tracker_id -> deque of (is_on: bool)
        self._history: dict[int, deque] = {}

    def update(self, frame: np.ndarray, tracker_id: int, bbox: np.ndarray) -> bool:
        """Update blinker history for one vehicle and return whether blinker is active.

        Args:
            frame: Full BGR frame.
            tracker_id: Unique vehicle track ID.
            bbox: [x1, y1, x2, y2] bounding box.

        Returns:
            True if blinker is currently active (flashing detected).
        """
        if tracker_id not in self._history:
            self._history[tracker_id] = deque(maxlen=config.BLINKER_WINDOW_FRAMES)

        amber_ratio = self._amber_ratio(frame, bbox)
        is_on = amber_ratio >= config.BLINKER_PIXEL_RATIO_THRESHOLD
        self._history[tracker_id].append(is_on)

        return self._is_blinking(self._history[tracker_id])

    def remove(self, tracker_id: int):
        self._history.pop(tracker_id, None)

    # ------------------------------------------------------------------

    def _amber_ratio(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """Return fraction of amber pixels in the side strips of the bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1] - 1)
        y2 = min(y2, frame.shape[0] - 1)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        bw = x2 - x1
        side_w = max(1, int(bw * config.BLINKER_SIDE_RATIO))

        # Left strip + right strip
        left_crop = frame[y1:y2, x1:x1 + side_w]
        right_crop = frame[y1:y2, x2 - side_w:x2]
        combined = np.concatenate([left_crop, right_crop], axis=1)

        if combined.size == 0:
            return 0.0

        hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        lower = np.array([config.BLINKER_AMBER_HUE_LOW, config.BLINKER_SAT_LOW, config.BLINKER_VAL_LOW])
        upper = np.array([config.BLINKER_AMBER_HUE_HIGH, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        total_pixels = mask.shape[0] * mask.shape[1]
        amber_pixels = int(np.sum(mask > 0))
        return amber_pixels / total_pixels if total_pixels > 0 else 0.0

    def _is_blinking(self, history: deque) -> bool:
        """Count ON→OFF or OFF→ON transitions; blink if enough transitions."""
        if len(history) < 4:
            return False
        transitions = sum(
            1 for i in range(1, len(history))
            if history[i] != history[i - 1]
        )
        return transitions >= config.BLINKER_BLINK_MIN_TRANSITIONS
