# detectors/lane_detector.py - Hough-transform lane detection with temporal smoothing

import cv2
import numpy as np
import math
from collections import deque
import config


class LaneDetector:
    def __init__(self):
        # Separate history for left/right lane boundary x positions
        self._left_history: deque = deque(maxlen=config.LANE_SMOOTHING_FRAMES)
        self._right_history: deque = deque(maxlen=config.LANE_SMOOTHING_FRAMES)

    def detect(self, frame: np.ndarray) -> list[float]:
        """Return smoothed, sorted list of lane boundary x-coordinates."""
        h, w = frame.shape[:2]
        roi = self._extract_roi(frame, h, w)
        edges = self._preprocess(roi)
        lines = self._hough(edges)
        raw_left, raw_right = self._split_lines(lines, w)

        # Update history only when new detections are available
        if raw_left:
            self._left_history.append(float(np.mean(raw_left)))
        if raw_right:
            self._right_history.append(float(np.mean(raw_right)))

        # Build smoothed lane list from history averages
        lane_xs = []
        if self._left_history:
            lane_xs.append(float(np.mean(self._left_history)))
        if self._right_history:
            lane_xs.append(float(np.mean(self._right_history)))

        return sorted(lane_xs)

    # ------------------------------------------------------------------

    def _extract_roi(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        top = int(h * config.ROI_TOP_RATIO)
        bottom = int(h * config.ROI_BOTTOM_RATIO)
        return frame[top:bottom, :]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        return cv2.Canny(blur, config.CANNY_LOW, config.CANNY_HIGH)

    def _hough(self, edges: np.ndarray):
        theta = config.HOUGH_THETA_DEG * math.pi / 180
        return cv2.HoughLinesP(
            edges,
            rho=config.HOUGH_RHO,
            theta=theta,
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LINE_LEN,
            maxLineGap=config.HOUGH_MAX_LINE_GAP,
        )

    def _split_lines(self, lines, w: int) -> tuple[list[float], list[float]]:
        """Separate detected lines into left / right boundary candidates."""
        if lines is None:
            return [], []

        mid = w / 2
        left_xs, right_xs = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if not (config.LANE_SLOPE_MIN <= abs(slope) <= config.LANE_SLOPE_MAX):
                continue
            x_avg = (x1 + x2) / 2.0
            if x_avg < mid:
                left_xs.append(x_avg)
            else:
                right_xs.append(x_avg)

        return left_xs, right_xs

    def get_debug_edges(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        roi = self._extract_roi(frame, h, w)
        return self._preprocess(roi)
