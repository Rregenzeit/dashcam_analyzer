"""
Best-effort license plate recognizer using EasyOCR.

Designed to be modular and optional: if easyocr is not installed,
PlateDetector degrades gracefully (returns None for all crops).

Multi-frame voting: accumulate OCR results over PLATE_VOTE_WINDOW frames
per track_id and return the most-common non-empty string.
"""
from __future__ import annotations
import re
from collections import Counter, deque
from typing import Optional

import numpy as np

try:
    from config import PLATE_CROP_EXPAND, PLATE_VOTE_WINDOW, PLATE_MIN_CONFIDENCE
except ImportError:
    PLATE_CROP_EXPAND = 0.15
    PLATE_VOTE_WINDOW = 10
    PLATE_MIN_CONFIDENCE = 0.3

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False


def _clean_plate(text: str) -> str:
    """Strip whitespace and keep only alphanumeric + common plate chars."""
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text.upper().strip())
    return cleaned if len(cleaned) >= 3 else ""


class PlateDetector:
    """
    Per-vehicle plate text estimator with multi-frame voting.

    Usage:
        detector = PlateDetector()
        # each frame, for vehicles of interest:
        detector.update(track_id, frame, bbox_xyxy)
        # after enough frames:
        plate = detector.get_best(track_id)
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        self._reader = None
        self._languages = languages or ["en"]
        self._history: dict[int, deque] = {}

    def _ensure_reader(self) -> bool:
        if not _EASYOCR_AVAILABLE:
            return False
        if self._reader is None:
            try:
                self._reader = easyocr.Reader(self._languages, gpu=self._gpu_available())
            except Exception:
                return False
        return True

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _crop_bbox(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        expand_x = int(bw * PLATE_CROP_EXPAND)
        expand_y = int(bh * PLATE_CROP_EXPAND)
        nx1 = max(0, x1 - expand_x)
        ny1 = max(0, y1 - expand_y)
        nx2 = min(w, x2 + expand_x)
        ny2 = min(h, y2 + expand_y)
        crop = frame[ny1:ny2, nx1:nx2]
        return crop if crop.size > 0 else None

    def update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> None:
        """Add one frame's OCR candidate for a vehicle."""
        if not self._ensure_reader():
            return
        crop = self._crop_bbox(frame, bbox_xyxy)
        if crop is None:
            return
        try:
            results = self._reader.readtext(crop, detail=1)
        except Exception:
            return
        for (_bbox, text, conf) in results:
            if conf >= PLATE_MIN_CONFIDENCE:
                cleaned = _clean_plate(text)
                if cleaned:
                    hist = self._history.setdefault(
                        track_id, deque(maxlen=PLATE_VOTE_WINDOW)
                    )
                    hist.append(cleaned)

    def get_best(self, track_id: int) -> Optional[str]:
        """Return the most-common plate string seen so far, or None."""
        hist = self._history.get(track_id)
        if not hist:
            return None
        counter = Counter(hist)
        most_common, count = counter.most_common(1)[0]
        return most_common if count >= 2 else None

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def reset(self) -> None:
        self._history.clear()
        self._reader = None
