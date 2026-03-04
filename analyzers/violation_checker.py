# analyzers/violation_checker.py - Decide if a lane change is a violation

from collections import deque
from dataclasses import dataclass
import config
from analyzers.lane_change import LaneChangeEvent


@dataclass
class Violation:
    tracker_id: int
    frame_number: int
    timestamp_sec: float
    violation_type: str = "NO_BLINKER_LANE_CHANGE"


class ViolationChecker:
    """Checks blinker history around lane-change events to decide violations."""

    def __init__(self, fps: float):
        self.fps = fps
        self._pre_frames = int(fps * config.BLINKER_PRE_CHANGE_SEC)
        # tracker_id -> deque of (frame_no, blinker_active)
        self._blinker_history: dict[int, deque] = {}

    def record_blinker(self, tracker_id: int, frame_number: int, blinker_active: bool):
        """Call every frame for every tracked vehicle with its blinker state."""
        if tracker_id not in self._blinker_history:
            self._blinker_history[tracker_id] = deque(maxlen=self._pre_frames + 10)
        self._blinker_history[tracker_id].append((frame_number, blinker_active))

    def check(self, event: LaneChangeEvent) -> Violation | None:
        """Return a Violation if the blinker was NOT active before this lane change.

        Looks back BLINKER_PRE_CHANGE_SEC seconds in the blinker history.
        """
        history = self._blinker_history.get(event.tracker_id)
        if history is None:
            # No history at all → assume violation
            return self._make_violation(event)

        cutoff_frame = event.frame_number - self._pre_frames
        relevant = [
            active
            for (frame_no, active) in history
            if frame_no >= cutoff_frame and frame_no <= event.frame_number
        ]

        blinker_was_on = any(relevant)
        if not blinker_was_on:
            return self._make_violation(event)
        return None

    def remove(self, tracker_id: int):
        self._blinker_history.pop(tracker_id, None)

    # ------------------------------------------------------------------

    def _make_violation(self, event: LaneChangeEvent) -> Violation:
        return Violation(
            tracker_id=event.tracker_id,
            frame_number=event.frame_number,
            timestamp_sec=event.frame_number / self.fps,
        )
