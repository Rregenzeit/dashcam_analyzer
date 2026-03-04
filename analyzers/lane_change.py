# analyzers/lane_change.py - Per-vehicle lane-change event detection

from collections import deque
from dataclasses import dataclass
import config


@dataclass
class LaneChangeEvent:
    tracker_id: int
    frame_number: int
    from_lane: int
    to_lane: int


class LaneChangeAnalyzer:
    """Tracks each vehicle's x-centroid history and detects lane crossings."""

    def __init__(self):
        self._centroid_history: dict[int, deque] = {}
        self._last_lane: dict[int, int] = {}
        self._crossing_start: dict[int, int | None] = {}
        # After crossing, count frames vehicle stays in new lane for confirmation
        self._confirm_frames: dict[int, int] = {}
        self._pending_event: dict[int, LaneChangeEvent] = {}
        self._cooldown: dict[int, int] = {}

    def update(
        self,
        tracker_id: int,
        frame_number: int,
        centroid_x: float,
        lane_xs: list[float],
    ) -> LaneChangeEvent | None:
        # Cooldown: suppress new detections for a while after a confirmed change
        if tracker_id in self._cooldown:
            self._cooldown[tracker_id] -= 1
            if self._cooldown[tracker_id] <= 0:
                del self._cooldown[tracker_id]
            else:
                self._update_history(tracker_id, frame_number, centroid_x)
                return None

        current_lane = self._assign_lane(centroid_x, lane_xs)
        self._update_history(tracker_id, frame_number, centroid_x)

        # --- No lane data: update state but never trigger events ---
        if not lane_xs or current_lane == -1:
            self._last_lane[tracker_id] = current_lane
            self._crossing_start[tracker_id] = None
            self._confirm_frames.pop(tracker_id, None)
            self._pending_event.pop(tracker_id, None)
            return None

        # --- First appearance with valid lane ---
        if tracker_id not in self._last_lane or self._last_lane[tracker_id] == -1:
            self._last_lane[tracker_id] = current_lane
            self._crossing_start[tracker_id] = None
            return None

        prev_lane = self._last_lane[tracker_id]

        # ---- Phase 1: detect crossing ----
        if current_lane != prev_lane:
            if self._crossing_start.get(tracker_id) is None:
                self._crossing_start[tracker_id] = frame_number

            frames_crossing = frame_number - self._crossing_start[tracker_id]

            if frames_crossing >= config.LANE_CHANGE_MIN_FRAMES:
                # Vehicle has been on the other side long enough → start confirmation
                if tracker_id not in self._pending_event:
                    self._pending_event[tracker_id] = LaneChangeEvent(
                        tracker_id=tracker_id,
                        frame_number=frame_number,
                        from_lane=prev_lane,
                        to_lane=current_lane,
                    )
                    self._confirm_frames[tracker_id] = 0

                self._confirm_frames[tracker_id] = \
                    self._confirm_frames.get(tracker_id, 0) + 1

                # ---- Phase 2: confirm vehicle stays in new lane ----
                if self._confirm_frames[tracker_id] >= config.LANE_CHANGE_CONFIRM_FRAMES:
                    event = self._pending_event.pop(tracker_id)
                    self._confirm_frames.pop(tracker_id, None)
                    self._last_lane[tracker_id] = current_lane
                    self._crossing_start[tracker_id] = None
                    self._cooldown[tracker_id] = config.LANE_CHANGE_COOLDOWN_FRAMES
                    return event
        else:
            # Returned to original lane before confirmation → cancel
            self._crossing_start[tracker_id] = None
            self._confirm_frames.pop(tracker_id, None)
            self._pending_event.pop(tracker_id, None)
            self._last_lane[tracker_id] = current_lane

        return None

    def remove(self, tracker_id: int):
        for d in (
            self._centroid_history,
            self._last_lane,
            self._crossing_start,
            self._confirm_frames,
            self._pending_event,
            self._cooldown,
        ):
            d.pop(tracker_id, None)

    # ------------------------------------------------------------------

    def _update_history(self, tracker_id: int, frame_number: int, centroid_x: float):
        if tracker_id not in self._centroid_history:
            self._centroid_history[tracker_id] = deque(
                maxlen=config.BLINKER_WINDOW_FRAMES * 2
            )
        self._centroid_history[tracker_id].append((frame_number, centroid_x))

    @staticmethod
    def _assign_lane(centroid_x: float, lane_xs: list[float]) -> int:
        if not lane_xs:
            return -1
        for i, x in enumerate(lane_xs):
            if centroid_x < x:
                return i
        return len(lane_xs)
