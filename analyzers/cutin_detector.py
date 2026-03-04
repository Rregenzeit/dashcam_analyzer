"""
Cut-in event detector.

A "cut-in" occurs when a tracked vehicle transitions from an adjacent lane
into the ego lane and remains there for at least CUTIN_MIN_FRAMES_IN_EGO
consecutive frames.

Ego lane is estimated as the horizontal center band of the frame
(EGO_LANE_WIDTH_RATIO * frame_width on each side of center).
If lane boundary x-coordinates are provided, the ego lane is the lane
region that contains the frame center x.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Optional

try:
    from config import (
        CUTIN_MIN_FRAMES_IN_EGO,
        CUTIN_ENTRY_BUFFER_FRAMES,
        CUTIN_COOLDOWN_FRAMES,
        EGO_LANE_WIDTH_RATIO,
    )
except ImportError:
    CUTIN_MIN_FRAMES_IN_EGO = 8
    CUTIN_ENTRY_BUFFER_FRAMES = 6
    CUTIN_COOLDOWN_FRAMES = 90
    EGO_LANE_WIDTH_RATIO = 0.35


@dataclass
class CutInEvent:
    """A confirmed cut-in event."""
    event_id: str
    track_id: int
    frame_start: int
    frame_end: int
    ego_lane_bounds: tuple[float, float]
    centroid_history: list[tuple[float, float]] = field(default_factory=list)


class _VehicleState(Enum):
    UNKNOWN = auto()
    OUTSIDE_EGO = auto()
    ENTERING = auto()
    INSIDE_EGO = auto()
    COOLDOWN = auto()


@dataclass
class _VehicleTrack:
    state: _VehicleState = _VehicleState.UNKNOWN
    enter_frame: int = -1
    frames_inside: int = 0
    cooldown_remaining: int = 0
    centroids: deque = field(default_factory=lambda: deque(maxlen=60))


def _estimate_ego_lane(frame_w: int, lane_boundaries: Optional[list[float]]) -> tuple[float, float]:
    """Return (left_x, right_x) of the ego lane region."""
    cx = frame_w / 2.0
    if lane_boundaries and len(lane_boundaries) >= 2:
        bounds = sorted(lane_boundaries)
        for i in range(len(bounds) - 1):
            if bounds[i] <= cx <= bounds[i + 1]:
                return (bounds[i], bounds[i + 1])
    half = frame_w * EGO_LANE_WIDTH_RATIO / 2.0
    return (cx - half, cx + half)


def _centroid_in_ego(cx: float, ego_left: float, ego_right: float) -> bool:
    return ego_left <= cx <= ego_right


class CutInDetector:
    """
    Stateful per-vehicle cut-in detector.

    Call update() once per frame with all tracked vehicle centroids.
    Emits CutInEvent objects when a cut-in is confirmed.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _VehicleTrack] = {}
        self._event_counter: int = 0

    def update(
        self,
        frame_idx: int,
        frame_w: int,
        vehicle_centroids: dict[int, tuple[float, float]],
        lane_boundaries: Optional[list[float]] = None,
    ) -> list[CutInEvent]:
        """Process one frame. Returns list of newly confirmed CutInEvents."""
        ego_left, ego_right = _estimate_ego_lane(frame_w, lane_boundaries)
        events: list[CutInEvent] = []

        for track_id, (cx, cy) in vehicle_centroids.items():
            track = self._tracks.setdefault(track_id, _VehicleTrack())
            track.centroids.append((cx, cy))

            if track.state == _VehicleState.COOLDOWN:
                track.cooldown_remaining -= 1
                if track.cooldown_remaining <= 0:
                    track.state = _VehicleState.OUTSIDE_EGO
                continue

            in_ego = _centroid_in_ego(cx, ego_left, ego_right)

            if track.state in (_VehicleState.UNKNOWN, _VehicleState.OUTSIDE_EGO):
                if in_ego:
                    track.state = _VehicleState.ENTERING
                    track.enter_frame = frame_idx
                    track.frames_inside = 1

            elif track.state == _VehicleState.ENTERING:
                if in_ego:
                    track.frames_inside += 1
                    if track.frames_inside >= CUTIN_ENTRY_BUFFER_FRAMES:
                        track.state = _VehicleState.INSIDE_EGO
                else:
                    track.state = _VehicleState.OUTSIDE_EGO
                    track.frames_inside = 0

            elif track.state == _VehicleState.INSIDE_EGO:
                if in_ego:
                    track.frames_inside += 1
                    if track.frames_inside >= CUTIN_MIN_FRAMES_IN_EGO:
                        self._event_counter += 1
                        evt = CutInEvent(
                            event_id=f"cutin_{self._event_counter:04d}",
                            track_id=track_id,
                            frame_start=track.enter_frame,
                            frame_end=frame_idx,
                            ego_lane_bounds=(ego_left, ego_right),
                            centroid_history=list(track.centroids),
                        )
                        events.append(evt)
                        track.state = _VehicleState.COOLDOWN
                        track.cooldown_remaining = CUTIN_COOLDOWN_FRAMES
                        track.frames_inside = 0
                else:
                    track.state = _VehicleState.OUTSIDE_EGO
                    track.frames_inside = 0

        # Remove stale tracks not seen this frame
        for tid in list(self._tracks):
            if tid not in vehicle_centroids:
                del self._tracks[tid]

        return events

    def reset(self) -> None:
        self._tracks.clear()
        self._event_counter = 0
