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
        CUTIN_MIN_LATERAL_SPEED,
        CUTIN_FRONT_ZONE_RATIO,
    )
except ImportError:
    CUTIN_MIN_FRAMES_IN_EGO = 15
    CUTIN_ENTRY_BUFFER_FRAMES = 10
    CUTIN_COOLDOWN_FRAMES = 120
    EGO_LANE_WIDTH_RATIO = 0.45
    CUTIN_MIN_LATERAL_SPEED = 0.015
    CUTIN_FRONT_ZONE_RATIO = 0.70


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
    x_history: deque = field(default_factory=lambda: deque(maxlen=5))
    # 인접 차선(ego lane 바깥)에서 진입한 경우에만 True → 이 플래그가 True여야 이벤트 발생
    entered_from_adjacent: bool = False


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
        frame_h: Optional[int] = None,
    ) -> list[CutInEvent]:
        """Process one frame. Returns list of newly confirmed CutInEvents."""
        ego_left, ego_right = _estimate_ego_lane(frame_w, lane_boundaries)
        events: list[CutInEvent] = []

        # Front-zone threshold: ignore vehicles in the upper part of the frame
        front_zone_threshold = (frame_h * (1.0 - CUTIN_FRONT_ZONE_RATIO)) if frame_h else None

        for track_id, (cx, cy) in vehicle_centroids.items():
            # Filter A: front zone — skip vehicles too far away (high in the frame)
            if front_zone_threshold is not None and cy < front_zone_threshold:
                continue

            track = self._tracks.setdefault(track_id, _VehicleTrack())
            track.centroids.append((cx, cy))
            track.x_history.append(cx)

            if track.state == _VehicleState.COOLDOWN:
                track.cooldown_remaining -= 1
                if track.cooldown_remaining <= 0:
                    track.state = _VehicleState.OUTSIDE_EGO
                    track.entered_from_adjacent = False
                continue

            in_ego = _centroid_in_ego(cx, ego_left, ego_right)

            if track.state == _VehicleState.UNKNOWN:
                # 첫 등장 프레임: ego lane 위치만 기록, 이벤트 발생 경로 진입 없음
                track.state = _VehicleState.OUTSIDE_EGO if not in_ego else _VehicleState.INSIDE_EGO

            elif track.state == _VehicleState.OUTSIDE_EGO:
                if in_ego:
                    # Filter B: 횡방향 이동 속도 — 충분한 횡방향 이동이 있어야 ENTERING 전환
                    if len(track.x_history) >= 2:
                        lateral_displacement = abs(track.x_history[-1] - track.x_history[0])
                        min_displacement = CUTIN_MIN_LATERAL_SPEED * frame_w
                        if lateral_displacement < min_displacement:
                            continue
                    track.state = _VehicleState.ENTERING
                    track.enter_frame = frame_idx
                    track.frames_inside = 1
                    track.entered_from_adjacent = True  # 인접 차선에서 진입 확인

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
                    # entered_from_adjacent가 False인 경우 = 처음부터 앞에 있던 차량 → 이벤트 없음
                    if not track.entered_from_adjacent:
                        continue
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

    def candidate_track_ids(self) -> set[int]:
        """현재 끼어들기 후보(ENTERING / INSIDE_EGO) 차량의 track_id 집합 반환."""
        return {
            tid for tid, t in self._tracks.items()
            if t.state in (_VehicleState.ENTERING, _VehicleState.INSIDE_EGO)
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._event_counter = 0
