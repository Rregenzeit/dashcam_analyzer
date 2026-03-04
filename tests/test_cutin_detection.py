"""
Unit tests for CutInDetector using synthetic centroid data only.

State machine recap (from cutin_detector.py):
- UNKNOWN/OUTSIDE_EGO + in_ego  -> ENTERING, frames_inside=1, enter_frame=frame_idx
- ENTERING + in_ego              -> frames_inside += 1; if >= ENTRY_BUFFER -> INSIDE_EGO
  (frames_inside is NOT reset on transition)
- INSIDE_EGO + in_ego            -> frames_inside += 1; if >= MIN_IN_EGO  -> event + COOLDOWN

Default thresholds (from config.py):
    CUTIN_ENTRY_BUFFER_FRAMES = 6
    CUTIN_MIN_FRAMES_IN_EGO   = 8
    CUTIN_COOLDOWN_FRAMES     = 90
    EGO_LANE_WIDTH_RATIO      = 0.35

frame_w = 640 -> ego lane centre band: cx=320, half=640*0.35/2=112 -> [208, 432]
OUTSIDE_X = 50.0  (well outside left)
INSIDE_X  = 320.0 (dead centre)

To trigger exactly one event, a vehicle must spend at least MIN_IN_EGO=8 consecutive
frames with frames_inside >= MIN_IN_EGO while in INSIDE_EGO state.
Because frames_inside carries over from ENTERING, the minimum total consecutive
frames inside ego is MIN_IN_EGO = 8 (the check fires as soon as frames_inside >= 8,
which first happens after 8 total in-ego frames: 1 in ENTERING + 5 more -> INSIDE_EGO
at frames_inside=6, then 2 more -> frames_inside=8 -> event).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.cutin_detector import CutInDetector

FRAME_W = 640
OUTSIDE_X = 50.0
INSIDE_X = 320.0

# Total consecutive in-ego frames needed to fire one event:
# frames_inside reaches MIN_IN_EGO=8 after 8 in-ego frames (buffer carries over).
FRAMES_TO_CONFIRM = 8  # == CUTIN_MIN_FRAMES_IN_EGO


def _run_frames(detector, centroids_seq):
    """Feed a sequence of centroid dicts, return all events collected."""
    events = []
    for frame_idx, centroids in enumerate(centroids_seq):
        events.extend(detector.update(frame_idx, FRAME_W, centroids))
    return events


def _outside(track_id=1):
    return {track_id: (OUTSIDE_X, 300.0)}


def _inside(track_id=1):
    return {track_id: (INSIDE_X, 300.0)}


# ---------------------------------------------------------------------------

def test_no_cutin_vehicle_stays_outside():
    """Vehicle always outside ego lane -> no events."""
    detector = CutInDetector()
    seq = [_outside() for _ in range(30)]
    events = _run_frames(detector, seq)
    assert events == []


def test_no_cutin_brief_entry():
    """Vehicle enters ego lane for only 3 frames then leaves -> no events."""
    detector = CutInDetector()
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(3)] +
        [_outside() for _ in range(20)]
    )
    events = _run_frames(detector, seq)
    assert events == []


def test_cutin_confirmed():
    """Vehicle enters and stays for FRAMES_TO_CONFIRM frames -> exactly 1 event."""
    detector = CutInDetector()
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1


def test_cutin_event_fields():
    """Confirmed event has correct track_id and frame_start <= frame_end."""
    detector = CutInDetector()
    track_id = 7
    seq = (
        [{track_id: (OUTSIDE_X, 300.0)} for _ in range(5)] +
        [{track_id: (INSIDE_X, 300.0)} for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1
    evt = events[0]
    assert evt.track_id == track_id
    assert evt.frame_start <= evt.frame_end
    assert evt.event_id.startswith("cutin_")


def test_cooldown_prevents_duplicate():
    """After a confirmed cut-in, same vehicle re-enters immediately -> no second event."""
    detector = CutInDetector()
    # Trigger first event
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1

    # Re-enter immediately while still in cooldown (90 frames)
    more_events = []
    for frame_idx in range(len(seq), len(seq) + 50):
        more_events.extend(detector.update(frame_idx, FRAME_W, _inside()))
    assert more_events == [], "Expected no second event during cooldown"


def test_ego_lane_from_boundaries():
    """Pass lane boundaries placing ego off-centre; only in-region vehicle triggers."""
    detector = CutInDetector()
    # Boundaries at x=400 and x=600 -> ego lane covering frame centre? No.
    # Frame centre = 320, which is left of 400, so this won't straddle centre.
    # Use boundaries that DO straddle centre: 200 and 440.
    boundaries = [200.0, 440.0]
    inside_bounded_x = 320.0   # inside [200, 440]
    outside_bounded_x = 100.0  # outside [200, 440]

    # Vehicle stays outside bounded ego lane -> no events
    seq_out = [{1: (outside_bounded_x, 300.0)} for _ in range(30)]
    events = []
    for frame_idx, c in enumerate(seq_out):
        events.extend(detector.update(frame_idx, FRAME_W, c, boundaries))
    assert events == []

    # Fresh detector; vehicle inside bounded ego lane long enough -> event
    detector2 = CutInDetector()
    # Start outside, then move inside
    seq = (
        [{1: (outside_bounded_x, 300.0)} for _ in range(5)] +
        [{1: (inside_bounded_x, 300.0)} for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events2 = []
    for frame_idx, c in enumerate(seq):
        events2.extend(detector2.update(frame_idx, FRAME_W, c, boundaries))
    assert len(events2) == 1


def test_multiple_vehicles():
    """Two vehicles, only one does a cut-in -> exactly one event with correct track_id."""
    detector = CutInDetector()
    events = []
    for frame_idx in range(20):
        centroids = {
            1: (OUTSIDE_X, 300.0),   # always outside
            2: (INSIDE_X if frame_idx >= 5 else OUTSIDE_X, 300.0),  # cuts in at frame 5
        }
        events.extend(detector.update(frame_idx, FRAME_W, centroids))

    assert len(events) == 1
    assert events[0].track_id == 2


def test_reset():
    """After reset(), previously tracked vehicle triggers cut-in again from scratch."""
    detector = CutInDetector()
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events1 = _run_frames(detector, seq)
    assert len(events1) == 1

    detector.reset()

    # After reset, vehicle must go through full sequence again
    seq2 = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events2 = _run_frames(detector, seq2)
    assert len(events2) == 1
    # Event counter restarted -> first event again
    assert events2[0].event_id == "cutin_0001"
