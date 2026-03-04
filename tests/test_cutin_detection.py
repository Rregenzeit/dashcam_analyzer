"""
CutInDetector 유닛 테스트 (합성 중심 좌표 사용).

상태 머신 요약 (cutin_detector.py):
- UNKNOWN/OUTSIDE_EGO + in_ego + 충분한 횡방향 이동 -> ENTERING, frames_inside=1
- ENTERING + in_ego  -> frames_inside += 1; if >= ENTRY_BUFFER -> INSIDE_EGO
  (frames_inside는 전환 시 초기화되지 않음)
- INSIDE_EGO + in_ego -> frames_inside += 1; if >= MIN_IN_EGO  -> 이벤트 + COOLDOWN

현재 임계값 (config.py):
    CUTIN_ENTRY_BUFFER_FRAMES  = 10
    CUTIN_MIN_FRAMES_IN_EGO    = 15
    CUTIN_COOLDOWN_FRAMES      = 120
    EGO_LANE_WIDTH_RATIO       = 0.45
    CUTIN_MIN_LATERAL_SPEED    = 0.015
    CUTIN_FRONT_ZONE_RATIO     = 0.70

frame_w=640 -> ego lane: cx=320, half=640*0.45/2=144 -> [176, 464]

이벤트 발생 최소 연속 프레임:
  frames_inside가 MIN_IN_EGO(15)에 도달해야 함.
  최초 진입 프레임에 frames_inside=1로 시작하므로 최소 15 연속 프레임 필요.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.cutin_detector import CutInDetector

FRAME_W = 640
FRAME_H = 480
OUTSIDE_X = 50.0    # ego lane 왼쪽 바깥
INSIDE_X = 320.0    # 정중앙 (ego lane 내부)
BOTTOM_Y = 400.0    # 하단 (전방 영역 필터 통과)
TOP_Y = 50.0        # 상단 (전방 영역 필터 차단)

# 이벤트 발생에 필요한 최소 연속 프레임 수 = CUTIN_MIN_FRAMES_IN_EGO
FRAMES_TO_CONFIRM = 15


def _run_frames(detector, centroids_seq, frame_h=None):
    """중심 좌표 시퀀스를 입력하고 수집된 모든 이벤트를 반환."""
    events = []
    for frame_idx, centroids in enumerate(centroids_seq):
        events.extend(detector.update(frame_idx, FRAME_W, centroids, frame_h=frame_h))
    return events


def _outside(track_id=1, y=BOTTOM_Y):
    return {track_id: (OUTSIDE_X, y)}


def _inside(track_id=1, y=BOTTOM_Y):
    return {track_id: (INSIDE_X, y)}


# ---------------------------------------------------------------------------
# 기본 동작 테스트
# ---------------------------------------------------------------------------

def test_no_cutin_vehicle_stays_outside():
    """항상 ego lane 바깥에 있는 차량 -> 이벤트 없음."""
    detector = CutInDetector()
    seq = [_outside() for _ in range(30)]
    assert _run_frames(detector, seq) == []


def test_no_cutin_brief_entry():
    """3프레임만 진입 후 이탈 -> 이벤트 없음."""
    detector = CutInDetector()
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(3)] +
        [_outside() for _ in range(20)]
    )
    assert _run_frames(detector, seq) == []


def test_cutin_confirmed():
    """FRAMES_TO_CONFIRM 프레임 이상 진입 유지 -> 정확히 1개 이벤트."""
    detector = CutInDetector()
    # 충분한 횡방향 이동 보장: OUTSIDE에서 INSIDE로 이동
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1


def test_cutin_event_fields():
    """확인된 이벤트의 필드 검증."""
    detector = CutInDetector()
    track_id = 7
    seq = (
        [{track_id: (OUTSIDE_X, BOTTOM_Y)} for _ in range(5)] +
        [{track_id: (INSIDE_X, BOTTOM_Y)} for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1
    evt = events[0]
    assert evt.track_id == track_id
    assert evt.frame_start <= evt.frame_end
    assert evt.event_id.startswith("cutin_")


def test_cooldown_prevents_duplicate():
    """끼어들기 확인 후 동일 차량 즉시 재진입 -> 두 번째 이벤트 없음."""
    detector = CutInDetector()
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1

    # 쿨다운(120 프레임) 중 재진입 -> 이벤트 없어야 함
    more_events = []
    for frame_idx in range(len(seq), len(seq) + 50):
        more_events.extend(detector.update(frame_idx, FRAME_W, _inside()))
    assert more_events == [], "쿨다운 중 두 번째 이벤트가 발생하면 안 됨"


def test_ego_lane_from_boundaries():
    """차선 경계 좌표 사용 시 ego lane 영역 판별 정확성 검증."""
    detector = CutInDetector()
    boundaries = [200.0, 440.0]
    inside_x = 320.0    # [200, 440] 내부
    outside_x = 100.0   # [200, 440] 외부

    seq_out = [{1: (outside_x, BOTTOM_Y)} for _ in range(30)]
    events = []
    for frame_idx, c in enumerate(seq_out):
        events.extend(detector.update(frame_idx, FRAME_W, c, boundaries))
    assert events == []

    detector2 = CutInDetector()
    seq = (
        [{1: (outside_x, BOTTOM_Y)} for _ in range(5)] +
        [{1: (inside_x, BOTTOM_Y)} for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events2 = []
    for frame_idx, c in enumerate(seq):
        events2.extend(detector2.update(frame_idx, FRAME_W, c, boundaries))
    assert len(events2) == 1


def test_multiple_vehicles():
    """두 차량 중 한 대만 끼어들기 -> track_id 일치하는 이벤트 1개."""
    detector = CutInDetector()
    events = []
    for frame_idx in range(30):
        centroids = {
            1: (OUTSIDE_X, BOTTOM_Y),
            2: (INSIDE_X if frame_idx >= 5 else OUTSIDE_X, BOTTOM_Y),
        }
        events.extend(detector.update(frame_idx, FRAME_W, centroids))
    assert len(events) == 1
    assert events[0].track_id == 2


def test_reset():
    """reset() 후 동일 차량이 처음부터 다시 이벤트 트리거."""
    detector = CutInDetector()
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events1 = _run_frames(detector, seq)
    assert len(events1) == 1

    detector.reset()

    events2 = _run_frames(detector, seq)
    assert len(events2) == 1
    assert events2[0].event_id == "cutin_0001"


# ---------------------------------------------------------------------------
# 오탐지 방지 필터 테스트
# ---------------------------------------------------------------------------

def test_front_zone_filter_blocks_far_vehicles():
    """화면 상단 차량(먼 거리) -> 전방 영역 필터로 차단, 이벤트 없음."""
    detector = CutInDetector()
    # CUTIN_FRONT_ZONE_RATIO=0.70 -> threshold = 480 * 0.30 = 144
    # TOP_Y=50 < 144 이므로 필터 차단
    seq = (
        [{1: (OUTSIDE_X, TOP_Y)} for _ in range(5)] +
        [{1: (INSIDE_X, TOP_Y)} for _ in range(FRAMES_TO_CONFIRM + 5)]
    )
    events = _run_frames(detector, seq, frame_h=FRAME_H)
    assert events == [], "먼 거리(상단) 차량은 끼어들기로 판정하면 안 됨"


def test_front_zone_filter_allows_close_vehicles():
    """화면 하단 차량(가까운 거리) -> 전방 영역 필터 통과, 이벤트 발생."""
    detector = CutInDetector()
    # BOTTOM_Y=400 > 144 이므로 필터 통과
    seq = (
        [{1: (OUTSIDE_X, BOTTOM_Y)} for _ in range(5)] +
        [{1: (INSIDE_X, BOTTOM_Y)} for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq, frame_h=FRAME_H)
    assert len(events) == 1, "가까운 거리 차량은 끼어들기 이벤트를 발생시켜야 함"


def test_lateral_speed_filter_blocks_small_lateral_move():
    """ego lane 경계 근처에서 미세 횡방향 이동으로 진입 -> 필터 차단, 이벤트 없음.

    ego left boundary = 640*0.45/2 = 144 → 320-144 = 176.
    NEAR_OUTSIDE_X=170, NEAR_INSIDE_X=178: displacement=8px < min(9.6px) → 차단.
    """
    detector = CutInDetector()
    NEAR_OUTSIDE_X = 170.0  # 경계(176) 바로 바깥
    NEAR_INSIDE_X = 178.0   # 경계(176) 바로 안쪽, 이동 8px < 9.6px
    seq = (
        [{1: (NEAR_OUTSIDE_X, BOTTOM_Y)} for _ in range(5)] +
        [{1: (NEAR_INSIDE_X, BOTTOM_Y)} for _ in range(FRAMES_TO_CONFIRM + 5)]
    )
    events = _run_frames(detector, seq)
    assert events == [], "횡방향 이동이 최소값 미만이면 끼어들기로 판정하면 안 됨"


def test_lateral_speed_filter_allows_fast_lateral_move():
    """큰 횡방향 이동 후 진입 -> 필터 통과, 이벤트 발생."""
    detector = CutInDetector()
    # OUTSIDE_X=50 → INSIDE_X=320: 횡방향 이동 270px >> 0.015*640=9.6px
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(FRAMES_TO_CONFIRM + 2)]
    )
    events = _run_frames(detector, seq)
    assert len(events) == 1


def test_no_event_if_exits_before_confirmation():
    """ENTRY_BUFFER 이상 진입했지만 MIN_IN_EGO 도달 전 이탈 -> 이벤트 없음."""
    detector = CutInDetector()
    # 12프레임 진입(ENTRY_BUFFER=10 넘음) 후 이탈 -> MIN_IN_EGO=15 미달
    seq = (
        [_outside() for _ in range(5)] +
        [_inside() for _ in range(12)] +
        [_outside() for _ in range(10)]
    )
    events = _run_frames(detector, seq)
    assert events == [], "MIN_IN_EGO 도달 전 이탈 시 이벤트 없어야 함"
