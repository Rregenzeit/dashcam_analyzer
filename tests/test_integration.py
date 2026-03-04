import numpy as np
import cv2
import json
from pathlib import Path
import pytest


def _make_test_video(path: str, n_frames: int = 60, w: int = 640, h: int = 480) -> None:
    """Write a tiny synthetic MP4 with a moving rectangle simulating a vehicle."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw lane lines
        cv2.line(frame, (200, h), (200, h // 2), (255, 255, 255), 2)
        cv2.line(frame, (440, h), (440, h // 2), (255, 255, 255), 2)
        # Draw a moving vehicle: starts at x=80, moves to x=300 (into ego lane)
        vx = min(80 + i * 4, 300)
        cv2.rectangle(frame, (vx, 300), (vx + 60, 360), (0, 200, 0), -1)
        writer.write(frame)
    writer.release()


@pytest.mark.integration
def test_cutin_detector_on_synthetic_video(tmp_path):
    """
    Run CutInDetector directly on synthetic centroid data derived from a
    moving vehicle that enters the ego lane mid-video.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from analyzers.cutin_detector import CutInDetector

    detector = CutInDetector()
    frame_w = 640
    events = []

    for frame_idx in range(80):
        # Vehicle starts at x=80 (outside ego), crosses into ego (x=320) around frame 30
        vx = min(80 + frame_idx * 3, 320)
        centroids = {1: (float(vx + 30), 330.0)}
        evts = detector.update(frame_idx, frame_w, centroids)
        events.extend(evts)

    assert len(events) == 1, f"Expected 1 cut-in event, got {len(events)}"
    evt = events[0]
    assert evt.track_id == 1
    assert evt.frame_start < evt.frame_end


@pytest.mark.integration
def test_pipeline_web_on_synthetic_video(tmp_path, monkeypatch):
    """
    Run WebPipeline on a tiny synthetic video.
    Verifies result.json is written with correct structure.
    Mocks VehicleDetector to avoid needing model weights in CI.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    class _FakeDetections:
        tracker_id = None
        xyxy = np.zeros((0, 4))

    class _FakeVehicleDetector:
        def __init__(self, **kwargs):
            pass

        def detect_and_track(self, frame):
            return _FakeDetections()

    import detectors.vehicle_detector as vd_module
    monkeypatch.setattr(vd_module, "VehicleDetector", _FakeVehicleDetector)

    # Write synthetic video
    vid_path = str(tmp_path / "test.mp4")
    _make_test_video(vid_path, n_frames=30)

    import importlib
    import pipeline_web
    importlib.reload(pipeline_web)

    pipeline = pipeline_web.WebPipeline(
        job_id="test-job-001",
        device="cpu",
        progress_callback=lambda f, m: None,
    )
    # Override output dir to tmp_path
    pipeline.output_dir = tmp_path / "test-job-001"
    pipeline.clips_dir = pipeline.output_dir / "clips"
    pipeline.clips_dir.mkdir(parents=True, exist_ok=True)

    pipeline.run(vid_path)

    result_path = pipeline.output_dir / "result.json"
    assert result_path.exists(), "result.json was not created"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["job_id"] == "test-job-001"
    assert result["status"] == "done"
    assert "events" in result
