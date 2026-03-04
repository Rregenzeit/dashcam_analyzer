"""
WebPipeline: Asynchronous video processing pipeline for the web API.

Wraps existing detectors/analyzers with:
 - Cut-in event detection (CutInDetector)
 - Best-effort plate recognition (PlateDetector)
 - Turn signal detection (BlinkerDetector, existing)
 - Clip extraction per cut-in event
 - result.json output
 - Progress callback for live status updates

Does NOT modify or call the existing Pipeline class — runs independently
so the CLI workflow is untouched.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from detectors.vehicle_detector import VehicleDetector
from detectors.lane_detector import LaneDetector
from detectors.blinker_detector import BlinkerDetector
from detectors.plate_detector import PlateDetector
from analyzers.cutin_detector import CutInDetector, CutInEvent as _CutInEvent

try:
    from config import (
        OUTPUTS_DIR, CLIP_BUFFER_SECS_WEB,
        YOLO_MODEL, CONF_THRESHOLD as YOLO_CONF,
    )
except ImportError:
    OUTPUTS_DIR = "outputs"
    CLIP_BUFFER_SECS_WEB = 3.0
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONF = 0.4


ProgressCallback = Callable[[float, str], None]

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


class WebPipeline:
    """
    End-to-end pipeline for a single job.

    Usage (called by api/worker.py):
        pipeline = WebPipeline(job_id=..., device=..., progress_callback=...)
        pipeline.run("/path/to/video.mp4")
    """

    def __init__(
        self,
        job_id: str,
        device: str = "cpu",
        progress_callback: Optional[ProgressCallback] = None,
        model_path: str = YOLO_MODEL,
    ) -> None:
        self.job_id = job_id
        self.device = device
        self._progress_cb = progress_callback or (lambda f, m: None)

        self.output_dir = Path(OUTPUTS_DIR) / job_id
        self.clips_dir = self.output_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Detectors
        self.vehicle_detector = VehicleDetector(model_path=model_path)
        self.lane_detector = LaneDetector()
        self.blinker_detector = BlinkerDetector()
        self.plate_detector = PlateDetector()

        # Analyzers
        self.cutin_detector = CutInDetector()

        self._events: list[dict] = []

        # Cache of most-recent blinker state per track (updated each frame)
        self._blinker_states: dict[int, bool] = {}

    # ------------------------------------------------------------------ #

    def _progress(self, frac: float, msg: str = "") -> None:
        self._progress_cb(frac, msg)

    def run(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buffer_frames = int(CLIP_BUFFER_SECS_WEB * fps)

        ring: list[np.ndarray] = []
        frame_idx = 0

        # Active clip writers: event_id -> {writer, frames_remaining}
        active_writers: dict[str, dict] = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # -- Detection ------------------------------------------
                detections = self.vehicle_detector.detect_and_track(frame)
                lane_bounds = self.lane_detector.detect(frame)
                centroids: dict[int, tuple[float, float]] = {}

                if detections.tracker_id is not None:
                    for i, tid in enumerate(detections.tracker_id):
                        tid = int(tid)
                        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        centroids[tid] = (cx, cy)

                        bbox = detections.xyxy[i]
                        # BlinkerDetector.update returns bool directly
                        self._blinker_states[tid] = self.blinker_detector.update(
                            frame, tid, bbox
                        )
                        self.plate_detector.update(tid, frame, (x1, y1, x2, y2))

                # -- Cut-in detection -----------------------------------
                new_events: list[_CutInEvent] = self.cutin_detector.update(
                    frame_idx, frame_w, centroids, lane_bounds, frame_h
                )

                for evt in new_events:
                    blinker_on = self._blinker_states.get(evt.track_id)
                    if blinker_on is True:
                        turn_signal = "on"
                    elif blinker_on is False:
                        turn_signal = "off"
                    else:
                        turn_signal = "unknown"

                    plate = self.plate_detector.get_best(evt.track_id)
                    clip_name = f"{evt.event_id}_track{evt.track_id}.mp4"

                    self._events.append({
                        "event_id": evt.event_id,
                        "frame_start": evt.frame_start,
                        "frame_end": evt.frame_end,
                        "track_id": evt.track_id,
                        "plate_text": plate,
                        "turn_signal": turn_signal,
                        "clip_filename": clip_name,
                    })

                    # Start clip writer (pre-buffer + post-buffer)
                    clip_path = str(self.clips_dir / clip_name)
                    writer = cv2.VideoWriter(
                        clip_path, _FOURCC, fps, (frame_w, frame_h)
                    )
                    # Write pre-buffered frames from ring
                    for pre_frame in ring[-buffer_frames:]:
                        writer.write(pre_frame)
                    active_writers[evt.event_id] = {
                        "writer": writer,
                        "frames_remaining": buffer_frames,
                    }

                # -- Write to active clip writers -----------------------
                for eid, state in list(active_writers.items()):
                    state["writer"].write(frame)
                    state["frames_remaining"] -= 1
                    if state["frames_remaining"] <= 0:
                        state["writer"].release()
                        del active_writers[eid]

                # -- Ring buffer ----------------------------------------
                ring.append(frame.copy())
                if len(ring) > buffer_frames:
                    ring.pop(0)

                frame_idx += 1
                if frame_idx % 30 == 0 and total_frames > 0:
                    frac = frame_idx / total_frames
                    self._progress(frac, f"Frame {frame_idx}/{total_frames}")

        finally:
            cap.release()
            for state in active_writers.values():
                state["writer"].release()

        # -- Write result.json ------------------------------------------
        finished = time.time()
        result = {
            "job_id": self.job_id,
            "status": "done",
            "events": self._events,
            "created_at": finished,
            "finished_at": finished,
        }
        (self.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
