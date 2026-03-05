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
import json, os, re, subprocess, sys, time
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

# H.264(avc1) 우선 — 브라우저 호환. 지원 안 되면 mp4v로 대체
_FOURCC_H264 = cv2.VideoWriter_fourcc(*"avc1")
_FOURCC_FALLBACK = cv2.VideoWriter_fourcc(*"mp4v")


def _reencode_h264(path: str) -> bool:
    """ffmpeg으로 H.264 MP4 재인코딩. 성공 시 True 반환."""
    tmp = path + ".reenc.mp4"
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                tmp,
            ],
            capture_output=True,
            timeout=120,
        )
        if r.returncode == 0 and Path(tmp).exists():
            os.replace(tmp, path)
            return True
        if r.returncode != 0:
            print(f"[ffmpeg] 재인코딩 실패: {r.stderr.decode(errors='replace')[-300:]}",
                  file=sys.stderr)
    except FileNotFoundError:
        print("[ffmpeg] ffmpeg를 찾을 수 없음 — mp4v 클립은 브라우저에서 재생되지 않을 수 있음",
              file=sys.stderr)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[ffmpeg] 재인코딩 오류: {e}", file=sys.stderr)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)
    return False


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

    def _extract_recording_date(
        self, cap: cv2.VideoCapture, frame_h: int, frame_w: int
    ) -> Optional[str]:
        """첫 프레임 상단 중앙 오버레이에서 날짜(YYYYMMDD)를 추출. 실패 시 None 반환."""
        if not self.plate_detector._ensure_reader():
            return None
        reader = self.plate_detector._reader

        h_crop = max(1, int(frame_h * 0.12))
        w_start = int(frame_w * 0.20)
        w_end = int(frame_w * 0.80)

        saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        recording_date = None

        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            crop = frame[:h_crop, w_start:w_end]
            crop = cv2.resize(
                crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                interpolation=cv2.INTER_CUBIC,
            )
            try:
                results = reader.readtext(crop, detail=0, paragraph=True)
                text = " ".join(results)
                # YYYY-MM-DD 또는 YYYY/MM/DD 형식에서 날짜만 추출
                m = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', text)
                if m:
                    recording_date = m.group(1) + m.group(2) + m.group(3)
            except Exception as e:
                print(f"[WebPipeline] 날짜 OCR 오류: {e}", file=sys.stderr)
            if recording_date:
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
        return recording_date

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

        self._progress(0.0, "촬영일 추출 중…")
        recording_date = self._extract_recording_date(cap, frame_h, frame_w)
        if recording_date:
            print(f"[WebPipeline] 촬영일(오버레이): {recording_date}", file=sys.stderr)
        else:
            # 파일명에서 YYYYMMDD 추출 (예: 20260106-20h03m04s_myN.avi)
            fname = Path(video_path).stem
            m = re.search(r'(\d{8})', fname)
            if m:
                recording_date = m.group(1)
                print(f"[WebPipeline] 촬영일(파일명 폴백): {recording_date}", file=sys.stderr)
            else:
                print("[WebPipeline] 촬영일 추출 실패 — 날짜 없이 진행", file=sys.stderr)

        ring: list[np.ndarray] = []
        frame_idx = 0

        # Active clip writers: event_id -> {writer, frames_remaining}
        active_writers: dict[str, dict] = {}
        # mp4v fallback 사용 클립 경로 → 루프 후 ffmpeg 재인코딩 대상
        _reenc_paths: list[str] = []

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

                    # writer 먼저 생성 → clip_filename 결정
                    clip_path = str(self.clips_dir / clip_name)
                    writer = cv2.VideoWriter(
                        clip_path, _FOURCC_H264, fps, (frame_w, frame_h)
                    )
                    _used_fallback = False
                    if not writer.isOpened():
                        # H.264 미지원 환경 → mp4v fallback + 나중에 ffmpeg 재인코딩
                        writer = cv2.VideoWriter(
                            clip_path, _FOURCC_FALLBACK, fps, (frame_w, frame_h)
                        )
                        _used_fallback = True
                    clip_filename = clip_name if writer.isOpened() else None
                    if clip_filename and _used_fallback:
                        _reenc_paths.append(clip_path)

                    self._events.append({
                        "event_id": evt.event_id,
                        "frame_start": evt.frame_start,
                        "frame_end": evt.frame_end,
                        "track_id": evt.track_id,
                        "plate_text": plate,
                        "turn_signal": turn_signal,
                        "clip_filename": clip_filename,
                    })

                    if writer.isOpened():
                        # 사전 버퍼 프레임 기록
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

        # -- ffmpeg 재인코딩 (mp4v fallback 클립 → H.264) -----------------
        if _reenc_paths:
            self._progress(0.98, "클립 H.264 재인코딩 중…")
            for cp in _reenc_paths:
                _reencode_h264(cp)

        # -- Write result.json ------------------------------------------
        finished = time.time()
        result = {
            "job_id": self.job_id,
            "status": "done",
            "recording_date": recording_date,
            "events": self._events,
            "created_at": finished,
            "finished_at": finished,
        }
        (self.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
