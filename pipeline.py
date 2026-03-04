# pipeline.py - Main analysis pipeline

import cv2
import numpy as np
import supervision as sv

import config
from detectors.vehicle_detector import VehicleDetector
from detectors.lane_detector import LaneDetector
from detectors.blinker_detector import BlinkerDetector
from analyzers.lane_change import LaneChangeAnalyzer
from analyzers.violation_checker import ViolationChecker
from output.violation_logger import ViolationLogger
from output.clip_saver import ClipSaver
from utils.visualization import Visualizer


class Pipeline:
    def __init__(self, input_path: str, output_dir: str, show: bool = False):
        self.input_path = input_path
        self.output_dir = output_dir
        self.show = show

        self._cap = cv2.VideoCapture(input_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"[Pipeline] {self.width}x{self.height} @ {self.fps:.1f}fps  "
            f"frames={total}  source={input_path}"
        )
        print("[Pipeline] Controls: press Q in preview window to quit\n")

        self._vehicle_det = VehicleDetector()
        self._lane_det = LaneDetector()
        self._blinker_det = BlinkerDetector()
        self._lane_analyzer = LaneChangeAnalyzer()
        self._violation_checker = ViolationChecker(self.fps)
        self._logger = ViolationLogger(output_dir)
        self._clip_saver = ClipSaver(self.fps, self.width, self.height,
                                     output_dir=f"{output_dir}/clips")
        self._viz = Visualizer()

        self._violation_ids: set[int] = set()
        self._violation_clear: dict[int, int] = {}
        # (clear_at_frame, tracker_id, timestamp_sec)
        self._flash_queue: list[tuple[int, int, float]] = []
        # tracker_id -> current lane index
        self._lane_states: dict[int, int] = {}

    def run(self):
        frame_number = 0
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame_number += 1
                self._process_frame(frame, frame_number)

                if self.show:
                    cv2.imshow("Dashcam Analyzer  [Q=quit]", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Pipeline] Interrupted by user.")
                        break

        finally:
            self._cap.release()
            self._clip_saver.finalize()
            if self.show:
                cv2.destroyAllWindows()
            total = self._logger.summary()
            print(f"\n{'='*50}")
            print(f"[Pipeline] Analysis complete.")
            print(f"[Pipeline] Total violations: {total}")
            print(f"[Pipeline] Log  -> {self.output_dir}/violation_log.csv")
            print(f"[Pipeline] Clips -> {self.output_dir}/clips/")
            print(f"{'='*50}")

    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, frame_number: int):
        # 1. Detect + track vehicles
        detections = self._vehicle_det.detect_and_track(frame)

        # 2. Detect lane boundaries
        lane_xs = self._lane_det.detect(frame)

        active_ids: set[int] = set()
        blinker_states: dict[int, bool] = {}

        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                tid = int(tracker_id)
                active_ids.add(tid)
                bbox = detections.xyxy[i]
                cx = (bbox[0] + bbox[2]) / 2.0

                # 3. Blinker state
                blinker_on = self._blinker_det.update(frame, tid, bbox)
                blinker_states[tid] = blinker_on
                self._violation_checker.record_blinker(tid, frame_number, blinker_on)

                # 4. Lane change detection
                event = self._lane_analyzer.update(tid, frame_number, cx, lane_xs)
                self._lane_states[tid] = self._lane_analyzer._last_lane.get(tid, -1)

                if event is not None:
                    ts = frame_number / self.fps
                    blink_str = "BLINK ON" if blinker_on else "no blink"
                    print(
                        f"[LANE CHANGE] frame={frame_number} t={ts:.1f}s "
                        f"vehicle_id={tid}  L{event.from_lane}->L{event.to_lane}  "
                        f"blinker={blink_str}"
                    )

                    # 5. Violation check
                    violation = self._violation_checker.check(event)
                    if violation is not None:
                        self._logger.log(violation)
                        self._clip_saver.trigger(violation)
                        self._violation_ids.add(tid)
                        self._violation_clear[tid] = frame_number + int(self.fps * 3)
                        self._flash_queue.append(
                            (frame_number + int(self.fps * 2), tid, violation.timestamp_sec)
                        )

        # Expire violation highlights
        expired = [tid for tid, clear_at in self._violation_clear.items()
                   if frame_number >= clear_at]
        for tid in expired:
            self._violation_ids.discard(tid)
            del self._violation_clear[tid]

        # Purge stale tracker state
        all_known = (
            set(self._lane_analyzer._last_lane.keys())
            | set(self._blinker_det._history.keys())
        )
        for tid in all_known - active_ids:
            self._lane_analyzer.remove(tid)
            self._blinker_det.remove(tid)
            self._violation_checker.remove(tid)
            self._lane_states.pop(tid, None)

        # 6. Draw overlays onto frame (always — so saved clips include annotations)
        self._viz.draw_lanes(frame, lane_xs)
        self._viz.draw_vehicles(
            frame, detections, blinker_states,
            self._violation_ids, self._lane_states,
            frame_number=frame_number,
        )

        # Violation flash banners
        still_active = []
        for clear_at, tid, ts in self._flash_queue:
            if frame_number <= clear_at:
                self._viz.draw_violation_flash(frame, tid, ts)
                still_active.append((clear_at, tid, ts))
        self._flash_queue[:] = still_active

        self._viz.draw_status_panel(
            frame, frame_number, self.fps,
            self._logger.summary(),
            blinker_states,
            self._lane_states,
        )

        # 7. Push annotated frame to clip ring-buffer
        #    ClipSaver only writes to disk when trigger() was called (violation).
        #    Frames with no pending clip writer are buffered and auto-discarded.
        self._clip_saver.push(frame, frame_number)
