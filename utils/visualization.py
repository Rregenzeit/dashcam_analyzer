# utils/visualization.py - On-screen overlay drawing

import cv2
import numpy as np
import supervision as sv
import config


class Visualizer:
    def draw_vehicles(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        blinker_states: dict[int, bool],
        violation_ids: set[int],
        lane_states: dict[int, int] | None = None,
        frame_number: int = 0,
    ) -> np.ndarray:
        if detections.tracker_id is None:
            return frame

        for i, tracker_id in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            tid = int(tracker_id)
            is_violation = tid in violation_ids
            blinker_on = blinker_states.get(tid, False)
            lane_idx = (lane_states or {}).get(tid, -1)

            if is_violation:
                self._draw_violation_tag(frame, x1, y1, x2, y2, tid, frame_number)
            else:
                self._draw_normal_tag(frame, x1, y1, x2, y2, tid, blinker_on, lane_idx)

        return frame

    # ------------------------------------------------------------------
    # Violation vehicle: very prominent
    # ------------------------------------------------------------------

    def _draw_violation_tag(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        tid: int,
        frame_number: int,
    ):
        # Pulse: alternate between two red shades every 10 frames
        pulse = (frame_number // 10) % 2 == 0
        box_color   = (0, 0, 255) if pulse else (0, 0, 180)
        badge_color = (0, 0, 220) if pulse else (0, 0, 160)

        # 1. Thick red bounding box (4 px outer + 2 px inner white gap effect)
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), box_color, 4)

        # 2. Semi-transparent red fill overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # 3. Corner L-shaped targeting markers
        arm = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 3
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for cx, cy in corners:
            dx = arm if cx == x1 else -arm
            dy = arm if cy == y1 else -arm
            cv2.line(frame, (cx, cy), (cx + dx, cy), box_color, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + dy), box_color, thickness)

        # 4. Large badge at top-center: "!! 위반 NO BLINKER"
        badge_text = f"!! VIOLATION  ID:{tid}  NO BLINKER"
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        font_thick = 2
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)
        bx = x1 + (x2 - x1 - tw) // 2
        by = max(y1 - 10, th + 8)
        pad = 6
        cv2.rectangle(
            frame,
            (bx - pad, by - th - pad),
            (bx + tw + pad, by + baseline + pad),
            badge_color, -1,
        )
        cv2.rectangle(
            frame,
            (bx - pad, by - th - pad),
            (bx + tw + pad, by + baseline + pad),
            (255, 255, 255), 1,
        )
        cv2.putText(frame, badge_text, (bx, by), font, font_scale,
                    (255, 255, 255), font_thick, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Normal vehicle: compact info label
    # ------------------------------------------------------------------

    def _draw_normal_tag(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        tid: int,
        blinker_on: bool,
        lane_idx: int,
    ):
        box_color = (0, 215, 255) if blinker_on else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        lane_str  = f"L{lane_idx}" if lane_idx >= 0 else "L?"
        blink_str = "BLINK ON" if blinker_on else "no blink"
        label     = f"ID:{tid} {lane_str} [{blink_str}]"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        lx = x1
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (lx + 2, ly), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Blinker dot
        dot_color = (0, 215, 255) if blinker_on else (60, 60, 60)
        cv2.circle(frame, (x2 - 8, y1 + 8), 6, dot_color, -1)
        cv2.circle(frame, (x2 - 8, y1 + 8), 6, (255, 255, 255), 1)

    # ------------------------------------------------------------------

    def draw_lanes(self, frame: np.ndarray, lane_xs: list[float]) -> np.ndarray:
        h = frame.shape[0]
        for idx, x in enumerate(lane_xs):
            ix = int(x)
            cv2.line(frame, (ix, 0), (ix, h), config.LANE_LINE_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Lane{idx}|{idx+1}", (ix + 3, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        config.LANE_LINE_COLOR, 1, cv2.LINE_AA)
        return frame

    def draw_violation_flash(
        self, frame: np.ndarray, tracker_id: int, timestamp_sec: float
    ) -> np.ndarray:
        """Full-width red banner at top of frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 72), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        text = f"!! VIOLATION  Vehicle ID:{tracker_id}  t={timestamp_sec:.1f}s  (no blinker on lane change)"
        cv2.putText(frame, text, (12, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_status_panel(
        self,
        frame: np.ndarray,
        frame_number: int,
        fps: float,
        violation_count: int,
        blinker_states: dict[int, bool],
        lane_states: dict[int, int],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_h = min(32 + max(len(blinker_states), 1) * 22, 160)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        ts = frame_number / fps if fps > 0 else 0
        header = (f"Frame:{frame_number}  Time:{ts:.1f}s  "
                  f"Vehicles:{len(blinker_states)}  Violations:{violation_count}")
        cv2.putText(frame, header, (8, h - panel_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (200, 200, 200), 1, cv2.LINE_AA)

        x_cursor = 8
        for tid, blink_on in list(blinker_states.items())[:10]:
            lane_idx  = lane_states.get(tid, -1)
            lane_str  = f"L{lane_idx}" if lane_idx >= 0 else "L?"
            blink_lbl = "BLINK" if blink_on else "-----"
            color     = (0, 215, 255) if blink_on else (120, 120, 120)
            entry     = f"ID:{tid} {lane_str} [{blink_lbl}]"
            cv2.putText(frame, entry, (x_cursor, h - panel_h + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            x_cursor += 185
            if x_cursor + 185 > w:
                break
        return frame
