# output/violation_logger.py - CSV/JSON violation log

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
import config
from analyzers.violation_checker import Violation


_CSV_FIELDS = [
    "timestamp_sec",
    "frame_number",
    "tracker_id",
    "violation_type",
    "recorded_at",
]


class ViolationLogger:
    def __init__(self, output_dir: str = config.OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self._csv_path = os.path.join(output_dir, "violation_log.csv")
        self._json_path = os.path.join(output_dir, "violation_log.json")
        self._violations: list[dict] = []
        self._init_csv()

    def log(self, violation: Violation):
        row = {
            "timestamp_sec": f"{violation.timestamp_sec:.3f}",
            "frame_number": violation.frame_number,
            "tracker_id": violation.tracker_id,
            "violation_type": violation.violation_type,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._violations.append(row)
        self._append_csv(row)
        self._rewrite_json()
        print(
            f"[VIOLATION] frame={violation.frame_number} "
            f"t={violation.timestamp_sec:.1f}s "
            f"vehicle_id={violation.tracker_id}"
        )

    def summary(self) -> int:
        return len(self._violations)

    # ------------------------------------------------------------------

    def _init_csv(self):
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()

    def _append_csv(self, row: dict):
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(row)

    def _rewrite_json(self):
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(self._violations, f, indent=2, ensure_ascii=False)
