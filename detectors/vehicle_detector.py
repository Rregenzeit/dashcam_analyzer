# detectors/vehicle_detector.py - YOLOv8 detection + ByteTrack tracking

import numpy as np
from ultralytics import YOLO
import supervision as sv
import config


class VehicleDetector:
    def __init__(self, model_path: str = config.YOLO_MODEL):
        self.model = YOLO(model_path)
        self.device = config.DEVICE
        print(f"[VehicleDetector] device={self.device}")
        self.tracker = sv.ByteTrack(
            lost_track_buffer=config.TRACKER_MAX_AGE
        )
        self.vehicle_classes = set(config.VEHICLE_CLASSES)

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO detection and ByteTrack on a single frame.

        Returns sv.Detections with tracker_id populated.
        """
        results = self.model(
            frame,
            conf=config.CONF_THRESHOLD,
            device=self.device,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Keep only vehicle classes
        vehicle_mask = np.isin(detections.class_id, list(self.vehicle_classes))
        detections = detections[vehicle_mask]

        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        return detections
