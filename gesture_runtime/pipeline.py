import time
from typing import List, Tuple

import numpy as np

from gesture_runtime.backends import create_backend
from gesture_runtime.config import RuntimeConfig
from gesture_runtime.hybrid_recognizer import GestureRecognizer
from gesture_runtime.tracker import HandTracker
from gesture_runtime.types import GestureObservation


class GesturePipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.backend = create_backend(config.backend)
        self.tracker = HandTracker(config.tracker)
        self.recognizer = GestureRecognizer(
            event_cooldown=config.recognizer.event_cooldown,
            track_ttl=config.tracker.track_ttl,
            swipe_time_window=config.recognizer.swipe_time_window,
            zoom_time_window=config.recognizer.zoom_time_window,
            rotation_time_window=config.recognizer.rotation_time_window,
            min_recognition_confidence=config.recognizer.min_gesture_confidence,
            temporal_model_path=config.recognizer.temporal_model_path,
            temporal_window_size=config.recognizer.temporal_window_size,
            temporal_confidence=config.recognizer.temporal_confidence,
            temporal_stride=config.recognizer.temporal_stride,
        )

    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> Tuple[List[GestureObservation], List[dict]]:
        if timestamp <= 0.0:
            timestamp = time.time()
        detections = self.backend.detect(frame)
        tracked_hands = self.tracker.update(frame.shape, detections, timestamp)
        return self.recognizer.update(tracked_hands, timestamp)

    def close(self) -> None:
        self.backend.close()
