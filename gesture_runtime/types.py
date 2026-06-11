from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from gesture_runtime.labels import get_action_hint_labels_cn, get_gesture_label_cn


BBox = Tuple[int, int, int, int]


@dataclass
class HandDetection:
    bbox: BBox
    landmarks: np.ndarray
    score: float = 1.0
    handedness: str = "unknown"
    handedness_score: float = 0.0
    detector: str = "unknown"
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrackedHand(HandDetection):
    track_id: int = -1
    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    palm_size: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    age: int = 0


@dataclass
class GestureObservation:
    track_id: int
    bbox: BBox
    landmarks: np.ndarray
    center: np.ndarray
    palm_size: float
    finger_states: Dict[str, bool]
    raw_static_gesture: str
    static_confidence: float
    pinch_ratio: float
    palm_angle: float
    handedness: str = "unknown"
    handedness_score: float = 0.0
    static_gesture: str = "unknown"
    action_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "bbox": [int(value) for value in self.bbox],
            "center": [round(float(self.center[0]), 1), round(float(self.center[1]), 1)],
            "palm_size": round(float(self.palm_size), 3),
            "gesture": self.static_gesture,
            "gesture_cn": get_gesture_label_cn(self.static_gesture),
            "raw_gesture": self.raw_static_gesture,
            "raw_gesture_cn": get_gesture_label_cn(self.raw_static_gesture),
            "confidence": round(float(self.static_confidence), 3),
            "finger_states": self.finger_states,
            "handedness": self.handedness,
            "handedness_score": round(float(self.handedness_score), 3),
            "action_hints": self.action_hints,
            "action_hints_cn": get_action_hint_labels_cn(self.action_hints),
        }
