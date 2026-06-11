from typing import List

import cv2
import numpy as np

from gesture_runtime.backends.base import HandPoseBackend
from gesture_runtime.config import BackendConfig
from gesture_runtime.types import HandDetection


class MediaPipeHandBackend(HandPoseBackend):
    name = "mediapipe"

    def __init__(self, config: BackendConfig):
        self.config = config
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe backend requested but mediapipe is not installed. "
                "Install it with: pip install mediapipe"
            ) from exc
        self._mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            model_complexity=config.mediapipe_model_complexity,
            min_detection_confidence=config.mediapipe_min_detection_confidence,
            min_tracking_confidence=config.mediapipe_min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        detections: List[HandDetection] = []
        if result.multi_hand_landmarks is None:
            return detections
        handedness_list = result.multi_handedness or []
        frame_h, frame_w = frame.shape[:2]
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            coords = []
            for landmark in hand_landmarks.landmark:
                coords.append(
                    [
                        float(np.clip(landmark.x * frame_w, 0, frame_w - 1)),
                        float(np.clip(landmark.y * frame_h, 0, frame_h - 1)),
                    ]
                )
            landmarks = np.asarray(coords, dtype=np.float32)
            x_min = float(np.min(landmarks[:, 0]))
            y_min = float(np.min(landmarks[:, 1]))
            x_max = float(np.max(landmarks[:, 0]))
            y_max = float(np.max(landmarks[:, 1]))
            side = max(x_max - x_min, y_max - y_min) * 1.25
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            bbox = (
                int(np.clip(center_x - side / 2.0, 0, frame_w - 1)),
                int(np.clip(center_y - side / 2.0, 0, frame_h - 1)),
                int(np.clip(center_x + side / 2.0, 0, frame_w - 1)),
                int(np.clip(center_y + side / 2.0, 0, frame_h - 1)),
            )
            handedness = "unknown"
            score = 0.95
            if idx < len(handedness_list) and handedness_list[idx].classification:
                handedness = handedness_list[idx].classification[0].label.lower()
                score = float(handedness_list[idx].classification[0].score)
            detections.append(
                HandDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    score=score,
                    handedness=handedness,
                    handedness_score=score,
                    detector=self.name,
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.config.max_hands]

    def close(self) -> None:
        self.hands.close()
