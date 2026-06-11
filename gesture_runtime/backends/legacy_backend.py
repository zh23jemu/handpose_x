from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from gesture_runtime.backends.base import HandPoseBackend
from gesture_runtime.config import BackendConfig
from gesture_runtime.types import HandDetection
from models.rexnetv1 import ReXNetV1


def _build_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = ReXNetV1(num_classes=42)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    inter_w = max(0, right - left)
    inter_h = max(0, bottom - top)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(intersection / (area_a + area_b - intersection))


class LegacyHandPoseBackend(HandPoseBackend):
    name = "legacy"

    def __init__(self, config: BackendConfig):
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA backend requested but not available")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = _build_model(config.model_path, self.device)
        self.frame_index = 0
        self.tracked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.track_miss_count = 0

    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        self.frame_index += 1
        detections: List[HandDetection] = []
        if self.config.max_hands == 1 and self.tracked_bbox is not None:
            prediction = self._predict_landmarks(frame, self.tracked_bbox)
            if prediction is not None:
                landmarks, crop_bbox = prediction
                span_ratio = self._landmark_span_ratio(landmarks, crop_bbox)
                if span_ratio >= max(0.18, self.config.min_landmark_span_ratio * 0.85):
                    refined_bbox = self._bbox_from_landmarks(frame.shape, landmarks)
                    detections = [
                        HandDetection(
                            bbox=refined_bbox,
                            landmarks=landmarks,
                            score=max(0.10, self._detection_score(frame.shape, refined_bbox, landmarks, span_ratio)),
                            detector=self.name,
                        )
                    ]
                    self.tracked_bbox = refined_bbox
                    self.track_miss_count = 0
                else:
                    self.track_miss_count += 1
            else:
                self.track_miss_count += 1
            if self.track_miss_count > self.config.track_miss_limit:
                self.tracked_bbox = None

        should_redetect = (
            not detections
            or self.frame_index % max(1, self.config.redetect_interval) == 0
            or self.config.max_hands > 1
        )
        if should_redetect:
            candidates = []
            for bbox in self._detect_hands(frame):
                prediction = self._predict_landmarks(frame, bbox)
                if prediction is None:
                    continue
                landmarks, crop_bbox = prediction
                span_ratio = self._landmark_span_ratio(landmarks, crop_bbox)
                if span_ratio < self.config.min_landmark_span_ratio:
                    continue
                refined_bbox = self._bbox_from_landmarks(frame.shape, landmarks)
                score = self._detection_score(frame.shape, refined_bbox, landmarks, span_ratio)
                candidates.append(
                    HandDetection(
                        bbox=refined_bbox,
                        landmarks=landmarks,
                        score=score,
                        detector=self.name,
                    )
                )
            candidates.sort(key=lambda item: item.score, reverse=True)
            if self.config.max_hands == 1:
                detections = candidates[:1]
            else:
                detections = candidates[: self.config.max_hands]
            if detections and self.config.max_hands == 1:
                self.tracked_bbox = detections[0].bbox
                self.track_miss_count = 0
        return detections

    def _detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv_mask = cv2.inRange(
            hsv,
            np.array([0, 15, 40], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        ycrcb_mask = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )
        mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_hand_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            padding = int(0.08 * max(width, height))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1] - 1, x + width + padding)
            y2 = min(frame.shape[0] - 1, y + height + padding)
            bbox_width = max(1, x2 - x1)
            bbox_height = max(1, y2 - y1)
            bbox_area_ratio = (bbox_width * bbox_height) / frame_area
            if bbox_area_ratio > self.config.max_bbox_area_ratio:
                continue
            if (bbox_width / max(1, frame.shape[1])) > self.config.max_bbox_width_ratio:
                continue
            if (bbox_height / max(1, frame.shape[0])) > self.config.max_bbox_height_ratio:
                continue
            aspect_ratio = bbox_width / max(1.0, float(bbox_height))
            if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                continue
            candidates.append((x1, y1, x2, y2, area))
        candidates.sort(key=lambda item: item[4], reverse=True)
        selected: List[Tuple[int, int, int, int]] = []
        for candidate in candidates:
            bbox = candidate[:4]
            if any(_iou(bbox, existing) > 0.35 for existing in selected):
                continue
            selected.append(bbox)
            if len(selected) >= max(2, self.config.max_hands):
                break
        return selected

    def _square_bbox(self, frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int], scale: float = 1.15) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        width = float(x2 - x1)
        height = float(y2 - y1)
        side = max(width, height) * scale
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        crop_x1 = int(center_x - side / 2.0)
        crop_y1 = int(center_y - side / 2.0)
        crop_x2 = int(center_x + side / 2.0)
        crop_y2 = int(center_y + side / 2.0)
        crop_x1 = int(np.clip(crop_x1, 0, frame_shape[1] - 1))
        crop_y1 = int(np.clip(crop_y1, 0, frame_shape[0] - 1))
        crop_x2 = int(np.clip(crop_x2, 0, frame_shape[1] - 1))
        crop_y2 = int(np.clip(crop_y2, 0, frame_shape[0] - 1))
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _predict_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        crop_bbox = self._square_bbox(frame.shape, bbox)
        x1, y1, x2, y2 = crop_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (self.config.img_size, self.config.img_size), interpolation=cv2.INTER_CUBIC)
        normalized = (resized.astype(np.float32) - 128.0) / 256.0
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor.float()).detach().cpu().numpy().reshape(-1)
        crop_width = max(1, x2 - x1)
        crop_height = max(1, y2 - y1)
        landmarks = []
        for index in range(21):
            landmarks.append(
                [
                    float(output[index * 2]) * crop_width + x1,
                    float(output[index * 2 + 1]) * crop_height + y1,
                ]
            )
        return np.asarray(landmarks, dtype=np.float32), crop_bbox

    def _landmark_span_ratio(self, landmarks: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        bbox_side = max(1.0, float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])))
        span_x = float(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]))
        span_y = float(np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]))
        return max(span_x, span_y) / bbox_side

    def _bbox_from_landmarks(self, frame_shape: Tuple[int, int, int], landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        x_min = float(np.min(landmarks[:, 0]))
        y_min = float(np.min(landmarks[:, 1]))
        x_max = float(np.max(landmarks[:, 0]))
        y_max = float(np.max(landmarks[:, 1]))
        span = max(x_max - x_min, y_max - y_min, float(self.config.track_min_side))
        side = span * self.config.track_bbox_scale
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        x1 = int(np.clip(center_x - side / 2.0, 0, frame_shape[1] - 1))
        y1 = int(np.clip(center_y - side / 2.0, 0, frame_shape[0] - 1))
        x2 = int(np.clip(center_x + side / 2.0, 0, frame_shape[1] - 1))
        y2 = int(np.clip(center_y + side / 2.0, 0, frame_shape[0] - 1))
        return x1, y1, x2, y2

    def _detection_score(self, frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int], landmarks: np.ndarray, span_ratio: float) -> float:
        frame_height, frame_width = frame_shape[:2]
        frame_diag = max(1.0, float((frame_width ** 2 + frame_height ** 2) ** 0.5))
        center = np.mean(landmarks, axis=0)
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        center_penalty = float(np.linalg.norm(center - frame_center) / frame_diag)
        edge_touch_penalty = 0.0
        if bbox[0] <= 2 or bbox[1] <= 2:
            edge_touch_penalty += 0.18
        if bbox[2] >= frame_width - 3 or bbox[3] >= frame_height - 3:
            edge_touch_penalty += 0.18
        bbox_area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(1.0, float(frame_width * frame_height))
        area_penalty = max(0.0, bbox_area_ratio - 0.28) * 0.8
        return span_ratio * 2.0 - center_penalty * 0.35 - edge_touch_penalty - area_penalty
