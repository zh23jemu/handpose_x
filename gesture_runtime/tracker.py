from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from gesture_runtime.config import TrackerConfig
from gesture_runtime.types import HandDetection, TrackedHand


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _bbox_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    left = max(bbox_a[0], bbox_b[0])
    top = max(bbox_a[1], bbox_b[1])
    right = min(bbox_a[2], bbox_b[2])
    bottom = min(bbox_a[3], bbox_b[3])
    inter_w = max(0, right - left)
    inter_h = max(0, bottom - top)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(1, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = max(1, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    return float(intersection / (area_a + area_b - intersection))


def estimate_palm_center(landmarks: np.ndarray) -> np.ndarray:
    return np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)


def estimate_palm_size(landmarks: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.linalg.norm(landmarks[0] - landmarks[5]),
                np.linalg.norm(landmarks[0] - landmarks[9]),
                np.linalg.norm(landmarks[0] - landmarks[13]),
                np.linalg.norm(landmarks[0] - landmarks[17]),
            ]
        )
    )


def bbox_from_landmarks(frame_shape: Tuple[int, int, int], landmarks: np.ndarray, padding_ratio: float) -> Tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x_min = float(np.min(landmarks[:, 0]))
    y_min = float(np.min(landmarks[:, 1]))
    x_max = float(np.max(landmarks[:, 0]))
    y_max = float(np.max(landmarks[:, 1]))
    side = max(x_max - x_min, y_max - y_min)
    side = max(24.0, side * (1.0 + padding_ratio * 2.0))
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return (
        int(np.clip(center_x - side / 2.0, 0, frame_w - 1)),
        int(np.clip(center_y - side / 2.0, 0, frame_h - 1)),
        int(np.clip(center_x + side / 2.0, 0, frame_w - 1)),
        int(np.clip(center_y + side / 2.0, 0, frame_h - 1)),
    )


@dataclass
class _TrackMemory:
    track_id: int
    last_seen: float
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: np.ndarray
    palm_size: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    handedness: str = "unknown"
    handedness_score: float = 0.0
    age: int = 1


class HandTracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.next_track_id = 1
        self.tracks: Dict[int, _TrackMemory] = {}

    def update(self, frame_shape: Tuple[int, int, int], detections: List[HandDetection], timestamp: float) -> List[TrackedHand]:
        active_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen <= self.config.track_ttl
        ]
        active_tracks = {track_id: self.tracks[track_id] for track_id in active_track_ids}
        unmatched_tracks = set(active_tracks.keys())
        tracked_hands: List[TrackedHand] = []

        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            new_center = estimate_palm_center(detection.landmarks)
            new_palm_size = max(estimate_palm_size(detection.landmarks), 1.0)
            matched_track_id: Optional[int] = None
            best_score = float("inf")
            for track_id in list(unmatched_tracks):
                track = active_tracks[track_id]
                max_distance = max(
                    self.config.min_match_distance,
                    self.config.match_distance_ratio * max(track.palm_size, new_palm_size),
                )
                center_distance = _distance(track.center, new_center)
                bbox_iou = _bbox_iou(track.bbox, detection.bbox)
                if center_distance > max_distance and bbox_iou < self.config.min_match_iou:
                    continue
                score = center_distance / max(max_distance, 1.0) - bbox_iou
                if score < best_score:
                    matched_track_id = track_id
                    best_score = score

            if matched_track_id is None:
                matched_track_id = self.next_track_id
                self.next_track_id += 1
                smoothed_landmarks = detection.landmarks.copy()
                smoothed_bbox = bbox_from_landmarks(frame_shape, smoothed_landmarks, self.config.bbox_padding_ratio)
                track = _TrackMemory(
                    track_id=matched_track_id,
                    last_seen=timestamp,
                    landmarks=smoothed_landmarks,
                    bbox=smoothed_bbox,
                    center=estimate_palm_center(smoothed_landmarks),
                    palm_size=max(estimate_palm_size(smoothed_landmarks), 1.0),
                    handedness=detection.handedness,
                    handedness_score=detection.handedness_score,
                )
                self.tracks[matched_track_id] = track
            else:
                unmatched_tracks.discard(matched_track_id)
                track = self.tracks[matched_track_id]
                alpha = float(self.config.landmark_smoothing)
                prev_center = track.center.copy()
                prev_seen = track.last_seen
                track.landmarks = (1.0 - alpha) * track.landmarks + alpha * detection.landmarks
                track.bbox = bbox_from_landmarks(frame_shape, track.landmarks, self.config.bbox_padding_ratio)
                track.center = estimate_palm_center(track.landmarks)
                track.palm_size = max(estimate_palm_size(track.landmarks), 1.0)
                dt = max(1e-3, timestamp - prev_seen)
                track.velocity = (track.center - prev_center) / dt
                track.last_seen = timestamp
                track.age += 1
                if detection.handedness != "unknown":
                    track.handedness = detection.handedness
                    track.handedness_score = detection.handedness_score

            tracked_hands.append(
                TrackedHand(
                    bbox=track.bbox,
                    landmarks=track.landmarks.copy(),
                    score=detection.score,
                    handedness=track.handedness,
                    handedness_score=track.handedness_score,
                    detector=detection.detector,
                    details=dict(detection.details),
                    track_id=track.track_id,
                    center=track.center.copy(),
                    palm_size=track.palm_size,
                    velocity=track.velocity.copy(),
                    age=track.age,
                )
            )

        self._cleanup(timestamp)
        tracked_hands.sort(key=lambda item: item.track_id)
        return tracked_hands

    def _cleanup(self, timestamp: float) -> None:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen > self.config.track_ttl
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)
