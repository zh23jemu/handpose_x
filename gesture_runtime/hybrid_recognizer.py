import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gesture_runtime.config import RecognizerConfig
from gesture_runtime.labels import ACTION_HINTS, FINGER_NAMES, NON_THUMB_FINGERS, STATIC_EVENT_LABELS, get_action_hint_labels_cn, get_gesture_label_cn
from gesture_runtime.temporal_model import TemporalGestureClassifier, stack_sequence_features
from gesture_runtime.types import GestureObservation, TrackedHand


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    vec_ba = point_a - point_b
    vec_bc = point_c - point_b
    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0
    cosine = float(np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _angle_delta(start_angle: float, end_angle: float) -> float:
    delta = end_angle - start_angle
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def _orientation(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    return float(
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )


def _segments_intersect(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray, point_d: np.ndarray) -> bool:
    o1 = _orientation(point_a, point_b, point_c)
    o2 = _orientation(point_a, point_b, point_d)
    o3 = _orientation(point_c, point_d, point_a)
    o4 = _orientation(point_c, point_d, point_b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _points_from_history(history: Deque[Tuple[float, np.ndarray]], time_window: float, timestamp: float) -> List[np.ndarray]:
    return [point.copy() for ts, point in history if timestamp - ts <= time_window]


def _values_from_history(history: Deque[Tuple[float, float]], time_window: float, timestamp: float) -> List[float]:
    return [value for ts, value in history if timestamp - ts <= time_window]


def _trend_consistency(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    diffs = np.diff(np.asarray(values, dtype=np.float32))
    non_zero = diffs[np.abs(diffs) > 1e-4]
    if len(non_zero) == 0:
        return 0.0
    positive = float(np.sum(non_zero > 0))
    negative = float(np.sum(non_zero < 0))
    return max(positive, negative) / max(1.0, float(len(non_zero)))


def _simplify_path(points: Sequence[np.ndarray], min_distance: float) -> List[np.ndarray]:
    if not points:
        return []
    simplified = [points[0]]
    for point in points[1:]:
        if _distance(point, simplified[-1]) >= min_distance:
            simplified.append(point)
    if len(simplified) == 1 and len(points) > 1:
        simplified.append(points[-1])
    return simplified


@dataclass
class _TrackState:
    track_id: int
    last_seen: float = 0.0
    label_history: Deque[Tuple[float, str]] = field(default_factory=lambda: deque(maxlen=18))
    center_history: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=64))
    index_history: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=64))
    pinch_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
    angle_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
    landmark_history: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=96))
    smoothed_label: str = "unknown"
    label_started_at: float = 0.0
    fist_started_at: Optional[float] = None
    emitted_at: Dict[str, float] = field(default_factory=dict)
    emitted_tokens: Dict[str, float] = field(default_factory=dict)
    swipe_anchor_time: Optional[float] = None
    swipe_anchor_point: Optional[np.ndarray] = None
    swipe_last_active: Optional[float] = None
    palm_anchor_time: Optional[float] = None
    palm_anchor_center: Optional[np.ndarray] = None
    palm_last_active: Optional[float] = None
    pinch_anchor_time: Optional[float] = None
    pinch_anchor_ratio: Optional[float] = None
    pinch_anchor_center: Optional[np.ndarray] = None
    pinch_last_active: Optional[float] = None
    rotation_anchor_time: Optional[float] = None
    rotation_anchor_angle: Optional[float] = None
    rotation_anchor_center: Optional[np.ndarray] = None
    rotation_last_active: Optional[float] = None


class GestureRecognizer:
    def __init__(self, event_cooldown: float = 0.9, track_ttl: float = 2.5, swipe_time_window: float = 0.75, zoom_time_window: float = 0.85, rotation_time_window: float = 0.7, min_recognition_confidence: float = 0.72, temporal_model_path: str = "", temporal_window_size: int = 32, temporal_confidence: float = 0.82, temporal_stride: int = 2):
        self.config = RecognizerConfig(
            event_cooldown=event_cooldown,
            min_gesture_confidence=min_recognition_confidence,
            swipe_time_window=swipe_time_window,
            zoom_time_window=zoom_time_window,
            rotation_time_window=rotation_time_window,
            temporal_model_path=temporal_model_path,
            temporal_window_size=temporal_window_size,
            temporal_confidence=temporal_confidence,
            temporal_stride=temporal_stride,
        )
        self.track_ttl = track_ttl
        self.states: Dict[int, _TrackState] = {}
        self.temporal_classifier = TemporalGestureClassifier(temporal_model_path, window_size=temporal_window_size) if temporal_model_path else None

    def update(self, tracked_hands: Sequence[TrackedHand], timestamp: float) -> Tuple[List[GestureObservation], List[Dict[str, object]]]:
        observations = [self._analyze_hand(hand) for hand in tracked_hands]
        events: List[Dict[str, object]] = []
        active_track_ids = set()
        for observation in observations:
            active_track_ids.add(observation.track_id)
            state = self.states.get(observation.track_id)
            if state is None:
                state = _TrackState(track_id=observation.track_id)
                self.states[observation.track_id] = state
            self._update_state(state, observation, timestamp)
            dynamic_event = (
                self._detect_cross(state, observation, timestamp)
                or self._detect_swipe(state, observation, timestamp)
                or self._detect_palm_swipe(state, observation, timestamp)
                or self._detect_zoom(state, observation, timestamp)
                or self._detect_rotation(state, observation, timestamp)
                or self._detect_open_then_fist(state, observation, timestamp)
                or self._detect_fist_hold(state, observation, timestamp)
            )
            if dynamic_event is None and self.temporal_classifier is not None:
                dynamic_event = self._detect_temporal_model_event(state, observation, timestamp)
            if dynamic_event is not None:
                events.append(dynamic_event)
                continue
            static_event = self._detect_static_event(state, observation, timestamp)
            if static_event is not None:
                events.append(static_event)
        self._cleanup(timestamp, active_track_ids)
        return observations, events

    def _analyze_hand(self, hand: TrackedHand) -> GestureObservation:
        finger_states = self._finger_states(hand.landmarks, hand.palm_size)
        pinch_ratio = _distance(hand.landmarks[4], hand.landmarks[8]) / max(hand.palm_size, 1.0)
        palm_angle = math.degrees(
            math.atan2(
                float(hand.landmarks[17][1] - hand.landmarks[5][1]),
                float(hand.landmarks[17][0] - hand.landmarks[5][0]),
            )
        )
        raw_gesture, confidence = self._classify_static_gesture(hand.landmarks, finger_states, pinch_ratio, hand.palm_size)
        return GestureObservation(
            track_id=hand.track_id,
            bbox=hand.bbox,
            landmarks=hand.landmarks.copy(),
            center=hand.center.copy(),
            palm_size=hand.palm_size,
            finger_states=finger_states,
            raw_static_gesture=raw_gesture,
            static_confidence=confidence,
            pinch_ratio=pinch_ratio,
            palm_angle=palm_angle,
            handedness=hand.handedness,
            handedness_score=hand.handedness_score,
        )

    def _finger_states(self, landmarks: np.ndarray, palm_size: float) -> Dict[str, bool]:
        states: Dict[str, bool] = {}
        thumb_angle_1 = _angle(landmarks[1], landmarks[2], landmarks[3])
        thumb_angle_2 = _angle(landmarks[2], landmarks[3], landmarks[4])
        thumb_extension = _distance(landmarks[4], landmarks[5])
        states["thumb"] = (
            thumb_angle_1 > 145.0
            and thumb_angle_2 > 145.0
            and thumb_extension > 0.38 * max(palm_size, 1.0)
        )
        finger_joints = {
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }
        for finger_name, (mcp, pip, dip, tip) in finger_joints.items():
            angle_pip = _angle(landmarks[mcp], landmarks[pip], landmarks[dip])
            angle_dip = _angle(landmarks[pip], landmarks[dip], landmarks[tip])
            finger_span = _distance(landmarks[tip], landmarks[mcp])
            states[finger_name] = (
                angle_pip > 150.0
                and angle_dip > 150.0
                and finger_span > 0.55 * max(palm_size, 1.0)
            )
        return states

    def _classify_static_gesture(self, landmarks: np.ndarray, finger_states: Dict[str, bool], pinch_ratio: float, palm_size: float) -> Tuple[str, float]:
        extended_count = sum(1 for name in FINGER_NAMES if finger_states[name])
        other_fingers_extended = all(finger_states[name] for name in ("middle", "ring", "pinky"))
        if pinch_ratio < 0.26 and other_fingers_extended:
            return "ok", 0.93
        if pinch_ratio < 0.22:
            return "pinch", 0.82
        if finger_states["index"] and not finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return "one", 0.9 if not finger_states["thumb"] else 0.82
        if all(finger_states[name] for name in NON_THUMB_FINGERS) and extended_count >= 4:
            return "palm_open", 0.88 if finger_states["thumb"] else 0.8
        finger_tip_distances = [
            _distance(landmarks[tip_idx], landmarks[0]) / max(palm_size, 1.0)
            for tip_idx in (4, 8, 12, 16, 20)
        ]
        if extended_count <= 1 and max(finger_tip_distances) < 1.75:
            return "fist", 0.85
        if finger_states["index"] and finger_states["pinky"] and not finger_states["middle"] and not finger_states["ring"]:
            return "love", 0.86
        return "unknown", 0.4

    def _update_state(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> None:
        state.last_seen = timestamp
        state.center_history.append((timestamp, observation.center.copy()))
        state.index_history.append((timestamp, observation.landmarks[8].copy()))
        state.pinch_history.append((timestamp, float(observation.pinch_ratio)))
        state.angle_history.append((timestamp, float(observation.palm_angle)))
        state.landmark_history.append((timestamp, observation.landmarks.copy()))
        if observation.static_confidence >= self.config.min_gesture_confidence:
            state.label_history.append((timestamp, observation.raw_static_gesture))
        else:
            state.label_history.append((timestamp, "unknown"))
        smoothed_label = self._smoothed_label(state.label_history)
        if smoothed_label != state.smoothed_label:
            state.smoothed_label = smoothed_label
            state.label_started_at = timestamp
            state.fist_started_at = timestamp if smoothed_label == "fist" else None
        elif smoothed_label == "fist" and state.fist_started_at is None:
            state.fist_started_at = timestamp
        observation.static_gesture = smoothed_label
        observation.action_hints = ACTION_HINTS.get(smoothed_label, [])
        self._update_motion_anchors(state, observation, timestamp)

    def _smoothed_label(self, label_history: Deque[Tuple[float, str]]) -> str:
        recent_labels = [label for _, label in list(label_history)[-5:] if label != "unknown"]
        if not recent_labels:
            return "unknown"
        label, count = Counter(recent_labels).most_common(1)[0]
        if count >= 2 or len(recent_labels) == 1:
            return label
        return recent_labels[-1]

    def _is_one_pose(self, observation: GestureObservation) -> bool:
        return observation.finger_states["index"] and not observation.finger_states["middle"] and not observation.finger_states["ring"] and not observation.finger_states["pinky"]

    def _is_palm_pose(self, observation: GestureObservation) -> bool:
        return (
            all(observation.finger_states[name] for name in NON_THUMB_FINGERS)
            or observation.static_gesture == "palm_open"
            or observation.raw_static_gesture == "palm_open"
        )

    def _is_pinch_pose(self, observation: GestureObservation) -> bool:
        return (
            observation.static_gesture in {"pinch", "ok"}
            or observation.raw_static_gesture in {"pinch", "ok"}
            or observation.static_gesture == "one"
            or observation.raw_static_gesture == "one"
            or observation.pinch_ratio < 1.7
        )

    def _update_motion_anchors(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> None:
        index_tip = observation.landmarks[8].copy()
        if self._is_one_pose(observation) or observation.raw_static_gesture == "one" or observation.static_gesture == "one":
            if state.swipe_anchor_time is None or (state.swipe_last_active is not None and timestamp - state.swipe_last_active > 0.45):
                state.swipe_anchor_time = timestamp
                state.swipe_anchor_point = index_tip
            state.swipe_last_active = timestamp
        elif state.swipe_last_active is not None and timestamp - state.swipe_last_active > 0.22:
            state.swipe_anchor_time = None
            state.swipe_anchor_point = None
            state.swipe_last_active = None
        if self._is_palm_pose(observation):
            if state.palm_anchor_time is None or (state.palm_last_active is not None and timestamp - state.palm_last_active > 0.55):
                state.palm_anchor_time = timestamp
                state.palm_anchor_center = observation.center.copy()
                state.rotation_anchor_time = timestamp
                state.rotation_anchor_angle = float(observation.palm_angle)
                state.rotation_anchor_center = observation.center.copy()
            state.palm_last_active = timestamp
            state.rotation_last_active = timestamp
        else:
            if state.palm_last_active is not None and timestamp - state.palm_last_active > 0.28:
                state.palm_anchor_time = None
                state.palm_anchor_center = None
                state.palm_last_active = None
            if state.rotation_last_active is not None and timestamp - state.rotation_last_active > 0.35:
                state.rotation_anchor_time = None
                state.rotation_anchor_angle = None
                state.rotation_anchor_center = None
                state.rotation_last_active = None
        if self._is_pinch_pose(observation):
            if state.pinch_anchor_time is None or (state.pinch_last_active is not None and timestamp - state.pinch_last_active > 0.55):
                state.pinch_anchor_time = timestamp
                state.pinch_anchor_ratio = float(observation.pinch_ratio)
                state.pinch_anchor_center = observation.center.copy()
            state.pinch_last_active = timestamp
        elif state.pinch_last_active is not None and timestamp - state.pinch_last_active > 0.35:
            state.pinch_anchor_time = None
            state.pinch_anchor_ratio = None
            state.pinch_anchor_center = None
            state.pinch_last_active = None

    def _build_event(self, gesture_name: str, gesture_type: str, observation: GestureObservation, timestamp: float, confidence: float, details: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return {
            "event": gesture_name,
            "event_cn": get_gesture_label_cn(gesture_name),
            "gesture_type": gesture_type,
            "track_id": observation.track_id,
            "timestamp": round(float(timestamp), 3),
            "confidence": round(float(confidence), 3),
            "bbox": [int(value) for value in observation.bbox],
            "center": [round(float(observation.center[0]), 1), round(float(observation.center[1]), 1)],
            "static_gesture": observation.static_gesture,
            "static_gesture_cn": get_gesture_label_cn(observation.static_gesture),
            "action_hints": ACTION_HINTS.get(gesture_name, []),
            "action_hints_cn": get_action_hint_labels_cn(ACTION_HINTS.get(gesture_name, [])),
            "details": details or {},
        }

    def _emit_event(self, state: _TrackState, gesture_name: str, gesture_type: str, observation: GestureObservation, timestamp: float, confidence: float, cooldown: Optional[float] = None, details: Optional[Dict[str, object]] = None, episode_token: Optional[float] = None) -> Optional[Dict[str, object]]:
        effective_cooldown = self.config.event_cooldown if cooldown is None else cooldown
        last_emit_at = state.emitted_at.get(gesture_name, -1e9)
        if timestamp - last_emit_at < effective_cooldown:
            return None
        if episode_token is not None:
            last_token = state.emitted_tokens.get(gesture_name)
            if last_token is not None and abs(last_token - episode_token) < 1e-6:
                return None
        state.emitted_at[gesture_name] = timestamp
        if episode_token is not None:
            state.emitted_tokens[gesture_name] = episode_token
        return self._build_event(gesture_name, gesture_type, observation, timestamp, confidence, details)

    def _reset_swipe_anchor(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> None:
        state.swipe_anchor_time = timestamp
        state.swipe_anchor_point = observation.landmarks[8].copy()
        state.swipe_last_active = timestamp

    def _reset_palm_anchor(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> None:
        state.palm_anchor_time = timestamp
        state.palm_anchor_center = observation.center.copy()
        state.palm_last_active = timestamp

    def _reset_pinch_anchor(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> None:
        state.pinch_anchor_time = timestamp
        state.pinch_anchor_ratio = float(observation.pinch_ratio)
        state.pinch_anchor_center = observation.center.copy()
        state.pinch_last_active = timestamp

    def _reset_rotation_anchor(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> None:
        state.rotation_anchor_time = timestamp
        state.rotation_anchor_angle = float(observation.palm_angle)
        state.rotation_anchor_center = observation.center.copy()
        state.rotation_last_active = timestamp

    def _detect_swipe(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if state.swipe_anchor_time is None or state.swipe_anchor_point is None:
            return None
        if timestamp - state.swipe_anchor_time > 1.1:
            self._reset_swipe_anchor(state, observation, timestamp)
            return None
        recent_labels = [label for ts, label in state.label_history if timestamp - ts <= 1.2]
        if not self._is_one_pose(observation) and observation.static_gesture != "one" and recent_labels.count("one") < 1:
            return None
        recent_points = _points_from_history(state.index_history, self.config.swipe_time_window, timestamp)
        if len(recent_points) < 4:
            return None
        start_point = state.swipe_anchor_point.copy()
        end_point = observation.landmarks[8].copy()
        delta = end_point - start_point
        path_length = sum(_distance(recent_points[idx - 1], recent_points[idx]) for idx in range(1, len(recent_points)))
        displacement = _distance(start_point, end_point)
        axis_distance = max(abs(float(delta[0])), abs(float(delta[1])))
        threshold = max(20.0, 0.45 * observation.palm_size)
        if displacement < threshold or path_length < threshold * 1.02:
            return None
        if abs(delta[0]) > abs(delta[1]) * 1.22:
            gesture_name = "swipe_left" if delta[0] < 0 else "swipe_right"
            dominant_ratio = abs(float(delta[0])) / max(path_length, 1.0)
        elif abs(delta[1]) > abs(delta[0]) * 1.22:
            gesture_name = "swipe_up" if delta[1] < 0 else "swipe_down"
            dominant_ratio = abs(float(delta[1])) / max(path_length, 1.0)
        else:
            return None
        if dominant_ratio < 0.45 or axis_distance < threshold:
            return None
        distance_ratio = min(1.0, axis_distance / max(threshold * 1.8, 1.0))
        event = self._emit_event(
            state,
            gesture_name,
            "dynamic",
            observation,
            timestamp,
            min(0.96, 0.72 + dominant_ratio * 0.16 + distance_ratio * 0.12),
            details={"delta": [round(float(delta[0]), 1), round(float(delta[1]), 1)], "path_length": round(float(path_length), 1)},
        )
        if event is not None:
            self._reset_swipe_anchor(state, observation, timestamp)
        return event

    def _detect_palm_swipe(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if state.palm_anchor_time is None or state.palm_anchor_center is None:
            return None
        if timestamp - state.palm_anchor_time > 1.2:
            self._reset_palm_anchor(state, observation, timestamp)
            return None
        recent_labels = [label for ts, label in state.label_history if timestamp - ts <= 1.2]
        if observation.static_gesture != "palm_open" and recent_labels.count("palm_open") < 1:
            return None
        recent_centers = _points_from_history(state.center_history, self.config.swipe_time_window, timestamp)
        if len(recent_centers) < 4:
            return None
        start_center = state.palm_anchor_center.copy()
        end_center = observation.center.copy()
        delta = end_center - start_center
        threshold = max(22.0, 0.45 * observation.palm_size)
        if delta[0] > -threshold or abs(delta[0]) <= abs(delta[1]) * 1.1:
            return None
        event = self._emit_event(
            state,
            "palm_swipe_left",
            "dynamic",
            observation,
            timestamp,
            min(0.94, 0.76 + min(1.0, abs(float(delta[0])) / max(threshold * 2.0, 1.0)) * 0.16),
            details={"delta": [round(float(delta[0]), 1), round(float(delta[1]), 1)]},
        )
        if event is not None:
            self._reset_palm_anchor(state, observation, timestamp)
        return event

    def _detect_zoom(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if state.pinch_anchor_time is None or state.pinch_anchor_ratio is None or state.pinch_anchor_center is None:
            return None
        if timestamp - state.pinch_anchor_time > 1.2:
            self._reset_pinch_anchor(state, observation, timestamp)
            return None
        recent_values = _values_from_history(state.pinch_history, self.config.zoom_time_window, timestamp)
        recent_centers = _points_from_history(state.center_history, self.config.zoom_time_window, timestamp)
        if len(recent_values) < 4 or len(recent_centers) < 4:
            return None
        delta = float(observation.pinch_ratio) - float(state.pinch_anchor_ratio)
        if abs(delta) < 0.1:
            return None
        candidate_pinch_pose = (
            observation.static_gesture in {"pinch", "ok"}
            or observation.raw_static_gesture in {"pinch", "ok"}
            or observation.static_gesture == "one"
            or observation.raw_static_gesture == "one"
            or min(recent_values) < 0.95
            or abs(delta) > 0.18
        )
        if not candidate_pinch_pose:
            return None
        if _distance(state.pinch_anchor_center, observation.center) > 2.2 * observation.palm_size:
            return None
        if _trend_consistency(recent_values) < 0.65:
            return None
        gesture_name = "zoom_in" if delta > 0.0 else "zoom_out"
        event = self._emit_event(
            state,
            gesture_name,
            "dynamic",
            observation,
            timestamp,
            min(0.95, 0.76 + abs(delta) * 0.9),
            details={"pinch_delta": round(float(delta), 3)},
        )
        if event is not None:
            self._reset_pinch_anchor(state, observation, timestamp)
        return event

    def _detect_rotation(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if state.rotation_anchor_time is None or state.rotation_anchor_angle is None or state.rotation_anchor_center is None:
            return None
        if observation.static_gesture not in {"love", "palm_open", "pinch"} and observation.raw_static_gesture not in {"love", "palm_open", "pinch"}:
            return None
        if timestamp - state.rotation_anchor_time > 1.1:
            self._reset_rotation_anchor(state, observation, timestamp)
            return None
        recent_angles = _values_from_history(state.angle_history, self.config.rotation_time_window, timestamp)
        if len(recent_angles) < 4:
            return None
        if _distance(state.rotation_anchor_center, observation.center) > 1.4 * observation.palm_size:
            return None
        delta = _angle_delta(float(state.rotation_anchor_angle), float(observation.palm_angle))
        if abs(delta) < 24.0:
            return None
        gesture_name = "rotate_clockwise" if delta > 0.0 else "rotate_counterclockwise"
        event = self._emit_event(
            state,
            gesture_name,
            "dynamic",
            observation,
            timestamp,
            min(0.94, 0.72 + abs(delta) / 100.0),
            details={"angle_delta": round(float(delta), 1)},
        )
        if event is not None:
            self._reset_rotation_anchor(state, observation, timestamp)
        return event

    def _detect_cross(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if observation.static_gesture != "one":
            return None
        recent_points = _points_from_history(state.index_history, 1.0, timestamp)
        if len(recent_points) < 6:
            return None
        points = _simplify_path(recent_points, max(8.0, 0.2 * observation.palm_size))
        if len(points) < 4:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if (max(xs) - min(xs)) < 0.9 * observation.palm_size or (max(ys) - min(ys)) < 0.9 * observation.palm_size:
            return None
        for idx in range(len(points) - 1):
            for jdx in range(idx + 2, len(points) - 1):
                if jdx == idx + 1:
                    continue
                if _segments_intersect(points[idx], points[idx + 1], points[jdx], points[jdx + 1]):
                    return self._emit_event(state, "cross", "dynamic", observation, timestamp, 0.84, cooldown=1.3)
        return None

    def _detect_open_then_fist(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if observation.static_gesture != "fist":
            return None
        recent_labels = [label for ts, label in state.label_history if timestamp - ts <= 1.0]
        if "palm_open" not in recent_labels:
            return None
        if timestamp - state.label_started_at > 0.5:
            return None
        return self._emit_event(state, "open_then_fist", "dynamic", observation, timestamp, 0.8, cooldown=1.8)

    def _detect_fist_hold(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if observation.static_gesture != "fist" or state.fist_started_at is None:
            return None
        hold_duration = timestamp - state.fist_started_at
        if hold_duration < 2.2:
            return None
        return self._emit_event(
            state,
            "fist_hold",
            "dynamic",
            observation,
            timestamp,
            min(0.95, 0.75 + hold_duration / 4.0),
            cooldown=2.0,
            details={"hold_seconds": round(float(hold_duration), 2)},
            episode_token=state.fist_started_at,
        )

    def _detect_temporal_model_event(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if self.temporal_classifier is None:
            return None
        frames = [landmarks for _, landmarks in list(state.landmark_history)[:: max(1, self.config.temporal_stride)]]
        if len(frames) < max(4, self.config.temporal_window_size // 3):
            return None
        sequence = stack_sequence_features(frames[-self.config.temporal_window_size :])
        prediction = self.temporal_classifier.predict(sequence)
        if prediction is None:
            return None
        gesture_name, confidence, scores = prediction
        if confidence < self.config.temporal_confidence:
            return None
        return self._emit_event(
            state,
            gesture_name,
            "dynamic_model",
            observation,
            timestamp,
            confidence,
            details={"model_scores": {key: round(float(value), 4) for key, value in scores.items()}},
        )

    def _detect_static_event(self, state: _TrackState, observation: GestureObservation, timestamp: float) -> Optional[Dict[str, object]]:
        if observation.static_gesture not in STATIC_EVENT_LABELS:
            return None
        stable_duration = timestamp - state.label_started_at
        if stable_duration < 0.35:
            return None
        return self._emit_event(
            state,
            observation.static_gesture,
            "static",
            observation,
            timestamp,
            observation.static_confidence,
            cooldown=1.2,
            episode_token=state.label_started_at,
        )

    def _cleanup(self, timestamp: float, active_track_ids: Sequence[int]) -> None:
        active_set = set(active_track_ids)
        expired_ids = [
            track_id
            for track_id, state in self.states.items()
            if (track_id not in active_set) and (timestamp - state.last_seen > self.track_ttl)
        ]
        for track_id in expired_ids:
            self.states.pop(track_id, None)
