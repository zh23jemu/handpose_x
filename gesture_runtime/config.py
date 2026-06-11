import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, TypeVar

import yaml


@dataclass
class BackendConfig:
    name: str = "auto"
    device: str = "auto"
    model_path: str = "handpose_x_pre_text/ReXNetV1-size-256-wingloss102-0.122.pth"
    img_size: int = 256
    min_hand_area: int = 2600
    max_hands: int = 1
    redetect_interval: int = 10
    track_bbox_scale: float = 1.45
    track_min_side: int = 120
    track_miss_limit: int = 6
    min_landmark_span_ratio: float = 0.24
    max_bbox_area_ratio: float = 0.42
    max_bbox_width_ratio: float = 0.78
    max_bbox_height_ratio: float = 0.90
    mediapipe_model_complexity: int = 1
    mediapipe_min_detection_confidence: float = 0.60
    mediapipe_min_tracking_confidence: float = 0.55


@dataclass
class TrackerConfig:
    track_ttl: float = 2.5
    min_match_distance: float = 70.0
    match_distance_ratio: float = 3.2
    min_match_iou: float = 0.05
    landmark_smoothing: float = 0.35
    bbox_padding_ratio: float = 0.18


@dataclass
class RecognizerConfig:
    event_cooldown: float = 0.90
    min_gesture_confidence: float = 0.72
    min_event_confidence: float = 0.68
    swipe_time_window: float = 0.75
    zoom_time_window: float = 0.85
    rotation_time_window: float = 0.70
    temporal_model_path: str = ""
    temporal_window_size: int = 32
    temporal_stride: int = 2
    temporal_confidence: float = 0.82


@dataclass
class DisplayConfig:
    font_path: str = ""
    mirror: bool = True
    display: bool = True
    print_events: bool = True
    event_log: str = ""


@dataclass
class RuntimeConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DataclassT = TypeVar("_DataclassT")


def _merge_dataclass(instance: _DataclassT, updates: Dict[str, Any]) -> _DataclassT:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_runtime_config(config_path: str = "") -> RuntimeConfig:
    config = RuntimeConfig()
    if not config_path:
        return config
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError("runtime config not found: {}".format(config_path))
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    return _merge_dataclass(config, payload)
