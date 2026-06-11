import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gesture_runtime.labels import get_action_hint_labels_cn, get_gesture_label_cn
from gesture_runtime.types import GestureObservation
from hand_data_iter.datasets import draw_bd_handpose


FONT_CACHE = {}


def _candidate_font_paths(font_path: str = "") -> List[str]:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/mnt/c/Windows/Fonts/msyh.ttc",
            "/mnt/c/Windows/Fonts/simhei.ttf",
            "/mnt/c/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
    )
    return candidates


def _get_font(font_size: int, font_path: str = "") -> Optional[ImageFont.FreeTypeFont]:
    cache_key = (font_size, font_path)
    if cache_key in FONT_CACHE:
        return FONT_CACHE[cache_key]
    for candidate in _candidate_font_paths(font_path):
        if not candidate or not Path(candidate).exists():
            continue
        try:
            font = ImageFont.truetype(candidate, font_size)
            FONT_CACHE[cache_key] = font
            return font
        except OSError:
            continue
    FONT_CACHE[cache_key] = None
    return None


def draw_texts(frame: np.ndarray, text_items: Sequence[Tuple[str, Tuple[int, int], Tuple[int, int, int], int]], font_path: str = "") -> None:
    if not text_items:
        return
    fallback_items = []
    rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(rgb_frame)
    for text, position, color, font_size in text_items:
        font = _get_font(font_size, font_path)
        if font is None:
            fallback_items.append((text, position, color, font_size))
            continue
        drawer.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.asarray(rgb_frame), cv2.COLOR_RGB2BGR)
    for text, position, color, font_size in fallback_items:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, font_size / 34.0),
            color,
            2 if font_size >= 20 else 1,
            cv2.LINE_AA,
        )


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    hand_points = {
        str(index): {"x": float(point[0]), "y": float(point[1])}
        for index, point in enumerate(landmarks)
    }
    draw_bd_handpose(frame, hand_points, 0, 0)
    for point in landmarks:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 50, 60), -1)
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, (255, 150, 180), -1)


def draw_observation_overlay(frame: np.ndarray, observation: GestureObservation, font_path: str = "") -> None:
    bbox = observation.bbox
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 220, 0), 2)
    title = "ID {} {} {:.2f}".format(observation.track_id, get_gesture_label_cn(observation.static_gesture), observation.static_confidence)
    hint = " / ".join(get_action_hint_labels_cn(observation.action_hints[:2]))
    handedness = "左手" if observation.handedness == "left" else "右手" if observation.handedness == "right" else "未知手"
    text_items = [(title, (bbox[0], max(4, bbox[1] - 48)), (0, 220, 0), 24), (handedness, (bbox[0], max(18, bbox[1] - 24)), (255, 210, 70), 18)]
    if hint:
        text_items.append((hint, (bbox[0], max(30, bbox[1] - 2)), (40, 220, 240), 18))
    draw_texts(frame, text_items, font_path=font_path)


def draw_event_panel(frame: np.ndarray, recent_events: Sequence[str], font_path: str = "") -> None:
    if not recent_events:
        return
    panel_width = min(frame.shape[1] - 20, 460)
    panel_height = 30 + len(recent_events) * 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
    text_items = [("手势事件", (24, 20), (255, 255, 255), 30)]
    for index, text in enumerate(recent_events, start=1):
        text_items.append((text, (24, 24 + index * 22), (210, 240, 255), 22))
    draw_texts(frame, text_items, font_path=font_path)


class EventSink:
    def __init__(self, print_events: bool, event_log: str):
        self.print_events = print_events
        self.file = open(event_log, "a", encoding="utf-8") if event_log else None

    def emit(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        if self.print_events:
            print(payload, flush=True)
        if self.file is not None:
            self.file.write(payload + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class RecentEventBuffer:
    def __init__(self, maxlen: int = 6):
        self.items: Deque[str] = deque(maxlen=maxlen)

    def append(self, event: dict) -> None:
        action_text = " / ".join(event.get("action_hints_cn", [])) or "无提示"
        self.items.append("{} [{}]".format(event.get("event_cn", event.get("event", "unknown")), action_text))

    def as_list(self) -> List[str]:
        return list(self.items)
