from typing import Dict, List, Sequence


ACTION_HINTS: Dict[str, List[str]] = {
    "swipe_left": ["next_page"],
    "swipe_right": ["previous_page"],
    "swipe_up": ["scroll_down", "volume_up"],
    "swipe_down": ["scroll_up", "volume_down"],
    "zoom_in": ["zoom_in"],
    "zoom_out": ["zoom_out"],
    "rotate_clockwise": ["rotate_right"],
    "rotate_counterclockwise": ["rotate_left"],
    "one": ["mute"],
    "ok": ["accept"],
    "cross": ["reject"],
    "palm_swipe_left": ["close_task"],
    "open_then_fist": ["maximize"],
    "fist_hold": ["turn_off_pc"],
}


GESTURE_LABELS_CN: Dict[str, str] = {
    "unknown": "未识别",
    "fist": "握拳",
    "fist_hold": "握拳停留",
    "palm_open": "张开手掌",
    "one": "食指竖起",
    "ok": "OK手势",
    "pinch": "捏合",
    "love": "爱心手势",
    "swipe_left": "食指左滑",
    "swipe_right": "食指右滑",
    "swipe_up": "食指上滑",
    "swipe_down": "食指下滑",
    "zoom_in": "放大手势",
    "zoom_out": "缩小手势",
    "rotate_clockwise": "顺时针旋转",
    "rotate_counterclockwise": "逆时针旋转",
    "cross": "画叉手势",
    "palm_swipe_left": "手掌左挥",
    "open_then_fist": "张开后握拳",
}


ACTION_HINT_LABELS_CN: Dict[str, str] = {
    "zoom_in": "放大",
    "zoom_out": "缩小",
    "next_page": "下一页",
    "previous_page": "上一页",
    "scroll_down": "向下滚动",
    "volume_up": "音量增大",
    "scroll_up": "向上滚动",
    "volume_down": "音量减小",
    "rotate_right": "向右旋转",
    "rotate_left": "向左旋转",
    "mute": "静音",
    "accept": "确认",
    "reject": "拒绝",
    "close_task": "关闭任务",
    "maximize": "最大化",
    "turn_off_pc": "关机",
}


STATIC_EVENT_LABELS = {"one", "ok", "fist", "pinch", "palm_open", "love"}
NON_THUMB_FINGERS = ("index", "middle", "ring", "pinky")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

TEMPORAL_GESTURE_CLASSES = [
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "zoom_in",
    "zoom_out",
    "rotate_clockwise",
    "rotate_counterclockwise",
    "cross",
    "palm_swipe_left",
    "open_then_fist",
    "fist_hold",
]


def get_gesture_label_cn(gesture_name: str) -> str:
    return GESTURE_LABELS_CN.get(gesture_name, gesture_name)


def get_action_hint_labels_cn(action_hints: Sequence[str]) -> List[str]:
    return [ACTION_HINT_LABELS_CN.get(action_hint, action_hint) for action_hint in action_hints]
