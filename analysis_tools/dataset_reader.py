import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_handpose_annotations(dataset_dir: str) -> Iterator[Dict[str, object]]:
    """遍历原始 handpose_x 标注，输出统一后的样本记录。

    支持当前项目数据加载器使用的格式：
    `image.jpg` 对应同名 `image.json`，JSON 顶层包含 `info` 列表，每个元素包含：
    - `bbox`: 手部框 `[x1, y1, x2, y2]`
    - `pts`: 21 个关键点，键为 `"0"` 到 `"20"`，坐标相对 bbox 左上角

    输出记录中的 `landmarks` 会被转换为原图坐标，便于做坐标分布、NME/PCK 等分析。
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError("数据集目录不存在：{}".format(dataset_dir))
    import cv2

    for image_path in sorted(root.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = image_path.with_suffix(".json")
        if not label_path.exists():
            continue
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        hand_items = payload.get("info", [])
        if not isinstance(hand_items, list):
            continue

        image = cv2.imread(str(image_path))
        image_shape = image.shape[:2] if image is not None else (0, 0)
        for hand_index, item in enumerate(hand_items):
            bbox = np.asarray(item.get("bbox", []), dtype=np.float32)
            pts = item.get("pts", {})
            if bbox.shape[0] != 4 or not isinstance(pts, dict):
                continue
            landmarks: List[Tuple[float, float]] = []
            for keypoint_index in range(21):
                point = pts.get(str(keypoint_index), {})
                x_value = float(point.get("x", 0.0)) + float(bbox[0])
                y_value = float(point.get("y", 0.0)) + float(bbox[1])
                landmarks.append((x_value, y_value))

            yield {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "hand_index": hand_index,
                "bbox": bbox,
                "landmarks": np.asarray(landmarks, dtype=np.float32),
                "image_shape": image_shape,
            }


def collect_annotations(dataset_dir: str, max_samples: int = 0) -> List[Dict[str, object]]:
    """收集标注样本；`max_samples` 用于快速抽样分析。"""
    samples: List[Dict[str, object]] = []
    for sample in iter_handpose_annotations(dataset_dir):
        samples.append(sample)
        if max_samples > 0 and len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError("没有在目录中找到有效手部关键点标注：{}".format(dataset_dir))
    return samples


def landmarks_to_normalized_xy(samples: Iterable[Dict[str, object]]) -> np.ndarray:
    """将原图坐标转换为 `[0, 1]` 归一化坐标，形状为 `[N, 21, 2]`。"""
    normalized = []
    for sample in samples:
        height, width = sample["image_shape"]
        if height <= 0 or width <= 0:
            continue
        landmarks = np.asarray(sample["landmarks"], dtype=np.float32).copy()
        landmarks[:, 0] /= float(width)
        landmarks[:, 1] /= float(height)
        normalized.append(landmarks)
    if not normalized:
        raise RuntimeError("所有样本都无法读取图像尺寸，不能计算归一化坐标分布。")
    return np.stack(normalized, axis=0)
