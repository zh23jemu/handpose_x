import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.dataset_reader import collect_annotations


KEYPOINT_NAMES = [
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_tip",
    "index_1",
    "index_2",
    "index_3",
    "index_tip",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_tip",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_tip",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_tip",
]


def preprocess_sample(sample: Dict[str, object], img_size: int, crop_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """按推理习惯做确定性方形裁剪，并返回模型输入和归一化标签。

    训练数据加载器内部会包含随机旋转、随机缩放、镜像和颜色扰动；评估时如果继续使用
    随机增强，会导致同一模型多次评估结果不稳定。因此这里固定使用关键点外接框中心裁剪：
    1. 根据 21 点求最小外接框；
    2. 以较长边乘 `crop_scale` 得到正方形裁剪区域；
    3. 将关键点转换为裁剪图内归一化坐标；
    4. 将图像缩放到模型输入尺寸，并使用项目原有 `(img - 128) / 256` 归一化方式。
    """
    import cv2

    image = cv2.imread(str(sample["image_path"]))
    if image is None:
        raise RuntimeError("无法读取图像：{}".format(sample["image_path"]))

    landmarks = np.asarray(sample["landmarks"], dtype=np.float32)
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)
    side = max(float(x_max - x_min), float(y_max - y_min), 1.0) * crop_scale
    center_x = float(x_min + x_max) / 2.0
    center_y = float(y_min + y_max) / 2.0

    x1 = int(max(center_x - side / 2.0, 0))
    y1 = int(max(center_y - side / 2.0, 0))
    x2 = int(min(center_x + side / 2.0, image.shape[1] - 1))
    y2 = int(min(center_y + side / 2.0, image.shape[0] - 1))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("无效裁剪框：{}".format(sample["image_path"]))

    crop = image[y1:y2, x1:x2, :]
    label = landmarks.copy()
    label[:, 0] = (label[:, 0] - float(x1)) / float(max(x2 - x1, 1))
    label[:, 1] = (label[:, 1] - float(y1)) / float(max(y2 - y1, 1))
    label = np.clip(label, 0.0, 1.0)

    resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    tensor = resized.astype(np.float32)
    tensor = (tensor - 128.0) / 256.0
    tensor = tensor.transpose(2, 0, 1)
    return tensor, label.astype(np.float32)


def plot_per_keypoint_error(per_keypoint_nme: np.ndarray, output_path: Path) -> None:
    """绘制每个关键点的平均归一化误差，用于定位指尖等薄弱部位。"""
    plt.figure(figsize=(12, 5))
    colors = ["#EF4444" if "tip" in name else "#2563EB" for name in KEYPOINT_NAMES]
    plt.bar(np.arange(21), per_keypoint_nme, color=colors)
    plt.xticks(np.arange(21), KEYPOINT_NAMES, rotation=55, ha="right")
    plt.ylabel("NME")
    plt.title("Per-keypoint Mean Normalized Error")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pck_curve(thresholds: np.ndarray, pck_values: np.ndarray, output_path: Path) -> None:
    """绘制 PCK 曲线，展示不同归一化阈值下的关键点命中率。"""
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, pck_values, marker="o", linewidth=2)
    plt.axvline(0.2, color="#EF4444", linestyle="--", label="PCK@0.2")
    plt.xlabel("normalized distance threshold")
    plt.ylabel("PCK")
    plt.ylim(0, 1.02)
    plt.title("PCK Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="评估手部关键点模型的 NME、PCK 和整体识别准确率")
    parser.add_argument("--dataset-dir", required=True, help="评估数据集目录，格式与 handpose_x 原始标注一致")
    parser.add_argument("--model-path", required=True, help="待评估的 PyTorch 权重文件")
    parser.add_argument("--model", default="ReXNetV1", help="模型名称，例如 ReXNetV1/resnet_50/mobilenetv2")
    parser.add_argument("--output-dir", default="analysis_outputs/keypoint_accuracy", help="评估结果输出目录")
    parser.add_argument("--img-size", type=int, default=256, help="模型输入尺寸")
    parser.add_argument("--crop-scale", type=float, default=1.10, help="关键点外接框裁剪放大比例")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="评估设备")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批大小")
    parser.add_argument("--max-samples", type=int, default=0, help="最多评估多少个手部样本，0 表示全部")
    args = parser.parse_args()

    import torch

    from analysis_tools.model_factory import build_keypoint_model, load_checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_keypoint_model(args.model, num_classes=42, img_size=args.img_size)
    load_checkpoint(model, args.model_path, device)
    model.to(device).eval()

    samples = collect_annotations(args.dataset_dir, max_samples=args.max_samples)
    all_errors: List[np.ndarray] = []
    batch_images: List[np.ndarray] = []
    batch_labels: List[np.ndarray] = []

    def flush_batch() -> None:
        """执行一个批次推理并累积 21 点欧氏误差。"""
        if not batch_images:
            return
        images = torch.from_numpy(np.stack(batch_images, axis=0)).to(device).float()
        labels = np.stack(batch_labels, axis=0)
        with torch.no_grad():
            predictions = model(images).detach().cpu().numpy().reshape(-1, 21, 2)
        predictions = np.clip(predictions, 0.0, 1.0)
        errors = np.linalg.norm(predictions - labels, axis=2)
        all_errors.append(errors)
        batch_images.clear()
        batch_labels.clear()

    for sample in samples:
        image_tensor, label = preprocess_sample(sample, args.img_size, args.crop_scale)
        batch_images.append(image_tensor)
        batch_labels.append(label)
        if len(batch_images) >= args.batch_size:
            flush_batch()
    flush_batch()

    errors = np.concatenate(all_errors, axis=0)
    per_keypoint_nme = errors.mean(axis=0)
    overall_nme = float(errors.mean())
    thresholds = np.linspace(0.01, 0.30, 30)
    pck_values = np.asarray([(errors <= threshold).mean() for threshold in thresholds], dtype=np.float32)
    pck_at_02 = float((errors <= 0.2).mean())

    with (output_dir / "per_keypoint_nme.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["keypoint", "nme"])
        for name, value in zip(KEYPOINT_NAMES, per_keypoint_nme):
            writer.writerow([name, round(float(value), 6)])

    with (output_dir / "pck_curve.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["threshold", "pck"])
        for threshold, value in zip(thresholds, pck_values):
            writer.writerow([round(float(threshold), 4), round(float(value), 6)])

    metrics = {
        "samples": int(errors.shape[0]),
        "overall_nme": overall_nme,
        "pck_at_0_2": pck_at_02,
        "overall_accuracy": pck_at_02,
        "device": str(device),
        "model": args.model,
        "model_path": args.model_path,
    }
    (output_dir / "keypoint_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_per_keypoint_error(per_keypoint_nme, output_dir / "per_keypoint_nme.png")
    plot_pck_curve(thresholds, pck_values, output_dir / "pck_curve.png")

    summary = [
        "# 关键点预测精度分析摘要",
        "",
        "- 有效评估样本数：{}".format(errors.shape[0]),
        "- 整体 NME：{:.6f}".format(overall_nme),
        "- PCK@0.2：{:.4f}".format(pck_at_02),
        "- 整体识别准确率按 PCK@0.2 口径记录：{:.4f}".format(pck_at_02),
        "- 指尖类关键点在柱状图中使用红色标记，便于快速观察是否为主要误差来源。",
    ]
    (output_dir / "keypoint_accuracy_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("关键点精度评估已输出到：{}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
