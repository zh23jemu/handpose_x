import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.dataset_reader import collect_annotations, landmarks_to_normalized_xy


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


def save_keypoint_summary(normalized_xy: np.ndarray, output_path: Path) -> None:
    """保存每个关键点的均值、标准差和分布范围，便于判断数据覆盖是否均衡。"""
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["keypoint", "mean_x", "mean_y", "std_x", "std_y", "min_x", "max_x", "min_y", "max_y"])
        for index, name in enumerate(KEYPOINT_NAMES):
            xy = normalized_xy[:, index, :]
            writer.writerow(
                [
                    name,
                    round(float(xy[:, 0].mean()), 6),
                    round(float(xy[:, 1].mean()), 6),
                    round(float(xy[:, 0].std()), 6),
                    round(float(xy[:, 1].std()), 6),
                    round(float(xy[:, 0].min()), 6),
                    round(float(xy[:, 0].max()), 6),
                    round(float(xy[:, 1].min()), 6),
                    round(float(xy[:, 1].max()), 6),
                ]
            )


def plot_keypoint_scatter(normalized_xy: np.ndarray, output_path: Path, title: str) -> None:
    """绘制 21 个关键点在图像归一化平面中的分布。

    图中的坐标已归一化到 `[0, 1]`，可直观看出样本是否集中在某个区域、
    是否存在左右/上下覆盖不足，以及指尖等关键部位是否有足够多样性。
    """
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 21))
    for index, name in enumerate(KEYPOINT_NAMES):
        xy = normalized_xy[:, index, :]
        plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.28, color=colors[index], label=name)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.xlabel("normalized x")
    plt.ylabel("normalized y")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_keypoint_heatmap(normalized_xy: np.ndarray, output_path: Path) -> None:
    """绘制所有关键点合并后的二维热力图，用于观察整体空间覆盖。"""
    points = normalized_xy.reshape(-1, 2)
    plt.figure(figsize=(7, 6))
    plt.hist2d(points[:, 0], points[:, 1], bins=48, range=[[0, 1], [0, 1]], cmap="viridis")
    plt.gca().invert_yaxis()
    plt.colorbar(label="keypoint count")
    plt.xlabel("normalized x")
    plt.ylabel("normalized y")
    plt.title("All Keypoint Spatial Density")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_bbox_statistics(samples: List[dict], output_path: Path) -> None:
    """绘制 bbox 宽高、面积和宽高比统计，辅助判断手部尺度多样性。"""
    widths = []
    heights = []
    areas = []
    ratios = []
    for sample in samples:
        bbox = np.asarray(sample["bbox"], dtype=np.float32)
        width = max(float(bbox[2] - bbox[0]), 1.0)
        height = max(float(bbox[3] - bbox[1]), 1.0)
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        ratios.append(width / height)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(widths, bins=40, color="#3B82F6", alpha=0.85)
    axes[0, 0].set_title("bbox width")
    axes[0, 1].hist(heights, bins=40, color="#10B981", alpha=0.85)
    axes[0, 1].set_title("bbox height")
    axes[1, 0].hist(areas, bins=40, color="#F59E0B", alpha=0.85)
    axes[1, 0].set_title("bbox area")
    axes[1, 1].hist(ratios, bins=40, color="#EF4444", alpha=0.85)
    axes[1, 1].set_title("bbox width / height")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_before_after_compare(before_xy: np.ndarray, after_xy: np.ndarray, output_path: Path) -> None:
    """对比增强前后关键点分布，判断增强是否扩大了空间覆盖。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, xy, title in (
        (axes[0], before_xy.reshape(-1, 2), "before augmentation"),
        (axes[1], after_xy.reshape(-1, 2), "after augmentation"),
    ):
        axis.hist2d(xy[:, 0], xy[:, 1], bins=48, range=[[0, 1], [0, 1]], cmap="viridis")
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("normalized x")
        axis.set_ylabel("normalized y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="分析手部关键点数据集坐标分布与样本质量")
    parser.add_argument("--dataset-dir", required=True, help="原始 handpose_x 数据集目录")
    parser.add_argument("--augmented-dataset-dir", default="", help="可选：增强后数据目录，用于前后对比")
    parser.add_argument("--output-dir", default="analysis_outputs/dataset_quality", help="分析结果输出目录")
    parser.add_argument("--max-samples", type=int, default=0, help="最多分析多少个手部样本，0 表示全部")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_annotations(args.dataset_dir, max_samples=args.max_samples)
    normalized_xy = landmarks_to_normalized_xy(samples)
    save_keypoint_summary(normalized_xy, output_dir / "keypoint_distribution_summary.csv")
    plot_keypoint_scatter(normalized_xy, output_dir / "keypoint_scatter.png", "Keypoint Coordinate Distribution")
    plot_keypoint_heatmap(normalized_xy, output_dir / "keypoint_density_heatmap.png")
    plot_bbox_statistics(samples, output_dir / "bbox_statistics.png")

    summary_lines = [
        "# 数据集质量分析摘要",
        "",
        "- 有效手部样本数：{}".format(len(samples)),
        "- 输出 `keypoint_scatter.png` 用于查看 21 点位置规律和空间多样性。",
        "- 输出 `keypoint_density_heatmap.png` 用于查看整体关键点密度。",
        "- 输出 `bbox_statistics.png` 用于查看手部尺度、面积和宽高比覆盖。",
        "- 输出 `keypoint_distribution_summary.csv` 保存每个关键点的均值、方差和范围。",
    ]

    if args.augmented_dataset_dir:
        augmented_samples = collect_annotations(args.augmented_dataset_dir, max_samples=args.max_samples)
        augmented_xy = landmarks_to_normalized_xy(augmented_samples)
        plot_before_after_compare(normalized_xy, augmented_xy, output_dir / "augmentation_before_after_heatmap.png")
        summary_lines.append("- 已输出 `augmentation_before_after_heatmap.png`，用于对比增强前后坐标覆盖。")

    (output_dir / "dataset_quality_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("数据集质量分析已输出到：{}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
