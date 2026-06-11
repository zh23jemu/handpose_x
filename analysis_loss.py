import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wing_loss_value(error: np.ndarray, w: float, epsilon: float) -> np.ndarray:
    """计算 Wing Loss 曲线值。

    Wing Loss 在小误差区间使用对数曲线放大梯度，在大误差区间切换为线性段，
    因此比 L2 更不容易被大误差样本主导，又比 L1 更重视关键点微小偏差。
    """
    abs_error = np.abs(error)
    c_value = w * (1.0 - math.log(1.0 + w / epsilon))
    return np.where(abs_error < w, w * np.log(1.0 + abs_error / epsilon), abs_error - c_value)


def wing_loss_gradient(error: np.ndarray, w: float, epsilon: float) -> np.ndarray:
    """计算 Wing Loss 对误差的一阶梯度，用于解释小误差区域的梯度行为。"""
    abs_error = np.abs(error)
    sign = np.sign(error)
    small_region = w / (epsilon + abs_error)
    return np.where(abs_error < w, sign * small_region, sign)


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化分析 Wing Loss 与 L1/L2 的损失和梯度行为")
    parser.add_argument("--output-dir", default="analysis_outputs/loss", help="分析图和 CSV 的输出目录")
    parser.add_argument("--w", type=float, default=0.06, help="Wing Loss 分段阈值 w")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Wing Loss 平滑参数 epsilon")
    parser.add_argument("--max-error", type=float, default=0.30, help="横轴最大归一化误差")
    parser.add_argument("--points", type=int, default=1201, help="曲线采样点数量")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = np.linspace(-args.max_error, args.max_error, args.points)
    wing_values = wing_loss_value(errors, args.w, args.epsilon)
    l1_values = np.abs(errors)
    l2_values = errors ** 2
    wing_grad = wing_loss_gradient(errors, args.w, args.epsilon)
    l1_grad = np.sign(errors)
    l2_grad = 2.0 * errors

    csv_path = output_dir / "loss_curve_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["error", "wing_loss", "l1_loss", "l2_loss", "wing_grad", "l1_grad", "l2_grad"])
        for row in zip(errors, wing_values, l1_values, l2_values, wing_grad, l1_grad, l2_grad):
            writer.writerow([round(float(value), 8) for value in row])

    plt.figure(figsize=(9, 5))
    plt.plot(errors, wing_values, label="Wing Loss", linewidth=2.2)
    plt.plot(errors, l1_values, label="L1", linestyle="--")
    plt.plot(errors, l2_values, label="L2", linestyle=":")
    plt.axvline(-args.w, color="gray", linewidth=0.8)
    plt.axvline(args.w, color="gray", linewidth=0.8)
    plt.title("Loss Curve Comparison")
    plt.xlabel("Normalized keypoint error")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(errors, wing_grad, label="Wing gradient", linewidth=2.2)
    plt.plot(errors, l1_grad, label="L1 gradient", linestyle="--")
    plt.plot(errors, l2_grad, label="L2 gradient", linestyle=":")
    plt.axvline(-args.w, color="gray", linewidth=0.8)
    plt.axvline(args.w, color="gray", linewidth=0.8)
    plt.title("Gradient Behavior Comparison")
    plt.xlabel("Normalized keypoint error")
    plt.ylabel("Gradient")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_behavior_comparison.png", dpi=180)
    plt.close()

    summary_path = output_dir / "loss_analysis_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Wing Loss 分析摘要",
                "",
                "- 分段阈值 `w={:.4f}`，平滑参数 `epsilon={:.4f}`。".format(args.w, args.epsilon),
                "- 当误差绝对值小于 `w` 时，Wing Loss 使用对数段，小误差梯度更大，适合强调关键点精细定位。",
                "- 当误差绝对值不小于 `w` 时，Wing Loss 退化为线性段，降低异常大误差对整体训练的主导作用。",
                "- 输出文件：`loss_curve_comparison.png`、`gradient_behavior_comparison.png`、`loss_curve_values.csv`。",
            ]
        ),
        encoding="utf-8",
    )
    print("Wing Loss 分析已输出到：{}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
