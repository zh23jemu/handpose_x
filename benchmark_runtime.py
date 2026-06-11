import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def percentile(values: List[float], ratio: float) -> float:
    """计算延迟百分位数，避免额外引入 numpy 以外的统计依赖。"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[index])


def summarize_latency(values: List[float]) -> Dict[str, float]:
    """输出平均、P50、P90、P95 延迟，单位毫秒。"""
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(statistics.mean(values)),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
    }


def load_frames(source: str, max_frames: int) -> List[np.ndarray]:
    """读取图片、视频或摄像头帧，用于重复测速。

    如果输入是单张图片，会在后续 benchmark 中重复使用该图片；如果是视频或摄像头，
    则先缓存有限帧，避免磁盘/摄像头读取速度影响模型与识别逻辑的耗时统计。
    """
    import cv2

    path = Path(source)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    if path.exists() and path.suffix.lower() in image_suffixes:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError("无法读取图片：{}".format(source))
        return [image]

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError("无法打开输入源：{}".format(source))
    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("输入源没有可用帧：{}".format(source))
    return frames


def benchmark_pipeline_backend(
    backend_name: str,
    frames: List[np.ndarray],
    config_path: str,
    repeat: int,
    warmup: int,
    model_path: str,
    device: str,
) -> Optional[Dict[str, object]]:
    """测量 MediaPipe/Legacy 后端的单帧和端到端延迟。

    统计口径：
    - `single_frame_inference_ms`: 后端 `detect()` 耗时，可视为手部检测/关键点单帧推理延迟；
    - `gesture_recognition_ms`: tracker + recognizer 耗时，表示关键点进入规则/时序识别后的处理延迟；
    - `command_response_ms`: 出现手势事件的帧，从开始处理到事件生成的耗时；若没有事件则为空；
    - `end_to_end_ms`: `GesturePipeline.process_frame()` 总耗时，包含后端、跟踪和识别。
    """
    from gesture_runtime.config import load_runtime_config
    from gesture_runtime.pipeline import GesturePipeline

    config = load_runtime_config(config_path)
    config.backend.name = backend_name
    config.display.display = False
    config.display.print_events = False
    if model_path:
        config.backend.model_path = model_path
    if device:
        config.backend.device = device

    try:
        pipeline = GesturePipeline(config)
    except Exception as exc:
        return {"backend": backend_name, "status": "skipped", "reason": str(exc)}

    backend_ms: List[float] = []
    recognition_ms: List[float] = []
    response_ms: List[float] = []
    e2e_ms: List[float] = []
    event_count = 0

    try:
        for index in range(warmup + repeat):
            frame = frames[index % len(frames)].copy()
            timestamp = time.time()

            start = time.perf_counter()
            detections = pipeline.backend.detect(frame)
            after_backend = time.perf_counter()
            tracked_hands = pipeline.tracker.update(frame.shape, detections, timestamp)
            observations, events = pipeline.recognizer.update(tracked_hands, timestamp)
            end = time.perf_counter()
            _ = observations

            if index < warmup:
                continue
            backend_cost = (after_backend - start) * 1000.0
            recognition_cost = (end - after_backend) * 1000.0
            total_cost = (end - start) * 1000.0
            backend_ms.append(backend_cost)
            recognition_ms.append(recognition_cost)
            e2e_ms.append(total_cost)
            if events:
                event_count += len(events)
                response_ms.append(total_cost)
    finally:
        pipeline.close()

    return {
        "backend": backend_name,
        "status": "ok",
        "frames": repeat,
        "events": event_count,
        "single_frame_inference": summarize_latency(backend_ms),
        "gesture_recognition": summarize_latency(recognition_ms),
        "command_response": summarize_latency(response_ms),
        "end_to_end": summarize_latency(e2e_ms),
    }


def benchmark_model_specs(model_names: Iterable[str], img_size: int, repeat: int, warmup: int, device_name: str) -> List[Dict[str, object]]:
    """测量不同关键点模型规格的前向延迟、参数量和估算模型体积。"""
    import torch

    from analysis_tools.model_factory import build_keypoint_model, count_parameters

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    results = []
    sample = torch.randn(1, 3, img_size, img_size, device=device)

    for model_name in model_names:
        model = build_keypoint_model(model_name, num_classes=42, img_size=img_size).to(device).eval()
        latencies: List[float] = []
        with torch.no_grad():
            for index in range(warmup + repeat):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                if index >= warmup:
                    latencies.append((end - start) * 1000.0)
        params = count_parameters(model)
        results.append(
            {
                "model": model_name,
                "device": str(device),
                "params": params,
                "estimated_size_mb": round(params * 4.0 / 1024.0 / 1024.0, 3),
                "single_frame_inference": summarize_latency(latencies),
            }
        )
    return results


def save_backend_csv(results: List[Dict[str, object]], output_path: Path) -> None:
    """保存后端延迟表，便于直接放入报告或 PPT。"""
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "backend",
                "status",
                "single_frame_mean_ms",
                "gesture_recognition_mean_ms",
                "command_response_mean_ms",
                "end_to_end_mean_ms",
                "events",
                "reason",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.get("backend", ""),
                    item.get("status", ""),
                    item.get("single_frame_inference", {}).get("mean_ms", 0.0),
                    item.get("gesture_recognition", {}).get("mean_ms", 0.0),
                    item.get("command_response", {}).get("mean_ms", 0.0),
                    item.get("end_to_end", {}).get("mean_ms", 0.0),
                    item.get("events", 0),
                    item.get("reason", ""),
                ]
            )


def save_model_csv(results: List[Dict[str, object]], output_path: Path) -> None:
    """保存模型规格延迟表。"""
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["model", "device", "params", "estimated_size_mb", "single_frame_mean_ms", "single_frame_p95_ms"])
        for item in results:
            latency = item["single_frame_inference"]
            writer.writerow(
                [
                    item["model"],
                    item["device"],
                    item["params"],
                    item["estimated_size_mb"],
                    latency["mean_ms"],
                    latency["p95_ms"],
                ]
            )


def plot_latency_bar(backend_results: List[Dict[str, object]], output_path: Path) -> None:
    """绘制不同后端的延迟对比柱状图。"""
    ok_results = [item for item in backend_results if item.get("status") == "ok"]
    if not ok_results:
        return
    labels = [item["backend"] for item in ok_results]
    x_axis = np.arange(len(labels))
    single = [item["single_frame_inference"]["mean_ms"] for item in ok_results]
    recognition = [item["gesture_recognition"]["mean_ms"] for item in ok_results]
    e2e = [item["end_to_end"]["mean_ms"] for item in ok_results]

    width = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar(x_axis - width, single, width, label="single frame")
    plt.bar(x_axis, recognition, width, label="gesture recognition")
    plt.bar(x_axis + width, e2e, width, label="end to end")
    plt.xticks(x_axis, labels)
    plt.ylabel("latency (ms)")
    plt.title("Backend Latency Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_model_tradeoff(model_results: List[Dict[str, object]], output_path: Path) -> None:
    """绘制模型体积与单帧延迟权衡图。"""
    if not model_results:
        return
    plt.figure(figsize=(8, 5))
    for item in model_results:
        x_value = item["estimated_size_mb"]
        y_value = item["single_frame_inference"]["mean_ms"]
        plt.scatter(x_value, y_value, s=80)
        plt.text(x_value, y_value, item["model"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("estimated model size (MB)")
    plt.ylabel("single frame latency (ms)")
    plt.title("Model Size vs Latency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 HandPose X 推理性能和运行时响应延迟")
    parser.add_argument("--source", default="image/image_2021-02-04_20-05-49.jpg", help="图片、视频或摄像头编号")
    parser.add_argument("--config", default="configs/gesture_runtime.yaml", help="运行时配置文件")
    parser.add_argument("--backends", default="mediapipe,legacy", help="逗号分隔的后端列表：mediapipe,legacy")
    parser.add_argument("--legacy-model-path", default="", help="Legacy 后端权重路径覆盖")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--model-specs", default="ReXNetV1,mobilenetv2,resnet_50,resnet_101", help="逗号分隔的模型规格列表")
    parser.add_argument("--img-size", type=int, default=256, help="关键点模型输入尺寸")
    parser.add_argument("--repeat", type=int, default=50, help="正式重复测量次数")
    parser.add_argument("--warmup", type=int, default=5, help="预热次数")
    parser.add_argument("--output-dir", default="analysis_outputs/runtime_benchmark", help="测速结果输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.source, max(args.repeat + args.warmup, 1))
    backend_names = [item.strip() for item in args.backends.split(",") if item.strip()]
    backend_results = [
        benchmark_pipeline_backend(
            backend_name=name,
            frames=frames,
            config_path=args.config,
            repeat=args.repeat,
            warmup=args.warmup,
            model_path=args.legacy_model_path,
            device=args.device,
        )
        for name in backend_names
    ]

    model_names = [item.strip() for item in args.model_specs.split(",") if item.strip()]
    model_results = benchmark_model_specs(model_names, args.img_size, args.repeat, args.warmup, args.device)

    payload = {"backend_results": backend_results, "model_results": model_results}
    (output_dir / "runtime_benchmark_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_backend_csv(backend_results, output_dir / "backend_latency_summary.csv")
    save_model_csv(model_results, output_dir / "model_latency_size_summary.csv")
    plot_latency_bar(backend_results, output_dir / "backend_latency_comparison.png")
    plot_model_tradeoff(model_results, output_dir / "model_size_latency_tradeoff.png")

    summary = [
        "# 推理性能分析摘要",
        "",
        "- 重复测量次数：{}，预热次数：{}。".format(args.repeat, args.warmup),
        "- `backend_latency_summary.csv` 包含单帧推理延迟、手势识别延迟、指令响应延迟和系统端到端总延迟。",
        "- `model_latency_size_summary.csv` 包含不同模型规格的参数量、估算体积和单帧前向延迟。",
        "- 如果 `command_response_mean_ms` 为 0，表示测试输入没有触发可统计的手势事件，需要换成包含动作的视频样本。",
    ]
    (output_dir / "runtime_benchmark_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("运行时性能分析已输出到：{}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
