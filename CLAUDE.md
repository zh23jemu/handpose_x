# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HandPose X 是一个手部姿态检测与动态手势识别系统。核心功能分两层：
1. **手部关键点检测**：检测手部21个关键点（42维输出），支持多种CNN骨干网络
2. **手势识别运行时**：基于关键点的静态/动态手势识别，模块化四层管道架构

## 环境搭建

```bash
# 安装依赖（推荐 Python 3.11；mediapipe==0.10.11 支持 Python 3.11）
.venv/bin/pip install -r requirements-gesture.txt
```

依赖说明：`mediapipe==0.10.11` 版本锁定，不可随意升级；当前项目推荐 Python 3.11，PyTorch 需与 CUDA 版本匹配。

## 常用命令

### 实时视频演示
```bash
# 摄像头实时演示（推荐 MediaPipe 后端）
.venv/bin/python video_demo.py --source 0 --display --backend mediapipe

# 视频文件
.venv/bin/python video_demo.py --source path/to/video.mp4 --display

# 单张图片（不显示窗口，打印事件）
.venv/bin/python video_demo.py --source image.jpg --no-display --print-events

# 覆盖配置参数
.venv/bin/python video_demo.py --source 0 --display \
  --min-gesture-confidence 0.65 --temporal-model-path gesture_checkpoints/best.pt
```

### 训练手部关键点模型
```bash
# 修改 train.py 内部参数后运行（模型类型、数据集路径等在脚本顶部配置）
.venv/bin/python train.py
```

### 训练动态手势时序模型
```bash
.venv/bin/python train_dynamic_gesture.py \
  --train-manifest datasets/dynamic_gesture/train.jsonl \
  --val-manifest datasets/dynamic_gesture/val.jsonl \
  --output-dir gesture_checkpoints \
  --epochs 40 --hidden-dim 192 --num-layers 3
```

### 推理与模型转换
```bash
# 静态手势推理
.venv/bin/python inference.py

# 导出 ONNX 模型（先配置 model2onnx.py 内的路径）
.venv/bin/python model2onnx.py
.venv/bin/python onnx_inference.py
```

## 架构：gesture_runtime 四层管道

核心模块 `gesture_runtime/` 实现了从图像帧到手势事件的完整流水线：

```
图像帧
  → Backend（手部检测）     → HandDetection (bbox + 21关键点)
  → Tracker（多帧跟踪）     → TrackedHand (track_id + velocity)
  → Recognizer（手势识别）  → GestureObservation + GestureEvent
  → Visualization（渲染）   → 显示帧 + JSON事件输出
```

**关键文件：**
- `gesture_runtime/pipeline.py`：组装四层，入口是 `GesturePipeline.process_frame()`
- `gesture_runtime/backends/factory.py`：根据配置选择 MediaPipe 或 Legacy 后端
- `gesture_runtime/hybrid_recognizer.py`：静态手势（角度约束）+ 动态手势状态机，最复杂的模块
- `gesture_runtime/temporal_model.py`：Transformer 时序分类器，输入32帧特征，输出11类动态手势
- `gesture_runtime/tracker.py`：基于 IOU + 中心距离的轻量跟踪，带地标平滑

## 配置系统

所有运行参数通过 `configs/gesture_runtime.yaml` 管理，对应 `gesture_runtime/config.py` 中的四个嵌套 dataclass：

```
RuntimeConfig
  ├── BackendConfig     # 后端选择、模型路径、MediaPipe 置信度
  ├── TrackerConfig     # 跟踪存活时间、平滑系数
  ├── RecognizerConfig  # 手势置信度阈值、事件冷却时间、时序模型路径
  └── DisplayConfig     # 镜像、显示开关
```

命令行参数会覆盖 YAML 配置中对应字段（`video_demo.py:44-70`）。

## 数据格式

**关键点训练数据**：每张图片对应同名 JSON 文件
```json
{"0": {"x": 100.5, "y": 120.5}, ..., "20": {"x": 150.0, "y": 200.0}}
```
关键点索引 0=手腕，1-4=大拇指，5-8=食指，以此类推，共21个点。

**动态手势训练数据**（JSONL，每行一个样本）：
```json
{"label": "swipe_left", "frames": [{"frame_index": 0, "landmarks": [[x, y], ...]}, ...]}
```

**推理输入预处理**（`hand_data_iter/datasets.py` 中的关键逻辑）：
```python
w_ = max(abs(x_max-x_min), abs(y_max-y_min)) * 1.1
x1, y1 = int(x_mid - w_/2), int(y_mid - w_/2)  # 正方形裁剪后送入模型
```

## 模型输出说明

- 关键点模型输出：42维向量（21点 × x,y），范围归一化到 [0, img_size]
- 时序模型输入特征（71维）：42维坐标 + 10维手指角度 + 5维指尖距离 + 2维捏合比/掌角
- 时序模型输出：11类动态手势概率（`gesture_runtime/labels.py` 中的 `TEMPORAL_GESTURE_CLASSES`）

## 手势标签

静态手势（`STATIC_EVENT_LABELS`）：`one`, `ok`, `fist`, `pinch`, `palm_open`, `love`

动态手势（`TEMPORAL_GESTURE_CLASSES`）：`swipe_left/right/up/down`, `zoom_in/out`, `rotate_clockwise/counterclockwise`, `cross`, `palm_swipe_left`, `open_then_fist`

## 预训练权重

`Weight/resnet_101-size-256-best_model2.pth`（512 MB）是 Legacy 后端的默认权重。配置文件中的 `model_path` 需使用绝对路径或相对于运行目录的路径。

## 后端选择建议

- **开发/演示**：`--backend mediapipe`，无需加载大权重文件，启动快
- **离线/自定义**：`--backend legacy`，需配置 `model_path` 指向 `.pth` 或 `.onnx` 文件
- **`--backend auto`**：优先尝试 MediaPipe，失败则回退 Legacy
