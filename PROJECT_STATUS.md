# HandPose X — 项目现状文档

> 整理时间：2026-06-11

---

## 一、项目定位

手部姿态检测与动态手势识别系统，主要能力：

1. **手部21关键点检测**：基于 CNN 骨干网络，输入手部图像，输出 21 个关键点的 (x, y) 坐标（42维向量）
2. **静态手势识别**：基于关键点角度约束规则，无需额外模型
3. **动态手势识别**：基于 Transformer 时序模型，识别滑动、缩放、旋转等连续动作
4. **实时运行时**：四层管道架构，支持摄像头/视频/图片输入

---

## 二、技术栈

| 组件 | 版本/说明 |
|------|---------|
| Python | 3.8（mediapipe 版本约束，不可随意升级） |
| PyTorch | ≥1.5.1 |
| MediaPipe | ==0.10.11（锁定版本） |
| OpenCV | opencv-python |
| 配置格式 | YAML / JSON |

---

## 三、目录结构

```
handpose_x/
├── gesture_runtime/              # 核心运行时（四层管道）
│   ├── pipeline.py               # 主入口 GesturePipeline.process_frame()
│   ├── config.py                 # 配置体系（嵌套 dataclass）
│   ├── types.py                  # 数据结构定义
│   ├── labels.py                 # 手势标签 + 动作提示（中英双语）
│   ├── tracker.py                # 多帧跟踪（IOU + 中心距离）
│   ├── hybrid_recognizer.py      # 手势识别核心（最复杂模块）
│   ├── temporal_model.py         # Transformer 时序分类模型
│   ├── visualization.py          # 渲染 + 事件输出
│   └── backends/
│       ├── base.py               # 后端基类
│       ├── factory.py            # auto 选择逻辑
│       ├── mediapipe_backend.py  # MediaPipe 后端（推荐）
│       └── legacy_backend.py     # Legacy 后端（ReXNetV1）
│
├── models/                       # CNN 骨干网络定义
│   ├── resnet.py                 # ResNet 18/34/50/101
│   ├── mobilenetv2.py
│   ├── shufflenet.py / shufflenetv2.py
│   ├── squeezenet.py
│   └── rexnetv1.py               # 轻量级，Legacy 后端专用
│
├── hand_data_iter/               # 数据加载与增强
│   ├── datasets.py               # LoadImagesAndLabels
│   ├── data_agu.py               # 通用数据增强
│   └── handpose_agu.py           # 手部关键点专用增强
│
├── loss/
│   └── loss.py                   # Wing Loss（w=0.06, ε=0.01）
│
├── configs/
│   ├── gesture_runtime.yaml      # 运行时参数（后端/跟踪/识别/显示）
│   └── dynamic_gesture_train.json # 时序模型训练参数
│
├── Weight/                       # 预训练权重（不入 git，需单独传输）
│   ├── resnet_101-size-256-best_model.pth   （512 MB）
│   └── resnet_101-size-256-best_model2.pth  （512 MB）
│
├── handpose_datasets/            # 训练数据集（不入 git，约 1.2 GB）
│
├── video_demo.py                 # 实时演示入口
├── train.py                      # 关键点模型训练
├── train_dynamic_gesture.py      # 时序手势模型训练
├── inference.py                  # 单图/视频推理
├── model2onnx.py                 # 导出 ONNX
├── onnx_inference.py             # ONNX 推理
├── gesture_recognition.py        # 对外 API 封装
└── simple_gesture_recognizer.py  # 独立简易识别器
```

---

## 四、运行时四层管道

```
输入帧
  ↓
[Backend]      手部检测 → HandDetection（bbox + 21关键点）
  ↓
[Tracker]      多帧跟踪 → TrackedHand（track_id + velocity + 平滑坐标）
  ↓
[Recognizer]   手势识别 → GestureObservation + GestureEvent（JSON）
  ↓
[Visualization] 渲染叠加 + 事件打印
```

---

## 五、支持的手势

**静态手势**（角度约束规则，无需训练）

| 标识 | 说明 |
|------|------|
| `one` | 食指伸直 |
| `ok` | OK 手势 |
| `fist` | 握拳 |
| `pinch` | 捏合 |
| `palm_open` | 手掌张开 |
| `love` | 比心 |

**动态手势**（需 Transformer 时序模型）

`swipe_left/right/up/down` · `zoom_in/out` · `rotate_clockwise/counterclockwise` · `cross` · `palm_swipe_left` · `open_then_fist`

---

## 六、支持的骨干网络

| 模型 | 参数量 | 特点 |
|------|--------|------|
| ResNet 101 | ~44M | 已有预训练权重，精度最高 |
| ResNet 50 | ~25M | 精度与速度平衡 |
| MobileNetV2 | ~3.5M | 轻量级 |
| ReXNetV1 | ~2.9M | Legacy 后端专用 |
| ShuffleNet 系列 | ~2.4M | 移动端友好 |
| SqueezeNet | ~1.2M | 最小体积 |

所有模型输出头统一为 `Linear(n, 42)`（21点 × x,y）。

---

## 七、损失函数

使用 **Wing Loss**（`loss/loss.py`），参数 `w=0.06, ε=0.01`：

```
loss(x) = w·ln(1 + |x|/ε)   当 |x| < w   （对数段，小误差精细梯度）
          |x| - c             当 |x| ≥ w   （线性段，大误差稳定梯度）
```

连接常数 `c = -0.05675`，保证两段在 `|x|=w` 处连续。

---

## 八、配置说明

`configs/gesture_runtime.yaml` 控制所有运行时行为，关键参数：

```yaml
backend:
  name: auto                  # auto / mediapipe / legacy
  model_path: ...             # Legacy 后端权重路径（需绝对路径）
  max_hands: 1

tracker:
  track_ttl: 2.5              # 跟踪保活秒数
  landmark_smoothing: 0.35    # 平滑系数（0=不平滑，1=完全平滑）

recognizer:
  min_gesture_confidence: 0.72
  min_event_confidence: 0.68
  event_cooldown: 0.90        # 同一事件两次触发最短间隔（秒）
  temporal_model_path: ""     # 留空则不启用时序模型
```

---

## 九、常用命令

```bash
# 实时摄像头演示
python video_demo.py --source 0 --display --backend mediapipe

# 视频文件
python video_demo.py --source demo.mp4 --display

# 训练关键点模型（参数在 train.py 顶部修改）
python train.py

# 训练时序手势模型
python train_dynamic_gesture.py \
  --train-manifest datasets/dynamic_gesture/train.jsonl \
  --val-manifest   datasets/dynamic_gesture/val.jsonl \
  --output-dir     gesture_checkpoints

# 导出 ONNX
python model2onnx.py
```

---

## 十、当前已完成 / 待完善

**已完成**
- 关键点检测完整训练流程（多骨干网络、Wing Loss、数据增强）
- gesture_runtime 四层管道（MediaPipe + Legacy 双后端）
- 静态手势识别（角度约束规则）
- 动态手势状态机（基于速度/方向）
- Transformer 时序模型定义与训练脚本
- ONNX 导出与推理
- 实时视频演示

**待完善**
- 时序模型训练数据集尚未建立（train.jsonl/val.jsonl 需自行采集）
- `train.py` 训练参数硬编码在脚本内，未迁移至配置文件
- 静态手势缺乏模型方式识别（目前纯规则，鲁棒性有限）
- 无评估脚本（NME、PCK 等指标需手动编写）

---

## 十一、注意事项

1. **Python 版本**：mediapipe==0.10.11 要求 Python 3.8，升级 Python 版本前需确认 mediapipe 兼容性
2. **权重文件**：`Weight/` 目录未入 git（单文件 512MB），换机器需单独传输
3. **数据集**：`handpose_datasets/` 未入 git（1.2GB），百度网盘链接见 README.md
4. **Legacy 后端路径**：`gesture_runtime.yaml` 中 `model_path` 默认为绝对路径，跨机器需修改
5. **摄像头**：WSL 环境下无法直接访问摄像头，需在原生 Windows Python 下运行
