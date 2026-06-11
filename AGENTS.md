# AGENTS.md — HandPose X

## 项目目标

手部姿态检测与动态手势识别系统。包含：
1. 基于 CNN 骨干网络的手部21关键点检测模型训练
2. 模块化四层手势识别运行时（Backend→Tracker→Recognizer→Visualization）

## 技术栈

- Python 3.8（mediapipe==0.10.11 版本约束）
- PyTorch + torchvision（关键点检测 + 时序 Transformer）
- MediaPipe（推荐手部检测后端）
- OpenCV（图像处理 + 视频捕获）
- PyYAML（配置管理）

## 当前架构

```
gesture_runtime/          # 核心运行时（四层管道）
  ├── pipeline.py         # 主入口：GesturePipeline.process_frame()
  ├── backends/           # 后端：mediapipe_backend / legacy_backend
  ├── tracker.py          # IOU+中心距离轻量跟踪
  ├── hybrid_recognizer.py # 静态角度约束 + 动态状态机识别
  ├── temporal_model.py   # Transformer 时序手势分类
  ├── config.py           # 嵌套 dataclass 配置体系
  ├── types.py            # HandDetection / TrackedHand / GestureObservation
  └── labels.py           # 手势标签 + 动作提示

models/                   # CNN 骨干网络（resnet/mobilenet/shufflenet 等）
hand_data_iter/           # 数据加载 + 增强
loss/                     # 损失函数
configs/                  # YAML/JSON 配置文件
Weight/                   # 预训练权重（resnet101, 512MB×2）
video_demo.py             # 演示入口
train.py                  # 关键点模型训练
train_dynamic_gesture.py  # 时序手势模型训练
```

## 开发规范

- 新增手势类别需同步更新 `labels.py` 中的 `STATIC_EVENT_LABELS` 或 `TEMPORAL_GESTURE_CLASSES`
- 配置变更通过 `configs/gesture_runtime.yaml` 管理，对应 `config.py` dataclass
- 后端扩展通过继承 `backends/base.py` 的 `HandDetectorBackend` 基类实现
- 时序模型特征维度变更需同步更新 `temporal_model.py` 中的 `FEATURE_DIM` 常量

## Current Status

初次分析项目结构，创建了 `CLAUDE.md` 和 `AGENTS.md`，完成项目上下文建立。

## Recent Changes

- 2026-06-11：初始化 `AGENTS.md` 和 `CLAUDE.md`，完整梳理四层管道架构和配置体系

## Next TODO

- 根据实际开发需求扩展手势类别或优化识别逻辑
- 考虑为时序模型补充评估脚本
- 可考虑将 `train.py` 内嵌参数迁移至配置文件

## Open Issues

- `train.py` 的训练参数（模型类型、数据集路径）硬编码在脚本顶部，不如 `video_demo.py` 的命令行参数风格一致
- `mediapipe==0.10.11` 锁定 Python 3.8，限制了依赖升级空间
- Legacy 后端权重路径在 `configs/gesture_runtime.yaml` 中默认为绝对路径，跨机器需手动修改

## Architecture Decisions

- 四层管道（Backend/Tracker/Recognizer/Visualization）解耦了检测和识别，便于独立替换后端
- 静态手势用角度约束规则（无需额外模型），动态手势用 Transformer 时序模型（需训练数据）
- `auto` 后端策略优先 MediaPipe，保持零权重文件启动的开发体验
