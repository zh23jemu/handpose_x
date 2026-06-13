# AGENTS.md — HandPose X

## 项目目标

手部姿态检测与动态手势识别系统。包含：
1. 基于 CNN 骨干网络的手部21关键点检测模型训练
2. 模块化四层手势识别运行时（Backend→Tracker→Recognizer→Visualization）

## 技术栈

- Python 3.11（服务器推荐版本；mediapipe==0.10.11 支持 Python 3.11）
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

已完成项目文档与当前结构复核，并新增面向报告/论文的分析套件；项目推荐运行版本已根据服务器可用模块调整为 Python 3.11，代码已推送到 public GitHub 仓库 `https://github.com/zh23jemu/handpose_x`。
- 2026-06-12：本地已 `git pull origin master` 拉回服务器分析结果提交 `ba0909c`，`analysis_outputs/server_run/` 已同步到当前工作区，仓库状态干净。
- 2026-06-13：用户已提供动态手势视频到 `video/`；本地完成 `video1.mp4` 与 `video2.mp4` 的 MediaPipe 动态视频运行时补测，其中 `video1.mp4` 触发 4 个事件，可作为动态响应延迟补充结果。

## Recent Changes

- 2026-06-11：初始化 `AGENTS.md` 和 `CLAUDE.md`，完整梳理四层管道架构和配置体系
- 2026-06-11：复核 `README.md`、`PROJECT_STATUS.md`、`CLAUDE.md`、`docs/gesture_pipeline.md`、运行时/训练配置和依赖清单，确认当前主要风险仍集中在依赖版本、硬编码路径、缺少动态手势数据集和训练评估脚本。
- 2026-06-11：新增 `analysis_loss.py`、`analysis_dataset_quality.py`、`evaluate_keypoint_accuracy.py`、`benchmark_runtime.py` 和 `slurm/run_analysis_suite.slurm`，覆盖用户要求的损失函数、数据质量、预测精度和推理性能分析。
- 2026-06-11：运行 Wing Loss 分析并生成 `analysis_outputs/loss/` 下的曲线图、梯度图、CSV 和摘要。
- 2026-06-11：根据服务器可用 module（`python/3.11.7` 为默认版本）和 `mediapipe==0.10.11` 兼容性，将项目推荐 Python 版本调整为 3.11，并更新 `.python-version`。
- 2026-06-11：曾将旧 Python 3.14 `.venv` 非破坏性移动到 `.venv_backup_py314`，并在本地 Python 3.10.0 环境验证过 `mediapipe`、`torch`、`cv2` 可导入；服务器侧后续以 Python 3.11 module 为准重新创建 `.venv`。
- 2026-06-11：创建 GitHub public 仓库 `zh23jemu/handpose_x` 并推送代码；本地生成 `data_release/Weight.zip` 与 `data_release/handpose_datasets.zip`，准备用 GitHub Release 同步服务器运行资产。
- 2026-06-11：服务器验证发现默认依赖安装会得到 `torch 2.12.0+cu130`，而集群驱动为 CUDA 12.8 兼容级别导致 `torch.cuda.is_available()` 为 `False`；新增 `requirements-torch-cu126.txt` 并让 Slurm 脚本优先安装 CUDA 12.6 wheel。
- 2026-06-12：服务器 Slurm 快速验证中，损失函数分析和数据集质量分析已输出；关键点精度评估失败原因是默认 `MODEL_NAME=ReXNetV1` 与 `resnet_101` 权重不匹配，且 checkpoint 使用 `model_state_dict` 包装。已修复 checkpoint 兼容并将 Slurm 默认模型改为 `resnet_101`。
- 2026-06-12：服务器短任务已生成关键点精度和运行时性能输出；其中 `legacy` 后端因使用 ResNet 权重加载 ReXNet 结构被跳过。Slurm 默认后端改为仅测 `mediapipe`，Legacy 测速需显式提供 ReXNet 权重并设置 `BACKENDS=legacy`。
- 2026-06-12：完整数据集 `handpose_datasets_v1` 已在服务器跑通，样本数 49062，PCK@0.2 为 0.97049，overall NME 为 0.05595；动态手势视频任务因 `SOURCE=/path/to/gesture_video.mp4` 为占位路径失败，已为 Slurm 脚本增加 DATASET/MODEL/SOURCE 输入路径预检查，并将默认 CPU 数调为 2。
- 2026-06-12：服务器已成功推送完整分析结果，当前本地已同步损失、数据集质量、关键点精度和运行时性能四类正式输出。
- 2026-06-13：新增本地动态视频补测输出 `analysis_outputs/video1_runtime_benchmark/` 与 `analysis_outputs/video2_runtime_benchmark/`；`video1.mp4` 的指令响应平均延迟为 45.47 ms，端到端平均延迟为 41.87 ms，`video2.mp4` 未触发事件。

## Next TODO

- 根据实际开发需求扩展手势类别或优化识别逻辑
- 考虑为时序模型补充评估脚本
- 可考虑将 `train.py` 内嵌参数迁移至配置文件
- 统一文档中的运行命令为项目本地 `.venv` 调用方式，避免误用系统 Python。
- 在服务器准备真实数据集、权重和动态手势视频后，通过 Slurm 运行完整分析套件，产出 PCK@0.2、整体识别准确率和 50 次重复测量延迟。
- 服务器拉取代码后，从 GitHub Release 下载 `Weight.zip` 和 `handpose_datasets.zip`，解压到项目根目录再执行分析/训练任务。
- 如需形成最终交付材料，补一份汇总文档或报告，把 `analysis_outputs/server_run/` 中的核心图表、CSV 和 JSON 结果串起来。
- 若要补齐动态手势正式指标，需要上传真实动态手势视频后重跑 `SOURCE` 路径对应的延迟分析。
- 可将 `video1.mp4` 的动态补测指标纳入最终报告，同时说明该结果来自本地 CPU 环境，服务器 GPU 指标仍以 `analysis_outputs/server_run/` 为准。

## Open Issues

- `train.py` 的训练参数（模型类型、数据集路径）硬编码在脚本顶部，不如 `video_demo.py` 的命令行参数风格一致
- `mediapipe==0.10.11` 已锁定版本，当前推荐 Python 3.11；升级 Python 或 MediaPipe 前需重新验证兼容性
- Legacy 后端权重路径在 `configs/gesture_runtime.yaml` 中默认为绝对路径，跨机器需手动修改
- `PROJECT_STATUS.md` 和 `docs/gesture_pipeline.md` 中仍有部分示例命令使用 `python ...`，与当前必须使用项目 `.venv` 的执行规范不完全一致。
- 本地仓库未包含大体积真实数据集和权重文件，因此关键点精度、PCK@0.2、后端对比和指令响应延迟需要在服务器或具备数据/权重的环境中实测。
- `pip show mediapipe` 和 `importlib.metadata` 显示发行包版本为 `0.10.11`，但 `mediapipe.__version__` 返回 `0.10.10`，属于包内部版本常量与发行元数据不一致；后续记录环境版本时应优先使用包元数据。
- 服务器 GPU 环境不要使用 `torch 2.12.0+cu130`；当前应卸载后按 `requirements-torch-cu126.txt` 安装 CUDA 12.6 兼容 PyTorch。
- `Weight/resnet_101-size-256-best_model2.pth` 必须搭配 `MODEL_NAME=resnet_101` 评估；如果换用 ReXNet 权重，再覆盖为 `MODEL_NAME=ReXNetV1`。
- 默认 Release 权重不适合 Legacy 后端；没有 ReXNet 权重时，运行时性能报告应以 MediaPipe 后端和模型规格前向延迟为主。
- 动态手势响应延迟不能使用占位路径或静态图片得出正式结论，必须上传真实动态手势视频并将 `SOURCE` 指向该文件。
- 当前正式结果已齐全，但动态手势/指令响应延迟的严谨结论仍依赖真实视频输入；若继续交付，需明确说明这是待补测项。
- 当前动态视频补测在本地 CPU 环境完成，MediaPipe 链路可用，但与服务器 CUDA 环境下的模型规格测速不可直接横向比较。

## Architecture Decisions

- 四层管道（Backend/Tracker/Recognizer/Visualization）解耦了检测和识别，便于独立替换后端
- 静态手势用角度约束规则（无需额外模型），动态手势用 Transformer 时序模型（需训练数据）
- `auto` 后端策略优先 MediaPipe，保持零权重文件启动的开发体验
- 分析能力独立放在根目录脚本与 `analysis_tools/` 辅助包中，不侵入现有训练、推理和实时演示主链路。
- 代码通过 GitHub 仓库同步，数据集和权重通过 GitHub Release 资产同步，避免大文件进入 Git 历史。
