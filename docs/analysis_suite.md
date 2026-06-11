# HandPose X 分析套件

本文档对应用户提出的四类分析需求，所有脚本默认从项目根目录执行，并且必须使用项目本地 `.venv`。

## 1. 损失函数分析

生成 Wing Loss、L1、L2 的损失曲线与梯度曲线：

```bash
.venv/bin/python analysis_loss.py --output-dir analysis_outputs/loss
```

输出：
- `loss_curve_comparison.png`
- `gradient_behavior_comparison.png`
- `loss_curve_values.csv`
- `loss_analysis_summary.md`

## 2. 数据集质量分析

分析 21 个关键点坐标分布、样本空间多样性、bbox 尺度分布，并可选对比增强前后效果：

```bash
.venv/bin/python analysis_dataset_quality.py \
  --dataset-dir handpose_datasets/ \
  --augmented-dataset-dir handpose_datasets_augmented/ \
  --output-dir analysis_outputs/dataset_quality
```

如果没有增强后落盘数据，可省略 `--augmented-dataset-dir`。

## 3. 模型预测精度分析

基于真实权重和验证集统计每个关键点 NME、PCK 曲线、PCK@0.2 与整体识别准确率：

```bash
.venv/bin/python evaluate_keypoint_accuracy.py \
  --dataset-dir handpose_datasets/ \
  --model-path Weight/resnet_101-size-256-best_model2.pth \
  --model ReXNetV1 \
  --output-dir analysis_outputs/keypoint_accuracy
```

输出：
- `per_keypoint_nme.csv`
- `per_keypoint_nme.png`
- `pck_curve.csv`
- `pck_curve.png`
- `keypoint_metrics.json`
- `keypoint_accuracy_summary.md`

其中 `keypoint_metrics.json` 内的 `pck_at_0_2` 即 PCK@0.2；`overall_accuracy` 当前按 PCK@0.2 口径记录。

## 4. 推理性能分析

对比 MediaPipe、Legacy 后端和不同关键点模型规格的延迟与模型体积：

```bash
.venv/bin/python benchmark_runtime.py \
  --source image/image_2021-02-04_20-05-49.jpg \
  --legacy-model-path Weight/resnet_101-size-256-best_model2.pth \
  --repeat 50 \
  --output-dir analysis_outputs/runtime_benchmark
```

输出：
- `backend_latency_summary.csv`：单帧推理延迟、手势识别延迟、指令响应延迟、系统端到端总延迟
- `model_latency_size_summary.csv`：模型参数量、估算模型体积、单帧前向延迟
- `backend_latency_comparison.png`
- `model_size_latency_tradeoff.png`
- `runtime_benchmark_metrics.json`

注意：如果输入源是静态图片且没有触发手势事件，`command_response_mean_ms` 会为 0。要得到“指令响应延迟”，应使用包含真实动态手势的视频，并保持 `--repeat 50`。

## Slurm 运行

服务器上可使用：

```bash
sbatch --export=ALL,DATASET_DIR=/path/to/handpose_datasets/,MODEL_PATH=/path/to/best.pth,SOURCE=/path/to/gesture.mp4 slurm/run_analysis_suite.slurm
```

GPU 任务默认使用 `gpu` 分区、`gpo-ifv7xx` 账号和 `normal` QOS。预计 1 小时内完成的短评测可覆盖：

```bash
sbatch --qos=shortjobs --time=01:00:00 slurm/run_analysis_suite.slurm
```
