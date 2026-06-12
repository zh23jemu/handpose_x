#!/bin/bash
#
# HandPose X 分析任务 Slurm 脚本
#
# 用途：
# 1. 生成 Wing Loss 分段结构和梯度行为图；
# 2. 统计数据集关键点坐标分布、bbox 尺度和增强前后覆盖；
# 3. 基于真实权重评估 NME、PCK 曲线、PCK@0.2 和整体识别准确率；
# 4. 通过 50 次重复测量统计单帧推理延迟、手势识别延迟、指令响应延迟和端到端延迟。
#
# 运行前请按服务器实际路径覆盖下面几个变量，例如：
#   sbatch --export=ALL,DATASET_DIR=/path/to/handpose_datasets/,MODEL_PATH=/path/to/best.pth,SOURCE=/path/to/gesture.mp4 slurm/run_analysis_suite.slurm
#
# 如果只做 CPU 文档分析，可提交时覆盖分区：
#   sbatch --partition=defq slurm/run_analysis_suite.slurm
#
# 如果使用 GPU 评估/测速，默认使用 gpu 分区；短任务可尝试：
#   sbatch --qos=shortjobs --time=01:00:00 slurm/run_analysis_suite.slurm

#SBATCH --job-name=handpose-analysis
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=analysis_outputs/slurm/%x-%j.out
#SBATCH --error=analysis_outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
DATASET_DIR="${DATASET_DIR:-handpose_datasets/}"
AUGMENTED_DATASET_DIR="${AUGMENTED_DATASET_DIR:-}"
MODEL_PATH="${MODEL_PATH:-Weight/resnet_101-size-256-best_model2.pth}"
MODEL_NAME="${MODEL_NAME:-resnet_101}"
SOURCE="${SOURCE:-image/image_2021-02-04_20-05-49.jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis_outputs/server_run}"
REPEAT="${REPEAT:-50}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.7}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" analysis_outputs/slurm

# 服务器使用 environment modules 管理 Python。默认加载集群上可用的
# python/3.11.7；如需切换到 python/3.11.2，可提交时覆盖：
#   sbatch --export=ALL,PYTHON_MODULE=python/3.11.2 slurm/run_analysis_suite.slurm
if command -v module >/dev/null 2>&1; then
  module load "$PYTHON_MODULE"
fi

# 始终使用项目本地虚拟环境，避免服务器系统 Python 版本和依赖污染结果。
if [ ! -x ".venv/bin/python" ]; then
  echo "未找到 .venv/bin/python，正在使用当前 Python module 创建项目虚拟环境。"
  python -m venv .venv
fi

.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements-torch-cu126.txt
.venv/bin/pip install -r requirements-gesture.txt

.venv/bin/python analysis_loss.py \
  --output-dir "$OUTPUT_DIR/loss"

DATASET_QUALITY_ARGS=(
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR/dataset_quality"
)
if [ -n "$AUGMENTED_DATASET_DIR" ]; then
  DATASET_QUALITY_ARGS+=(--augmented-dataset-dir "$AUGMENTED_DATASET_DIR")
fi

.venv/bin/python analysis_dataset_quality.py "${DATASET_QUALITY_ARGS[@]}"

.venv/bin/python evaluate_keypoint_accuracy.py \
  --dataset-dir "$DATASET_DIR" \
  --model-path "$MODEL_PATH" \
  --model "$MODEL_NAME" \
  --device auto \
  --output-dir "$OUTPUT_DIR/keypoint_accuracy"

.venv/bin/python benchmark_runtime.py \
  --source "$SOURCE" \
  --legacy-model-path "$MODEL_PATH" \
  --device auto \
  --repeat "$REPEAT" \
  --output-dir "$OUTPUT_DIR/runtime_benchmark"

echo "HandPose X 分析任务完成，结果目录：$OUTPUT_DIR"
