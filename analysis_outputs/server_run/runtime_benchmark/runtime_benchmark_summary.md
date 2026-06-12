# 推理性能分析摘要

- 重复测量次数：50，预热次数：5。
- `backend_latency_summary.csv` 包含单帧推理延迟、手势识别延迟、指令响应延迟和系统端到端总延迟。
- `model_latency_size_summary.csv` 包含不同模型规格的参数量、估算体积和单帧前向延迟。
- 如果 `command_response_mean_ms` 为 0，表示测试输入没有触发可统计的手势事件，需要换成包含动作的视频样本。