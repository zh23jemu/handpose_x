# Wing Loss 分析摘要

- 分段阈值 `w=0.0600`，平滑参数 `epsilon=0.0100`。
- 当误差绝对值小于 `w` 时，Wing Loss 使用对数段，小误差梯度更大，适合强调关键点精细定位。
- 当误差绝对值不小于 `w` 时，Wing Loss 退化为线性段，降低异常大误差对整体训练的主导作用。
- 输出文件：`loss_curve_comparison.png`、`gradient_behavior_comparison.png`、`loss_curve_values.csv`。