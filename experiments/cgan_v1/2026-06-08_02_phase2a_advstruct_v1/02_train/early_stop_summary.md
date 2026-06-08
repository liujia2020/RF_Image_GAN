# 第十四工作单元提前停止摘要

时间：2026-06-08 12:30 +0800

- 状态：用户要求提前停止
- 最后完整 epoch：19
- 原计划 epoch：50
- NaN/Inf：未出现
- carrier：已跳过，`G_carrier_weighted=0`，`G_carrier_skipped=1`
- 用户观察：epoch10 三联图中 pred 仍接近 baseline，因此停止本轮并转给 Claude 决策

主要数据见：

- `metrics/stability.csv`
- `metrics/speckle_check.csv`
- `figures/*_epoch010_triplet_y16.png`
