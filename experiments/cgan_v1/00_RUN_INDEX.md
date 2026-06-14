# cGAN v1 Run 索引

更新时间：2026-06-10

## 当前正式 run

| 顺序 | 目录 | 状态 | 说明 |
|---|---|---|---|
| 最新 | `2026-06-10_01_bmode_supervised_v1/` | 已完成 50 epoch | B-mode 单通道纯监督基线，loss = 0.84*(1-SSIM3D)+0.16*L1；训练集改善，fixed-val 未改善 |
| 上一轮 | `2026-06-08_03_phase2a_normcarrier_v1/` | 用户中止于 epoch 25 | carrier 与 struct 使用同一 label 派生尺度归一化；三损失可比，但 D_fake_score 仍约 0.5 |
| 更早 | `2026-06-08_02_phase2a_advstruct_v1/` | 用户中止于 epoch 19 | 去掉 carrier 后仍观察到 pred≈baseline；D_fake_score 贴近 0.5 |
| 更早 | `2026-06-08_01_phase2a_normstruct_v1/` | 已完成 50 epoch | struct 用 label 均值包络尺度归一化，暴露 carrier 主导 |
| 更早 | `2026-06-07_01_phase2a_full_newcache/` | 已完成 50 epoch | 新 cache + 未归一化 struct，暴露 struct 绝对主导 |

## 归档 run

旧结构、smoke、早期 pilot 已移入：

```text
00_archive_legacy_runs/
```

这些目录不再作为当前正式训练入口，只保留历史参考。

## 下一次正式训练命名

如果仍在 2026-06-10 开新训练，使用：

```text
2026-06-10_02_<short_name>/
```

如果日期变化，则从当天 `01` 重新开始：

```text
YYYY-MM-DD_01_<short_name>/
```
