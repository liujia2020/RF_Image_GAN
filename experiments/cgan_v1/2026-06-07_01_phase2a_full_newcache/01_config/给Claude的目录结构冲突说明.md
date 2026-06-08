# 给 Claude 的目录结构冲突说明

时间：2026-06-07 14:02:18 +0800

## 冲突点

Claude 第十二工作单元建议正式 run 目录直接包含：

```text
README/config.yaml/run_manifest.json/train脚本/metrics/figures/logs/checkpoints.txt
```

但项目最新文档 `docs/Codex参考/执行规范.md` 明确要求每个正式运行按三段式组织：

```text
experiments/<track>/runs/<YYYY-MM-DD_short_name>/
├── 01_config/          训练前配置
├── 02_train/           训练产物
└── 03_validate/        验证结果
```

## 本次执行决定

按项目文档执行，正式训练目录使用：

```text
experiments/cgan_v1/runs/2026-06-07_01_phase2a_full_newcache/
├── 01_config/
├── 02_train/
└── 03_validate/
```

## 原因

用户已明确要求训练文件夹必须按项目文档规则存放，不采用扁平结构。后续 Claude 指令请按三段式目录组织书写。

## 新增执行事实：num_workers=2 数据加载瓶颈

时间：2026-06-07 15:09 +0800

按重订指令，`config.yaml` 已设置 `num_workers=2`，并通过 Cell1 自检。随后用 `nbconvert --execute --inplace` 对 `train.ipynb` 做一次性 Run All。

运行进入 Cell5 后约 9 分钟仍未完成 epoch 1，`metrics/stability.csv` 没有写出首行。观察到两个 DataLoader worker 均接近 100% CPU，RSS 分别约 20GB~21GB，GPU utilization 约 6%~8%，说明瓶颈在数据加载/预取而非 GPU 计算。

Codex 已停止运行并保存停止记录：

`02_train/logs/本次notebook停止记录_20260607_1509/停止原因.md`

需要 Claude/用户决策：是否允许将 `num_workers` 改回 `0`，或调整 reader/batch 传输策略。Codex 不自行改动 frozen 配置。
