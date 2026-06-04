# 实验运行手册

最后更新：2026-06-05 01:57:34 +08:00

## 1. 目的

本手册规定 RF-to-volume / cGAN 实验如何创建、记录、审核和关闭。它同时服务于人类阅读和 AI 接力。

正式训练 run 不能临时开跑。必须先有 run folder 和冻结 config。

## 2. run folder 标准

每个正式 run 一个独立文件夹：

```text
experiments/<track>/runs/<YYYY-MM-DD_short_name>/
  README.md
  config.yaml
  run_manifest.json
  train.ipynb
  validate.ipynb
  metrics/
  figures/
  nii/
  logs/
  checkpoints.txt
  verdict.md
```

大体积二进制文件可以放在 git 外，但路径必须写进 `run_manifest.json` 和 `README.md`。

## 3. 文件职责

`README.md`：

- 为什么要做这个 run。
- 单变量假设是什么。
- 预期看到什么结果。
- 当前状态和最终状态。

`config.yaml`：

- 训练配置的唯一事实来源。
- 一旦训练开始，config 不允许原地改；要改就新建 run。

`run_manifest.json`：

- 时间戳。
- 任务名。
- git commit。
- git dirty status。
- config hash。
- dataset 路径。
- checkpoint 路径。
- Python / CUDA / GPU 环境。

`train.ipynb`：

- startup check。
- 完整 config 打印。
- loss 曲线或表格。
- OOM、中断、恢复训练等情况。

`validate.ipynb`：

- NIfTI 导出。
- 指标生成。
- 固定对比图生成。
- 指向 Slicer 材料的位置。

`verdict.md`：

- 人眼视觉状态。
- 指标状态。
- 最终决定。
- 失败原因。
- 下一步动作。

## 4. 生命周期

1. 计划：写清 run 目的和假设。
2. 冻结配置：写 `config.yaml`，记录 git 状态。
3. 训练：只从冻结 config 启动。
4. 导出材料：NIfTI、figures、metrics。
5. 人眼审查：用户在 Slicer 中检查。
6. 收尾：写 `verdict.md`，更新 `EXPERIMENTS_LOG.md`，再 commit。

## 5. 时间戳规则

每个 run 或诊断脚本开头都要打印：

```text
YYYY-MM-DD HH:MM:SS +08:00 | TASK: <task name>
```

同一个时间戳或生成时间戳必须写入 `run_manifest.json`。

## 6. 失败记录规则

失败是一等结果，必须认真记录：

- 原本期待什么。
- 实际发生什么。
- 如何发现的。
- 是指标失败、视觉失败，还是两者都失败。
- 是否推翻原假设。

不要把失败写成模糊的“还需提升”。如果失败推翻了路线，就明确写。

## 7. AI 角色边界

Codex 可以：

- 写代码、配置、notebook。
- 跑指标、导图、导 NIfTI。
- 汇总事实和指出风险。

Codex 不可以：

- 未经用户 Slicer 审查就宣布图像质量通过。
- 把代理指标当最终质量。
- 没有 run folder 和冻结 config 就启动正式训练。

Claude / Claude Code 可以：

- 做策略判断。
- 审核方案和结果。
- 给 Codex 下精确执行指令。

Claude / Claude Code 不直接修改项目文件，除非用户显式改变角色分工。

## 8. commit 规则

一个有意义动作，一个 commit。commit message 必须准确描述实际改动。

不要把训练代码、文档修正、大文件清理混在一个 commit，除非事先明确计划。

## 9. smoke test 规则

smoke test 不是正式实验。它只回答：

- 能不能跑。
- 是否 NaN。
- 是否 OOM。
- G/D 是否一边倒崩溃。

smoke test 通过不代表图像质量进步，也不触发验证关卡。
