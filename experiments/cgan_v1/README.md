# cGAN 训练运行目录规范

本目录只放 cGAN v1 的运行记录。正式训练必须使用三段式结构，并在日期后加入两位编号，便于判断顺序。

## 命名规则

```text
YYYY-MM-DD_NN_short_name/
```

- `YYYY-MM-DD`：创建或正式开跑日期。
- `NN`：当天第几个正式 run，从 `01` 开始递增。
- `short_name`：短说明，只写关键变量，例如 `phase2a_normstruct_v1`。

示例：

```text
2026-06-08_01_phase2a_normstruct_v1/
2026-06-08_02_phase2a_normstruct_carrierfix/
```

## 正式 run 结构

```text
YYYY-MM-DD_NN_short_name/
├── 01_config/      配置、manifest、运行说明
├── 02_train/       train.py、train.ipynb、metrics、figures、checkpoints、logs
└── 03_validate/    后续独立验证结果
```

## 文件原则

- `01_config/config.yaml` 是唯一参数源。
- `02_train/train.py` 放训练逻辑。
- `02_train/train.ipynb` 只做展示层，导入 `train.py`，不复制训练逻辑。
- 正式结果必须由 `train.ipynb` 一次性 Run All 产出。
- 新建 run 时必须先清空 notebook output，避免复制旧输出误导。
- 临时监控脚本、旧 lock、`__pycache__` 不作为正式产物保留。

## 归档区

旧结构、smoke、早期 pilot 放入：

```text
00_archive_legacy_runs/
```

归档区只保留历史参考，不作为当前正式训练入口。
